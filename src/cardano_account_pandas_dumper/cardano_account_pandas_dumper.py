""" Cardano Account To Pandas Dumper."""
import datetime
import itertools
from base64 import b64decode
from collections import defaultdict
from io import BytesIO
import os
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    cast,
)

import blockfrost.utils
import matplotlib as mpl
import matplotlib.pyplot as pyplot

from matplotlib.font_manager import FontProperties
from matplotlib.image import BboxImage
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle
from matplotlib.transforms import TransformedBbox
import numpy as np
import pandas as pd
from blockfrost import BlockFrostApi

CREATOR_STRING="https://github.com/pixelsoup42/cardano_account_pandas_dumper"



class AccountData:
    """Hold data retrieved from the API to allow checkpointing it."""

    LOVELACE_ASSET = "lovelace"

    def __init__(
        self,
        api: BlockFrostApi,
        staking_addresses: FrozenSet[str],
        to_block: Optional[int],
        include_rewards: bool,
    ) -> None:
        self.staking_addresses = staking_addresses
        self.to_block = to_block or int(api.block_latest().height - 1)
        block_last = api.block(self.to_block)
        block_after_last = api.block(self.to_block + 1)
        self.end_time = block_after_last.time
        self.end_epoch = block_last.epoch + 1
        self.rewards = pd.Series(
            name="Rewards",
            data=[
                (s_a, a_r)
                for s_a in self.staking_addresses
                for a_r in api.account_rewards(s_a, gather_pages=True)
                if a_r.epoch < self.end_epoch
            ]
            if include_rewards
            else [],
        )
        self.epochs = pd.Series(
            name="Epochs",
            data={
                e: api.epoch(e)
                for e in frozenset(
                    itertools.chain(
                        *[[r[1].epoch, r[1].epoch + 1] for r in self.rewards]
                    )
                )
            }
            if include_rewards
            else {},
        ).sort_index()
        self.pools = pd.Series(
            name="Pools",
            data={
                pool: api.pool_metadata(pool)
                for pool in frozenset([r[1].pool_id for r in self.rewards])
            }
            if include_rewards
            else {},
        ).sort_index()
        self.own_addresses = frozenset(
            [
                a.address
                for a in itertools.chain(
                    *[
                        api.account_addresses(s, gather_pages=True)
                        for s in self.staking_addresses
                    ]
                )
            ]
        )
        self.transactions = pd.Series(
            name="Transactions", data=self._transaction_data(api)
        )
        self.assets = pd.Series(
            name="Assets",
            data={
                a: api.asset(a)
                for a in frozenset(
                    [
                        a.unit
                        for tx_obj in self.transactions
                        for i in (tx_obj.utxos.inputs + tx_obj.utxos.outputs)
                        for a in i.amount
                    ]
                ).difference([self.LOVELACE_ASSET])
            },
        ).sort_index()

    def _transaction_data(
        self,
        api: BlockFrostApi,
    ) -> List[blockfrost.utils.Namespace]:
        result_list = []
        for tx_hash in frozenset(
            [
                outer_tx.tx_hash
                for addr in self.own_addresses
                for outer_tx in api.address_transactions(
                    addr,
                    to_block=self.to_block,
                    gather_pages=True,
                )
            ]
        ):
            transaction = api.transaction(tx_hash)
            transaction.utxos = api.transaction_utxos(tx_hash)
            transaction.utxos.nonref_inputs = [
                i for i in transaction.utxos.inputs if not i.reference
            ]
            transaction.metadata = api.transaction_metadata(tx_hash)
            transaction.redeemers = (
                api.transaction_redeemers(tx_hash) if transaction.redeemer_count else []
            )
            transaction.withdrawals = (
                api.transaction_withdrawals(tx_hash)
                if transaction.withdrawal_count
                else []
            )
            transaction.reward_amount = None

            result_list.append(transaction)
        return result_list


class AccountPandasDumper:
    """Hold logic to convert an instance of AccountData to a Pandas dataframe."""

    TRANSACTION_OFFSET = np.timedelta64(1000, "ns")
    OWN_LABEL = " own"
    OTHER_LABEL = "other"
    ADA_ASSET = " ADA"
    ADA_DECIMALS = 6
    METADATA_MESSAGE_LABEL = "674"
    METADATA_NFT_MINT_LABEL = "721"

    def __init__(
        self,
        data: AccountData,
        known_dict: Any,
        truncate_length: int,
        unmute: bool,
        detail_level: int,
    ):
        self.data = data
        self.truncate_length = truncate_length
        self.unmute = unmute
        self.detail_level = detail_level
        self.address_names = pd.Series(
            {a: " wallet" for a in self.data.own_addresses}
            | known_dict.get("addresses", {})
        )
        self.policy_names = pd.Series(known_dict.get("policies", {}))
        self.asset_names = pd.Series(
            {asset.asset: self._decode_asset_name(asset) for asset in self.data.assets}
        )
        self.asset_decimals = pd.Series(
            {
                asset.asset: np.longlong(asset.metadata.decimals or 0)
                if hasattr(asset, "metadata") and hasattr(asset.metadata, "decimals")
                else 0
                for asset in self.data.assets
            }
            | {self.ADA_ASSET: self.ADA_DECIMALS}
        )
        self.muted_policies = pd.Series(known_dict.get("muted_policies", []))
        self.pinned_policies = pd.Series(known_dict.get("pinned_policies", []))
        self.scripts = pd.Series(known_dict.get("scripts", {}))
        self.labels = pd.Series(known_dict.get("labels", {}))
        transactions = pd.concat(
            [
                self.data.transactions,
                pd.Series([self.reward_transaction(r) for r in self.data.rewards]),
            ]
        )

        transactions.index = pd.Index(
            [self.extract_timestamp(t) for t in transactions],
            dtype="datetime64[ns]",
        )
        transactions.sort_index(inplace=True)
        assert len(transactions) == len(self.data.transactions) + len(self.data.rewards)
        self.transactions = transactions

    def _truncate(self, value: str) -> str:
        return (
            value
            if not self.truncate_length or len(value) <= self.truncate_length
            else ("..." + value[-self.truncate_length :])
        )

    def _decode_asset_name(self, asset: blockfrost.utils.Namespace) -> str:
        if (
            hasattr(asset, "metadata")
            and hasattr(asset.metadata, "name")
            and asset.metadata.name
        ):
            return asset.metadata.name
        asset_hex_name = asset.asset.removeprefix(asset.policy_id)
        try:
            return bytes.fromhex(asset_hex_name).decode()
        except UnicodeDecodeError:
            return f"{self._format_policy(asset.policy_id)}@{self._truncate(asset_hex_name)}"

    @staticmethod
    def _is_hex_number(num: Any) -> bool:
        return isinstance(num, str) and not bool(
            frozenset(num.lower().removeprefix("0x")).difference(
                frozenset("0123456789abcdef")
            )
        )

    def _munge_metadata(self, obj: blockfrost.utils.Namespace) -> Any:
        if isinstance(obj, blockfrost.utils.Namespace):
            result = {}
            for att in dir(obj):
                if att.startswith("_") or att in (
                    "to_json",
                    "to_dict",
                ):
                    continue
                hex_name = self._is_hex_number(att)
                value = getattr(obj, att)
                if (hex_name and self._is_hex_number(value)) and not self.unmute:
                    continue
                value = self._munge_metadata(value)
                if value:
                    out_att = self._truncate(att) if hex_name else att
                    if out_att == "msg":
                        return (
                            " ".join(value) if isinstance(value, list) else str(value)
                        )
                    result[out_att] = value
            return result
        elif self._is_hex_number(obj) and not self.unmute:
            return {}
        else:
            return obj

    def _parse_nft_mint(self, meta: blockfrost.utils.Namespace) -> str:
        meta_dict = meta.to_dict()
        result = "NFT Mint:"
        for policy, _v in meta_dict.items():
            if policy == "version":
                continue
            result += f"{self._format_policy(policy)}:"
            for asset_name in _v.to_dict().keys():
                result += f"{asset_name} "
        return result

    def _format_message(self, tx_obj: blockfrost.utils.Namespace) -> str:
        result: List[str] = []
        for metadata_key in tx_obj.metadata:
            if metadata_key.label == self.METADATA_NFT_MINT_LABEL:
                label = None
                val = self._parse_nft_mint(metadata_key.json_metadata)
            else:
                if metadata_key.label == self.METADATA_MESSAGE_LABEL:
                    label = None
                else:
                    label = self.labels.get(
                        metadata_key.label, self._truncate(metadata_key.label)
                    )
                val = self._munge_metadata(metadata_key.json_metadata)
            if (
                self._is_hex_number(label)
                and (not val or self._is_hex_number(val))
                and not self.unmute
            ):
                continue
            if label:
                result.extend([label, ":"])
            result.append(str(val))
        redeemer_scripts: Dict[str, Set] = defaultdict(set)
        for redeemer in tx_obj.redeemers:
            if redeemer.purpose == "spend":
                redeemer_scripts[redeemer.purpose].add(
                    self._format_script(redeemer.script_hash)
                )
            elif redeemer.purpose == "mint":
                redeemer_scripts[redeemer.purpose].add(
                    self._format_policy(redeemer.script_hash)
                )
            else:
                redeemer_scripts[redeemer.purpose].add(
                    self._truncate(redeemer.redeemer_data_hash)
                )
        for k, redeemer_script in redeemer_scripts.items():
            result.extend([k, str(redeemer_script)])
        if not result and all(
            [
                utxo.address in self.data.own_addresses
                for utxo in tx_obj.utxos.nonref_inputs + tx_obj.utxos.outputs
            ]
        ):
            result = ["(internal)"]
        return " ".join(result)

    def _format_script(self, script: str) -> str:
        return self.scripts.get(script, self._truncate(script))

    def _format_policy(self, policy: str) -> Optional[str]:
        return self.policy_names.get(policy, self._truncate(policy))

    @classmethod
    def extract_timestamp(
        cls, transaction: blockfrost.utils.Namespace
    ) -> np.datetime64:
        """Returns timestamp of transaction."""
        return np.datetime64(
            datetime.datetime.fromtimestamp(transaction.block_time)
        ) + (int(transaction.index) * cls.TRANSACTION_OFFSET)

    def _drop_muted_assets(self, balance: pd.DataFrame) -> None:
        # Drop assets that only touch foreign addresses
        assets_to_drop = frozenset(
            # Assets that touch other addresses
            x[:1]
            for x in balance.xs(self.OTHER_LABEL, level=1, axis=1).columns
        ).union(
            # Assets with muted policies
            frozenset(
                [
                    asset.asset
                    for asset in self.data.assets
                    if any(self.muted_policies == asset.policy_id)
                ]
            )
        ) - frozenset(
            # Assets that touch own addresses
            x[:1]
            for x in balance.xs(self.OWN_LABEL, level=1, axis=1).columns
        ).union(
            # Assets with pinned policies
            frozenset(
                [
                    asset.asset
                    for asset in self.data.assets
                    if any(self.pinned_policies == asset.policy_id)
                ]
            )
        )
        balance.drop(assets_to_drop, axis=1, inplace=True)

    def reward_transaction(
        self, reward: Tuple[str, blockfrost.utils.Namespace]
    ) -> blockfrost.utils.Namespace:
        """Build reward pseudo-transaction for tuple (staking_addr, reward)."""
        result = blockfrost.utils.Namespace()
        result.tx_hash = None
        result.metadata = [
            blockfrost.utils.Namespace(
                label=self.METADATA_MESSAGE_LABEL,
                json_metadata="Reward: "
                + f"{reward[1].type} - {self.data.pools[reward[1].pool_id].name}"
                + f" -  {reward[0]} - {reward[1].epoch}",
            )
        ]
        result.reward_amount = reward[1].amount
        result.reward_address = reward[0]
        epoch = self.data.epochs[
            reward[1].epoch + 1
        ]  # Time is right before start of next epoch.
        result.block_time = epoch.start_time
        result.index = -1
        result.fees = "0"
        result.deposit = "0"
        result.asset_mint_or_burn_count = 0
        result.redeemers = []
        result.hash = None
        result.withdrawals = []
        result.utxos = blockfrost.utils.Namespace()
        result.utxos.inputs = []
        result.utxos.outputs = []
        result.utxos.nonref_inputs = []
        return result

    def _column_key(
        self,
        utxo,
        amount,
        raw_values: bool,
    ):
        # Index: (asset_id, own, address_name)
        return (
            amount.unit if amount.unit != self.data.LOVELACE_ASSET else self.ADA_ASSET,
            self.OWN_LABEL
            if utxo.address in self.data.own_addresses
            else self.OTHER_LABEL,
            self._truncate(utxo.address)
            if raw_values
            else self.address_names.get(
                utxo.address,
                self.OTHER_LABEL,
            ),
        )

    def _transaction_balance(
        self,
        transaction: blockfrost.utils.Namespace,
        raw_values: bool,
    ) -> Any:
        result: MutableMapping[Tuple, np.longlong] = defaultdict(lambda: np.longlong(0))
        result[(self.ADA_ASSET, self.OTHER_LABEL, " fees")] += np.longlong(
            transaction.fees
        )
        result[(self.ADA_ASSET, self.OWN_LABEL, " deposit")] += np.longlong(
            transaction.deposit
        )
        if transaction.reward_amount:
            result[(self.ADA_ASSET, self.OTHER_LABEL, " rewards")] -= np.longlong(
                transaction.reward_amount
            )
            result[
                (
                    self.ADA_ASSET,
                    self.OWN_LABEL,
                    f" withdrawals-{self._truncate(transaction.reward_address)}",
                )
            ] += np.longlong(transaction.reward_amount)
        for _w in transaction.withdrawals:
            result[
                (
                    self.ADA_ASSET,
                    self.OWN_LABEL,
                    f" withdrawals-{self._truncate(_w.address)}",
                )
            ] -= np.longlong(_w.amount)
        for utxo in transaction.utxos.nonref_inputs:
            if not utxo.collateral or not transaction.valid_contract:
                for amount in utxo.amount:
                    result[self._column_key(utxo, amount, raw_values)] -= np.longlong(
                        amount.quantity
                    )

        for utxo in transaction.utxos.outputs:
            for amount in utxo.amount:
                result[self._column_key(utxo, amount, raw_values)] += np.longlong(
                    amount.quantity
                )

        sum_by_asset: MutableMapping[str, np.longlong] = defaultdict(
            lambda: np.longlong(0)
        )
        for key, value in result.items():
            sum_by_asset[key[0]] += value
        sum_by_asset = {k: v for k, v in sum_by_asset.items() if v}
        assert (
            self.ADA_ASSET not in sum_by_asset
            and len(sum_by_asset) == transaction.asset_mint_or_burn_count
        ), (
            f"Unbalanced transaction: {transaction.hash if transaction.hash else '-'} : "
            + f"{self._format_message(transaction)} : "
            + str(
                {
                    (
                        f"{self._format_policy(self.data.assets[k].policy_id)}@"
                        + f"{self._decode_asset_name(self.data.assets[k])}"
                    ): v
                    for k, v in sum_by_asset.items()
                    if v != np.longlong(0)
                }
            )
        )
        return result

    def make_balance_frame(
        self,
        with_total: bool,
        raw_values: bool,
        text_cleaner: Callable = lambda x: x,
    ):
        """Make DataFrame with transaction balances."""
        balance = pd.DataFrame(
            data=[self._transaction_balance(x, raw_values) for x in self.transactions],
            index=self.transactions.index,
            dtype="Int64",
        )
        balance.columns = pd.MultiIndex.from_tuples(balance.columns)
        balance.sort_index(axis=1, level=0, sort_remaining=True, inplace=True)
        if not self.unmute:
            self._drop_muted_assets(balance)

        if self.detail_level == 1:
            group: Tuple = (0, 1)
        elif raw_values:
            group = (0, 1, 2)
        else:
            group = (0, 2)
        balance = balance.T.groupby(level=group).sum(numeric_only=True).T
        balance[balance == 0] = pd.NA
        if with_total:
            balance = pd.concat(
                [
                    balance,
                    pd.DataFrame(
                        data=[[balance[column].sum() for column in balance.columns]],
                        columns=balance.columns,
                        index=[
                            balance.index.max() + self.TRANSACTION_OFFSET,
                        ],
                    ),
                ]
            )

        balance = pd.concat(
            [
                balance[c]
                .mul(
                    np.float_power(
                        10,
                        np.negative(self.asset_decimals[c[0]]),
                    )
                )
                .round(self.asset_decimals[c[0]])
                for c in balance.columns
            ],
            axis=1,
        )

        if not raw_values:
            balance.columns = pd.MultiIndex.from_tuples(
                [
                    (text_cleaner(self.asset_names.get(c[0], c[0])),)
                    + cast(tuple, c)[1:]
                    for c in balance.columns
                ]
            )
            balance.sort_index(axis=1, level=0, sort_remaining=True, inplace=True)
        if self.detail_level == 1:
            return balance.xs(self.OWN_LABEL, level=1, axis=1)
        else:
            return balance

    def make_transaction_frame(
        self,
        raw_values: bool,
        with_total: bool = True,
        text_cleaner: Callable = lambda x: x,
    ) -> pd.DataFrame:
        """Build a transaction spreadsheet."""

        msg_frame = pd.DataFrame(
            data=[
                {"hash": x.hash, "message": text_cleaner(self._format_message(x))}
                for x in self.transactions
            ],
            index=self.transactions.index,
            dtype="string",
        )
        if with_total:
            msg_frame = pd.concat(
                [
                    msg_frame,
                    pd.DataFrame(
                        data=[["", "Total"]],
                        columns=msg_frame.columns,
                        index=[
                            msg_frame.index.max() + self.TRANSACTION_OFFSET,
                        ],
                    ),
                ]
            )
        balance_frame = self.make_balance_frame(
            with_total=with_total, text_cleaner=text_cleaner, raw_values=raw_values
        )
        if self.detail_level > 1:
            msg_frame.columns = pd.MultiIndex.from_tuples(
                [
                    (c,) + (len(balance_frame.columns[0]) - 1) * ("",)
                    for c in msg_frame.columns
                ]
            )
        assert len(msg_frame) == len(
            balance_frame
        ), f"Frame lengths do not match {msg_frame=!s} , {balance_frame=!s}"
        joined_frame = pd.concat(objs=[msg_frame, balance_frame], axis=1)
        return joined_frame

    class _ImageLegendHandler(HandlerBase):
        def __init__(self, color, asset_id: str,
                     asset_obj: Optional[blockfrost.utils.Namespace]) -> None:
            self.asset_id=asset_id
            self.asset_obj=asset_obj
            self.color=color
            super().__init__()

        def create_artists(
            self,
            legend,
            orig_handle,
            xdescent,
            ydescent,
            width,
            height,
            fontsize,
            trans,
        ):
            rectangle = Rectangle(xy=(xdescent, ydescent),
                                  width=width, height=height, color=self.color)
            if self.asset_id == AccountPandasDumper.ADA_ASSET:
                image_data=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ada_logo.webp")
            elif (self.asset_obj is not None
                  and hasattr(self.asset_obj , "metadata")
                  and hasattr(self.asset_obj.metadata, "logo")
                    and self.asset_obj.metadata.logo):
                image_data= BytesIO(b64decode(self.asset_obj.metadata.logo))
            else:
                image_data = None
            if image_data is not None:
                image = BboxImage(
                    TransformedBbox(
                        rectangle.get_bbox().expanded(0.7, 0.7), transform=trans
                    ),
                    interpolation="antialiased",
                    resample=True,
                )
                image.set_data(mpl.image.imread(image_data))

                self.update_prop(image, orig_handle, legend)

                return [rectangle, image]
            else:
                return [rectangle]

    def _plot_title(self):
        return f"Asset balances in wallet until block {self.data.to_block}."

    def plot_balance(self):
        """ Create a Matplotlib plot with the asset balance over time."""
        balance = self.make_balance_frame(with_total=False, raw_values=True).cumsum()
        balance.sort_index(
            axis=1,
            level=0,
            sort_remaining=True,
            inplace=True,
            key=lambda i: [self.asset_names.get(x, x) for x in i],
        )
        font_properties = FontProperties(size=mpl.rcParams['legend.fontsize'])
        plot_ax=pyplot.subplot2grid(shape=(1,7),loc=(0,0),colspan=6)
        legend_ax=pyplot.subplot2grid(shape=(1,7),loc=(0,6),colspan=1)

        plot=balance.plot(
            ax=plot_ax,
            logy=True,
            title=self._plot_title(),
            legend=False,
        )
        # Get font size
        text=pyplot.text(x=0,y=0,s="M", font_properties=font_properties)
        text_bbox=text.get_window_extent()
        text.remove()
        legend_ax.axis("off")
        for text in legend_ax.legend(
            plot.get_lines(),
            [self.asset_names.get(c, c) for c in balance.columns],
            handler_map={
                plot.get_lines()[i]:
                self._ImageLegendHandler(color=f"C{i}",
                                         asset_id=balance.columns[i],
                                         asset_obj=self.data.assets.get(balance.columns[i],None))
                for i in range(len(balance.columns))
            },

        ).get_texts():
            text.set(y=text.get_window_extent().y0 +
                     mpl.rcParams['legend.handleheight'] * text_bbox.height / 2)

    def get_graph_metadata(self, filename:str) -> Mapping :
        """Return graph metadata depending on file extension."""
        save_format=os.path.splitext(filename)[1].removeprefix('.')
        if not save_format:
            save_format= mpl.rcParams["savefig.format"]
        if save_format in ("svg","pdf"):
            return { "Creator": CREATOR_STRING, "Title": self._plot_title() }
        elif save_format=="png":
            return {"Software" : CREATOR_STRING,"Title": self._plot_title()}
        elif save_format in ("ps","eps"):
            return { "Creator": CREATOR_STRING }
        else:
            return {}
