""" Cardano Account To Pandas Dumper."""

import datetime
from functools import cache
import itertools
import os
from base64 import b64decode
from collections import defaultdict
from io import BytesIO
from typing import (
    Any,
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
import numpy as np
import pandas as pd
from blockfrost import BlockFrostApi
from matplotlib import pyplot
from matplotlib.axes import Axes
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE


class AccountData:
    """Hold data retrieved from the API to allow checkpointing it."""

    LOVELACE_ASSET = "lovelace"

    def __init__(
        self,
        *,
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
            data=(
                [
                    (s_a, a_r)
                    for s_a in self.staking_addresses
                    for a_r in api.account_rewards(s_a, gather_pages=True)
                    if a_r.epoch < self.end_epoch
                ]
                if include_rewards
                else []
            ),
        )
        self.epochs = pd.Series(
            name="Epochs",
            data=(
                {
                    e: api.epoch(e)
                    for e in frozenset(
                        itertools.chain(
                            *[[r[1].epoch, r[1].epoch + 1] for r in self.rewards]
                        )
                    )
                }
                if include_rewards
                else {}
            ),
        ).sort_index()
        self.pools = pd.Series(
            name="Pools",
            data=(
                {
                    pool: api.pool_metadata(pool)
                    for pool in frozenset([r[1].pool_id for r in self.rewards])
                }
                if include_rewards
                else {}
            ),
        ).sort_index()
        self.addresses = pd.Series(
            name="Addresses",
            data={
                s: api.account_addresses(s, gather_pages=True)
                for s in self.staking_addresses
            },
        ).sort_index()
        tx_hashes = frozenset(
            [
                outer_tx.tx_hash
                for addr in itertools.chain(*self.addresses.values)
                for outer_tx in api.address_transactions(
                    addr.address,
                    to_block=self.to_block,
                    gather_pages=True,
                )
            ]
        )
        mir_hashes = frozenset(
            itertools.chain(
                *[
                    [m.tx_hash for m in api.account_mirs(s_a, gather_pages=True)]
                    for s_a in self.staking_addresses
                ]
            )
            if include_rewards
            else []
        )
        self.transactions = pd.Series(
            name="Transactions",
            data={
                t.hash: t
                for t in self._transaction_data(api, tx_hashes.union(mir_hashes))
            },
        ).sort_index()
        self.assets = pd.Series(
            name="Assets",
            data={
                a: api.asset(a)
                for a in frozenset(
                    [
                        a.unit
                        for tx_obj in self.transactions.values
                        for i in (tx_obj.utxos.inputs + tx_obj.utxos.outputs)
                        for a in i.amount
                    ]
                ).difference([self.LOVELACE_ASSET])
            },
        ).sort_index()

    def _transaction_data(
        self, api: BlockFrostApi, tx_hashes: FrozenSet[str]
    ) -> List[blockfrost.utils.Namespace]:
        result_list = []
        for tx_hash in tx_hashes:
            transaction = api.transaction(tx_hash)
            if transaction.block_height <= self.to_block:
                transaction.utxos = api.transaction_utxos(tx_hash)
                transaction.utxos.nonref_inputs = [
                    i for i in transaction.utxos.inputs if not i.reference
                ]
                transaction.metadata = api.transaction_metadata(tx_hash)
                transaction.redeemers = (
                    api.transaction_redeemers(tx_hash)
                    if transaction.redeemer_count
                    else []
                )
                transaction.withdrawals = (
                    api.transaction_withdrawals(tx_hash)
                    if transaction.withdrawal_count
                    else []
                )
                transaction.reward_amount = None
                transaction.mirs = api.transaction_mirs(tx_hash)
                result_list.append(transaction)
        return result_list


class AccountPandasDumper:
    """Hold logic to convert an instance of AccountData to a Pandas dataframe."""

    # Transaction timestamp = block time + transaction index in block * TRANSACTION_OFFSET
    TRANSACTION_OFFSET = np.timedelta64(1000, "ns")

    OWN_LABEL = " own"
    OTHER_LABEL = "other"
    ADA_ASSET = " ADA"
    ADA_DECIMALS = 6
    METADATA_MESSAGE_LABEL = "674"
    METADATA_NFT_MINT_LABEL = "721"

    # Constants for graph output
    CREATOR_STRING = "https://github.com/pixelsoup42/cardano_account_pandas_dumper"

    def __init__(
        self,
        *,
        data: AccountData,
        known_dict: Any,
    ):
        self.data = data
        self.address_stake = pd.Series(
            {
                vi.address: k
                for k in self.data.addresses.keys()
                for vi in self.data.addresses[k]
            },
        )
        self.address_names = pd.Series(known_dict.get("addresses", {}))
        self.policy_names = pd.Series(known_dict.get("policies", {}))
        self.asset_decimals = pd.Series(
            {
                asset.asset: (
                    np.longlong(asset.metadata.decimals or 0)
                    if hasattr(asset, "metadata")
                    and hasattr(asset.metadata, "decimals")
                    else 0
                )
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

    @staticmethod
    def _truncate(value: str, truncate_length: int) -> str:
        return (
            value
            if not truncate_length or len(value) <= truncate_length
            else ("..." + value[-truncate_length:])
        )

    @cache  # pylint: disable=method-cache-max-size-none
    def _decode_asset_name(
        self, asset_id: str, truncate_length: int, with_policy: bool
    ) -> str:
        if asset_id == self.ADA_ASSET:
            return asset_id
        asset = self.data.assets[asset_id]
        if (
            hasattr(asset, "metadata")
            and hasattr(asset.metadata, "name")
            and asset.metadata.name
        ):
            name = asset.metadata.name
        asset_hex_name = asset.asset.removeprefix(asset.policy_id)
        try:
            decoded = bytes.fromhex(asset_hex_name).decode()
            name = ILLEGAL_CHARACTERS_RE.sub(
                lambda y: "".join(["\\x0" + hex(ord(y.group(0))).removeprefix("0x")]),
                decoded,
            )
        except UnicodeDecodeError:
            name = self._truncate(asset_hex_name, truncate_length)
            with_policy = True
        if with_policy:
            return f"{self._format_policy(asset.policy_id,truncate_length)}@{name}"
        else:
            return name

    @staticmethod
    def _is_hex_number(num: Any) -> bool:
        return isinstance(num, str) and not bool(
            frozenset(num.lower().removeprefix("0x")).difference(
                frozenset("0123456789abcdef")
            )
        )

    @classmethod
    def _munge_metadata(
        cls, *, obj: blockfrost.utils.Namespace, truncate_length: int, unmute: bool
    ) -> Any:
        if isinstance(obj, blockfrost.utils.Namespace):
            result = {}
            for att in dir(obj):
                if att.startswith("_") or att in (
                    "to_json",
                    "to_dict",
                ):
                    continue
                hex_name = cls._is_hex_number(att)
                value = getattr(obj, att)
                if (hex_name and cls._is_hex_number(value)) and not unmute:
                    continue
                value = cls._munge_metadata(
                    obj=value, truncate_length=truncate_length, unmute=unmute
                )
                if value:
                    out_att = cls._truncate(att, truncate_length) if hex_name else att
                    if out_att == "msg":
                        return (
                            " ".join(value) if isinstance(value, list) else str(value)
                        )
                    result[out_att] = value
            return result
        elif cls._is_hex_number(obj) and not unmute:
            return {}
        else:
            return obj

    def _parse_nft_mint(
        self, meta: blockfrost.utils.Namespace, truncate_length: int
    ) -> str:
        meta_dict = meta.to_dict()
        result = "NFT Mint:"
        for policy, _v in meta_dict.items():
            if policy == "version":
                continue
            result += f"{self._format_policy(policy,truncate_length)}:"
            for asset_name in _v.to_dict().keys():
                result += f"{asset_name} "
        return result

    def _format_message(
        self, *, tx_obj: blockfrost.utils.Namespace, truncate_length: int, unmute: bool
    ) -> str:
        result: List[str] = []
        for metadata_key in tx_obj.metadata:
            if metadata_key.label == self.METADATA_NFT_MINT_LABEL:
                label = None
                val = self._parse_nft_mint(metadata_key.json_metadata, truncate_length)
            else:
                if metadata_key.label == self.METADATA_MESSAGE_LABEL:
                    label = None
                else:
                    label = self.labels.get(
                        metadata_key.label,
                        self._truncate(metadata_key.label, truncate_length),
                    )
                val = self._munge_metadata(
                    obj=metadata_key.json_metadata,
                    truncate_length=truncate_length,
                    unmute=unmute,
                )
            if (
                self._is_hex_number(label)
                and (not val or self._is_hex_number(val))
                and not unmute
            ):
                continue
            if label:
                result.extend([label, ":"])
            result.append(str(val))
        redeemer_scripts: Dict[str, Set] = defaultdict(set)
        for redeemer in tx_obj.redeemers:
            if redeemer.purpose == "spend":
                redeemer_scripts[redeemer.purpose].add(
                    self._format_script(redeemer.script_hash, truncate_length)
                )
            elif redeemer.purpose == "mint":
                redeemer_scripts[redeemer.purpose].add(
                    self._format_policy(redeemer.script_hash, truncate_length)
                )
            else:
                redeemer_scripts[redeemer.purpose].add(
                    self._truncate(redeemer.redeemer_data_hash, truncate_length)
                )
        for k, redeemer_script in redeemer_scripts.items():
            result.extend([k, str(redeemer_script)])
        if all(
            [
                utxo.address in self.address_stake.keys()
                for utxo in tx_obj.utxos.nonref_inputs + tx_obj.utxos.outputs
            ]
        ):
            result.extend(["(internal)"])
        return " ".join(result)

    def _format_script(self, script: str, truncate_length: int) -> str:
        return self.scripts.get(script, self._truncate(script, truncate_length))

    def _format_policy(self, policy: str, truncate_length: int) -> Optional[str]:
        return self.policy_names.get(policy, self._truncate(policy, truncate_length))

    @classmethod
    def extract_timestamp(
        cls, transaction: blockfrost.utils.Namespace
    ) -> np.datetime64:
        """Returns timestamp of transaction."""
        return np.datetime64(
            datetime.datetime.fromtimestamp(transaction.block_time)
        ) + (int(transaction.index) * cls.TRANSACTION_OFFSET)

    def _drop_muted_assets(self, balance: pd.DataFrame) -> None:
        all_assets = frozenset([asset.asset for asset in self.data.assets])
        own_assets = frozenset(
            # Assets that touch own addresses
            x[0]
            for x in balance.xs(self.OWN_LABEL, level=1, axis=1).columns
        )
        muted_assets = frozenset(
            [
                asset.asset
                for asset in self.data.assets
                if any(self.muted_policies == asset.policy_id)
            ]
        )
        pinned_assets = frozenset(
            [
                asset.asset
                for asset in self.data.assets
                if any(self.pinned_policies == asset.policy_id)
            ]
        )
        assets_to_drop = (
            all_assets.difference(own_assets)
            .union(muted_assets)
            .difference(pinned_assets)
        )
        balance.drop(assets_to_drop, axis=1, level=0, inplace=True)

    def reward_transaction(
        self, reward: Tuple[str, blockfrost.utils.Namespace]
    ) -> blockfrost.utils.Namespace:
        """Build reward pseudo-transaction for tuple (staking_addr, reward)."""
        result = blockfrost.utils.Namespace()
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
        result.withdrawal_count = 0
        result.withdrawals = []
        result.utxos = blockfrost.utils.Namespace()
        result.utxos.inputs = []
        result.utxos.outputs = []
        result.utxos.nonref_inputs = []
        result.mirs = []
        return result

    def _own_addr_key(
        self,
        *,
        suffix: str,
        stake: str,
        addr: Optional[str] = None,
        detail_level: int,
        truncate_length: int,
    ) -> str:
        fields: List[str] = []
        if detail_level > 2:
            fields.extend([self._truncate(stake, truncate_length)])
        if detail_level > 3 and addr:
            fields.extend([self._truncate(addr, truncate_length)])
        fields.extend([suffix])
        return " " + "-".join(fields)

    def _column_key(
        self, *, utxo, amount, raw_values: bool, detail_level: int, truncate_length: int
    ) -> Tuple[str, str, str]:
        # Index: (asset_id, own, address_name)
        own = utxo.address in self.address_stake.keys()
        if raw_values:
            addr = self._truncate(utxo.address, truncate_length)
        else:
            if utxo.address in self.address_names.keys():
                addr = self.address_names[utxo.address]
            elif own:
                addr = self._own_addr_key(
                    suffix="wallet",
                    stake=self.address_stake[utxo.address],
                    addr=utxo.address,
                    detail_level=detail_level,
                    truncate_length=truncate_length,
                )
            else:
                addr = self.OTHER_LABEL
        return (
            amount.unit if amount.unit != self.data.LOVELACE_ASSET else self.ADA_ASSET,
            self.OWN_LABEL if own else self.OTHER_LABEL,
            addr,
        )

    def _transaction_balance(
        self,
        *,
        transaction: blockfrost.utils.Namespace,
        raw_values: bool,
        detail_level: int,
        truncate_length: int,
        unmute: bool,
        asset_with_policy: bool,
    ) -> Any:
        result: MutableMapping[Tuple, np.longlong] = defaultdict(lambda: np.longlong(0))
        result[(self.ADA_ASSET, self.OTHER_LABEL, "  fees")] += np.longlong(
            transaction.fees
        )
        result[(self.ADA_ASSET, self.OWN_LABEL, "  deposit")] += np.longlong(
            transaction.deposit
        )
        if transaction.reward_amount:
            result[
                (
                    self.ADA_ASSET,
                    self.OTHER_LABEL,
                    self._own_addr_key(
                        suffix="rewards",
                        stake=transaction.reward_address,
                        detail_level=detail_level,
                        truncate_length=truncate_length,
                    ),
                )
            ] -= np.longlong(transaction.reward_amount)
            result[
                (
                    self.ADA_ASSET,
                    self.OWN_LABEL,
                    self._own_addr_key(
                        suffix="withdrawals",
                        stake=transaction.reward_address,
                        detail_level=detail_level,
                        truncate_length=truncate_length,
                    ),
                )
            ] += np.longlong(transaction.reward_amount)
        for _w in transaction.withdrawals:
            assert isinstance(_w.address, str)
            result[
                (
                    self.ADA_ASSET,
                    self.OWN_LABEL,
                    self._own_addr_key(
                        suffix="withdrawals",
                        stake=_w.address,
                        detail_level=detail_level,
                        truncate_length=truncate_length,
                    ),
                )
            ] -= np.longlong(_w.amount)
        for _m in transaction.mirs:
            if _m.address in self.data.addresses.keys():
                result[
                    (
                        self.ADA_ASSET,
                        self.OTHER_LABEL,
                        self._own_addr_key(
                            suffix="mirs-" + _m.pot,
                            stake=_m.address,
                            detail_level=detail_level,
                            truncate_length=truncate_length,
                        ),
                    )
                ] -= np.longlong(_m.amount)
                result[
                    (
                        self.ADA_ASSET,
                        self.OWN_LABEL,
                        self._own_addr_key(
                            suffix="withdrawals",
                            stake=_m.address,
                            detail_level=detail_level,
                            truncate_length=truncate_length,
                        ),
                    )
                ] += np.longlong(_m.amount)
        for utxo in transaction.utxos.nonref_inputs:
            if not utxo.collateral or not transaction.valid_contract:
                for amount in utxo.amount:
                    result[
                        self._column_key(
                            utxo=utxo,
                            amount=amount,
                            raw_values=raw_values,
                            detail_level=detail_level,
                            truncate_length=truncate_length,
                        )
                    ] -= np.longlong(amount.quantity)

        for utxo in transaction.utxos.outputs:
            for amount in utxo.amount:
                result[
                    self._column_key(
                        utxo=utxo,
                        amount=amount,
                        raw_values=raw_values,
                        detail_level=detail_level,
                        truncate_length=truncate_length,
                    )
                ] += np.longlong(amount.quantity)

        sum_by_asset: MutableMapping[str, np.longlong] = defaultdict(
            lambda: np.longlong(0)
        )
        for key, value in result.items():
            sum_by_asset[key[0]] += value
        sum_by_asset = {k: v for k, v in sum_by_asset.items() if v}
        assert (
            len(sum_by_asset) - (1 if self.ADA_ASSET in sum_by_asset else 0)
            == transaction.asset_mint_or_burn_count
        ), (
            f"Unbalanced transaction: {transaction.hash if transaction.hash else '-'} : "
            + self._format_message(
                tx_obj=transaction, truncate_length=truncate_length, unmute=unmute
            )
            + " : "
            + str(
                {
                    self._decode_asset_name(k, truncate_length, asset_with_policy): v
                    for k, v in sum_by_asset.items()
                    if v != np.longlong(0)
                }
            )
        )
        return result

    def make_balance_frame(
        self,
        *,
        with_total: bool,
        raw_values: bool,
        detail_level: int,
        truncate_length: int,
        unmute: bool,
        add_asset_id: bool,
        asset_with_policy: bool,
    ):
        """Make DataFrame with transaction balances."""
        balance = pd.DataFrame(
            data=[
                self._transaction_balance(
                    transaction=x,
                    raw_values=raw_values,
                    detail_level=detail_level,
                    truncate_length=truncate_length,
                    unmute=unmute,
                    asset_with_policy=asset_with_policy,
                )
                for x in self.transactions
            ],
            index=self.transactions.index,
            dtype="Int64",
        )
        balance.columns = pd.MultiIndex.from_tuples(balance.columns)
        balance.sort_index(axis=1, level=0, sort_remaining=True, inplace=True)
        if not unmute:
            self._drop_muted_assets(balance)
        group = (0, 1) if detail_level == 1 else (0, 1, 2)
        balance = balance.T.groupby(level=group).sum(numeric_only=True).T
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

        balance.columns = pd.MultiIndex.from_tuples(
            [
                (
                    (
                        self._truncate(c[0], truncate_length)
                        if raw_values
                        else self._decode_asset_name(
                            c[0], truncate_length, asset_with_policy
                        )
                    )
                    + (
                        f"\n{c[0]}" if (add_asset_id and c[0] != self.ADA_ASSET) else ""
                    ),
                )
                + cast(tuple, c)[1:]
                for c in balance.columns
            ]
        )
        balance.sort_index(axis=1, level=0, sort_remaining=True, inplace=True)
        if detail_level == 1:
            return balance.xs(self.OWN_LABEL, level=1, axis=1)
        else:
            return balance

    def make_transaction_frame(
        self,
        *,
        detail_level: int,
        raw_values: bool,
        with_total: bool,
        truncate_length: int,
        unmute: bool,
        add_asset_id: bool,
        asset_with_policy: bool,
    ) -> pd.DataFrame:
        """Build a transaction spreadsheet."""

        msg_frame = pd.DataFrame(
            data=[
                {
                    "hash": x.hash,
                    "message": self._format_message(
                        tx_obj=x, truncate_length=truncate_length, unmute=unmute
                    ),
                }
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
            detail_level=detail_level,
            with_total=with_total,
            raw_values=raw_values,
            truncate_length=truncate_length,
            unmute=unmute,
            add_asset_id=add_asset_id,
            asset_with_policy=asset_with_policy,
        ).replace(0, pd.NA)
        if isinstance(balance_frame.columns[0], tuple):
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

    def _plot_title(self):
        return f"Asset balances in wallet until block {self.data.to_block}"

    def _draw_asset_legend(
        self, *, ax: Axes, asset_id: str, truncate_length: int, asset_with_policy: bool
    ):
        ticker = None
        image_data: Any = None
        url = None
        if asset_id == self.ADA_ASSET:
            ticker = "ADA"
            image_data = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "ada_logo.webp"
            )
            url = "https://cardano.org/"
        else:
            asset_obj = self.data.assets[asset_id]
            if hasattr(asset_obj, "metadata"):
                if hasattr(asset_obj.metadata, "logo"):
                    image_data = BytesIO(b64decode(asset_obj.metadata.logo))
                if hasattr(asset_obj.metadata, "ticker"):
                    ticker = asset_obj.metadata.ticker
                if hasattr(asset_obj.metadata, "url"):
                    url = asset_obj.metadata.url
        if ticker:
            ax.text(
                0.5,
                0.9,
                ticker,
                horizontalalignment="center",
                transform=ax.transAxes,
                fontsize="large",
                fontweight="bold",
                clip_on=True,
                url=url,
            )
        ax.text(
            0.5,
            0.8,
            self._decode_asset_name(asset_id, truncate_length, asset_with_policy),
            horizontalalignment="center",
            transform=ax.transAxes,
            fontsize="xx-small",
            clip_on=True,
            url=url,
        )
        if image_data:
            ax.set_adjustable("datalim")
            ax.imshow(
                mpl.image.imread(image_data),
                aspect="equal",
                extent=(0.3, 0.7, 0.8, 0.4),
                url=url,
            )
            ax.set_xlim((0.0, 1.0))
            ax.set_ylim((1.0, 0.0))

    def plot_balance(
        self,
        *,
        order: str,
        graph_width: float,
        graph_height: float,
        width_ratio: int,
        truncate_length: int,
        unmute: bool,
        asset_with_policy: bool,
    ):
        """Create a Matplotlib plot with the asset balance over time."""
        balance = self.make_balance_frame(
            detail_level=1,
            with_total=False,
            raw_values=True,
            truncate_length=0,
            unmute=unmute,
            add_asset_id=False,
            asset_with_policy=asset_with_policy,
        ).cumsum()
        if order == "alpha":
            balance.sort_index(
                axis=1,
                level=0,
                sort_remaining=True,
                inplace=True,
                key=lambda i: [
                    self._decode_asset_name(x, truncate_length, asset_with_policy)
                    for x in i
                ],
            )
        elif order == "appearance":
            balance.sort_index(
                axis=1,
                level=0,
                sort_remaining=True,
                inplace=True,
                key=lambda i: [
                    balance[x].replace(0, pd.NA).first_valid_index() for x in i
                ],
            )
        else:
            raise ValueError(f"Unkown ordering: {order}")
        fig, ax = pyplot.subplots(
            len(balance.columns),
            2,
            width_ratios=(width_ratio, 1),
            figsize=(graph_width, graph_height * len(balance.columns)),
        )
        fig.suptitle("\n" + self._plot_title() + "\n")
        for i in range(  # pylint: disable=consider-using-enumerate
            len((balance.columns))
        ):
            ax[i][1].xaxis.set_visible(False)
            ax[i][1].yaxis.set_visible(False)
            ax[i][0].spines.right.set_visible(False)
            ax[i][1].spines.left.set_visible(False)
            balance.plot(
                y=balance.columns[i],
                xlim=(min(balance.index), max(balance.index)),
                ax=ax[i][0],
                legend=False,
            )
            if i == 0 and len(balance.columns) > 1:
                ax[i][0].xaxis.set_ticks_position("top")
            elif i < len(balance.columns) - 1:
                ax[i][0].xaxis.set_ticklabels([])
                ax[i][0].xaxis.set_ticks_position("none")
                ax[i][0].spines.bottom.set_visible(False)
                ax[i][1].spines.bottom.set_visible(False)
            self._draw_asset_legend(
                ax=ax[i][1],
                asset_id=balance.columns[i],
                truncate_length=truncate_length,
                asset_with_policy=asset_with_policy,
            )

    def get_graph_metadata(self, filename: str) -> Mapping:
        """Return graph metadata for file name."""
        save_format = os.path.splitext(filename)[1].removeprefix(".")
        if not save_format:
            save_format = mpl.rcParams["savefig.format"]
        if save_format in ("svg", "pdf"):
            return {"Creator": self.CREATOR_STRING, "Title": self._plot_title()}
        elif save_format == "png":
            return {"Software": self.CREATOR_STRING, "Title": self._plot_title()}
        elif save_format in ("ps", "eps"):
            return {"Creator": self.CREATOR_STRING}
        else:
            return {}
