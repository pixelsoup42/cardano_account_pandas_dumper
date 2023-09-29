""" Cardano Account To Pandas Dumper."""
import datetime
import functools
import itertools
from collections import defaultdict
from typing import Any, Dict, FrozenSet, List, MutableMapping, Optional, Set, Tuple

import blockfrost.utils
import numpy as np
import pandas as pd
from blockfrost import BlockFrostApi


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
        self.transactions = self._transaction_data(api)
        self.assets = pd.Series(
            name="Assets",
            data={
                a: api.asset(a)
                for a in frozenset(
                    [
                        a.unit
                        for tx_obj in self.transactions  # pylint: disable=not-an-iterable
                        for i in (tx_obj.utxos.inputs + tx_obj.utxos.outputs)
                        for a in i.amount
                    ]
                ).difference([self.LOVELACE_ASSET])
            },
        ).sort_index()

    def _transaction_data(
        self,
        api: BlockFrostApi,
    ) -> pd.Series:
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
        return pd.Series(
            name="Transactions", data={t.hash: t for t in result_list}
        ).sort_index()


class AccountPandasDumper:
    """Hold logic to convert an instance of AccountData to a Pandas dataframe."""

    TRANSACTION_OFFSET = np.timedelta64(1000, "ns")
    OWN_LABEL = "own"
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
        raw_values: bool,
        unmute: bool,
    ):
        self.data = data
        self.truncate_length = truncate_length
        self.raw_values = raw_values
        self.unmute = unmute
        self.address_names = pd.Series(
            {a: " wallet" for a in self.data.own_addresses}
            | known_dict.get("addresses", {})
        )
        self.policy_names = pd.Series(known_dict.get("policies", {}))
        self.asset_names = pd.Series(
            {asset.asset: self._decode_asset_name(asset) for asset in self.data.assets}
            | {self.ADA_ASSET: self.ADA_ASSET}
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
        for policy, v in meta_dict.items():
            if policy == "version":
                continue
            for asset_name in v.to_dict().keys():
                result += f"{self._format_policy(policy)}@{asset_name} "
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
                redeemer_scripts["Spend:"].add(
                    self._format_script(redeemer.script_hash)
                )
            elif redeemer.purpose == "mint":
                redeemer_scripts["Mint:"].add(self._format_policy(redeemer.script_hash))
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
        return ({} if self.raw_values else self.scripts).get(
            script, self._truncate(script)
        )

    def _format_policy(self, policy: str) -> Optional[str]:
        return ({} if self.raw_values else self.policy_names).get(
            policy, self._truncate(policy)
        )

    @classmethod
    def _extract_timestamp(cls, transaction: blockfrost.utils.Namespace) -> Any:
        return np.datetime64(
            datetime.datetime.fromtimestamp(transaction.block_time)
        ) + (int(transaction.index) * cls.TRANSACTION_OFFSET)

    def _drop_muted_assets(self, balance: pd.DataFrame) -> None:
        # Drop assets that only touch foreign addresses
        assets_to_drop = frozenset(
            # Assets that touch other addresses
            x[:1]
            for x in balance.xs(self.OTHER_LABEL, level=-1, axis=1).columns
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
            for x in balance.xs(self.OWN_LABEL, level=-1, axis=1).columns
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
        epoch = self.data.epochs[
            reward[1].epoch + 1
        ]  # Time is right before start of next epoch.
        result.block_time = epoch.start_time
        result.index = -1
        result.fees = "0"
        result.deposit = "0"
        result.redeemers = []
        result.hash = None
        result.withdrawals = []
        result.utxos = blockfrost.utils.Namespace()
        result.utxos.inputs = []
        result.utxos.outputs = []
        result.utxos.nonref_inputs = []
        return result

    def _column_key(self, utxo, amount):
        return (
            amount.unit if amount.unit != self.data.LOVELACE_ASSET else self.ADA_ASSET,
            self._truncate(utxo.address)
            if self.raw_values
            else self.address_names.get(
                utxo.address,
                self.OTHER_LABEL,
            ),
            self.OWN_LABEL
            if utxo.address in self.data.own_addresses
            else self.OTHER_LABEL,
        )

    def _transaction_balance(self, transaction: blockfrost.utils.Namespace) -> Any:
        # Index: (asset_id, address_name, own)
        result: MutableMapping[Tuple, np.longlong] = defaultdict(lambda: np.longlong(0))
        result[(self.ADA_ASSET, " fees", self.OWN_LABEL)] = np.longlong(
            transaction.fees
        )
        result[(self.ADA_ASSET, " deposit", self.OWN_LABEL)] = np.longlong(
            transaction.deposit
        )
        if transaction.reward_amount:
            result[(self.ADA_ASSET, " rewards", self.OWN_LABEL)] = np.longlong(
                transaction.reward_amount
            )
        if transaction.withdrawals:
            result[
                (self.ADA_ASSET, " rewards withdrawal", self.OWN_LABEL)
            ] = np.negative(
                functools.reduce(
                    np.add,
                    [np.longlong(w.amount) for w in transaction.withdrawals],
                    np.longlong(0),
                )
            )
        for utxo in transaction.utxos.nonref_inputs:
            if not utxo.collateral or not transaction.valid_contract:
                for amount in utxo.amount:
                    result[self._column_key(utxo, amount)] -= np.longlong(
                        amount.quantity
                    )

        for utxo in transaction.utxos.outputs:
            for amount in utxo.amount:
                result[self._column_key(utxo, amount)] += np.longlong(amount.quantity)

        return result

    def make_balance_frame(self, transactions: pd.Series, detail_level: int):
        """Make DataFrame with transaction balances."""
        balance = pd.DataFrame(
            data=[self._transaction_balance(x) for x in transactions],
            dtype="Int64",
        )
        balance.columns = pd.MultiIndex.from_tuples(balance.columns)
        balance.sort_index(axis=1, level=0, sort_remaining=True, inplace=True)
        if not self.unmute:
            self._drop_muted_assets(balance)
        if detail_level == 1:
            balance.drop(labels=self.OTHER_LABEL, axis=1, level=2, inplace=True)
        balance = (
            balance.T.groupby(
                level=(0, 1)
                if not (self.raw_values and detail_level > 1)
                else (0, 1, 2)
            )
            .sum(numeric_only=True)
            .T
        )

        balance = balance * [
            np.float_power(10, np.negative(self.asset_decimals[c[0]]))
            for c in balance.columns
        ]
        if not self.raw_values:
            balance.columns = pd.MultiIndex.from_tuples(
                [(self.asset_names[c[0]], c[1]) for c in balance.columns]
            )
        balance.sort_index(axis=1, level=0, sort_remaining=True, inplace=True)

        return balance

    def make_transaction_frame(
        self,
        transactions: pd.Series,
        detail_level: int,
        with_tx_hash: bool = True,
        with_tx_message: bool = True,
        with_total: bool = True,
    ) -> pd.DataFrame:
        """Build a transaction spreadsheet."""

        columns = [transactions.rename("timestamp").map(self._extract_timestamp)]
        total: List[Any] = [columns[0].max() + self.TRANSACTION_OFFSET]
        if with_tx_hash:
            columns.append(transactions.rename("hash").map(lambda x: x.hash))
            total.append("")
        if with_tx_message:
            columns.append(transactions.rename("message").map(self._format_message))
            total.append("Total")
        balance = self.make_balance_frame(transactions, detail_level)
        frame = pd.concat(columns, axis=1)
        frame.reset_index(drop=True, inplace=True)
        frame.columns = pd.MultiIndex.from_tuples(
            [
                ("metadata", c) + (len(balance.columns[0]) - 2) * ("",)
                for c in frame.columns
            ]
        )
        frame = frame.join(balance)
        frame.sort_values(by=frame.columns[0], inplace=True)
        # Add total line at the bottom
        if with_total:
            for column in balance.columns:
                total.append(balance[column].sum())
            frame = pd.concat(
                [frame, pd.DataFrame(data=[total], columns=frame.columns)]
            )
        return frame
