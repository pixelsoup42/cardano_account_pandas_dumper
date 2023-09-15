""" Cardano Account To Pandas Dumper."""
import argparse
import datetime
import functools
import itertools
from collections import defaultdict
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
)
import pandas as pd
import numpy as np
from blockfrost import BlockFrostApi
import blockfrost.utils


class AccountData:
    """Hold data retrieved from the API to allow checkpointing it."""

    LOVELACE_ASSET = "lovelace"
    LOVELACE_DECIMALS = 6

    def __init__(
        self,
        api: BlockFrostApi,
        staking_addresses: FrozenSet[str],
        to_block: Optional[int],
        rewards: bool,
    ) -> None:
        self.staking_addresses = staking_addresses
        if to_block is None:
            to_block = int(api.block_latest().height - 1)
        self.to_block = to_block
        block_last = api.block(self.to_block)
        block_after_last = api.block(self.to_block + 1)
        self.end_time = block_after_last.time
        self.end_epoch = block_last.epoch + 1
        self.own_addresses: FrozenSet[str] = self._own_addresses(api)
        self.rewards = rewards
        if self.rewards:
            self.reward_transactions = self._reward_transactions(api)
        self.transactions = self._transaction_data(api)
        self.assets = self._assets_from_transactions(api)

    def _own_addresses(self, api: BlockFrostApi) -> FrozenSet[str]:
        return frozenset(
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
            name="Transactions", data=result_list, index=[t.hash for t in result_list]
        ).sort_index()

    @classmethod
    def _fix_api_asset(
        cls, asset_id: str, asset: blockfrost.utils.Namespace
    ) -> blockfrost.utils.Namespace:
        asset.asset_id = asset_id
        if not hasattr(asset, "metadata") or asset.metadata is None:
            asset.metadata = blockfrost.utils.Namespace()
        if not (hasattr(asset.metadata, "name") and asset.metadata.name):
            asset.raw_name = asset_id.removeprefix(asset.policy_id)
        else:
            asset.raw_name = str(bytes(asset.metadata.name, "utf-8").hex())
        if not hasattr(asset.metadata, "decimals"):
            asset.metadata.decimals = 0
        return asset

    def _assets_from_transactions(self, api: BlockFrostApi) -> pd.Series:
        all_asset_ids: Set[str] = set()
        for tx_obj in self.transactions:  # pylint: disable=not-an-iterable
            if hasattr(tx_obj, "utxos"):
                all_asset_ids.update(
                    [a.unit for i in tx_obj.utxos.inputs for a in i.amount]
                    + [a.unit for i in tx_obj.utxos.outputs for a in i.amount]
                )
        lovelace_asset_obj = blockfrost.utils.Namespace()
        lovelace_asset_obj.metadata = blockfrost.utils.Namespace()
        lovelace_asset_obj.metadata.name = "ADA"
        lovelace_asset_obj.metadata.decimals = self.LOVELACE_DECIMALS
        lovelace_asset_obj.policy_id = ""
        lovelace_asset_obj.asset_name = "ADA"
        asset_list = [
            self._fix_api_asset(
                asset,
                api.asset(asset)
                if asset != self.LOVELACE_ASSET
                else lovelace_asset_obj,
            )
            for asset in all_asset_ids
        ]
        return pd.Series(
            data=asset_list,
            index=pd.MultiIndex.from_tuples(
                [
                    (asset.asset_id, asset.policy_id, asset.raw_name)
                    for asset in asset_list
                ]
            ),
        ).sort_index()

    @staticmethod
    def _reward_transaction(
        api: BlockFrostApi,
        reward: blockfrost.utils.Namespace,
        pools: Mapping[str, blockfrost.utils.Namespace],
    ) -> blockfrost.utils.Namespace:
        result = blockfrost.utils.Namespace()
        result.tx_hash = None
        pool_name = (
            pools[reward.pool_id].name if reward.pool_id in pools else reward.pool_id
        )
        result.metadata = [
            blockfrost.utils.Namespace(
                label="674",
                json_metadata=f"Reward: {reward.type} - {pool_name} - {reward.epoch}",
            )
        ]
        result.reward_amount = reward.amount
        epoch = api.epoch(reward.epoch + 1)  # Time is right before start of next epoch.
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

    def _reward_transactions(self, api: BlockFrostApi) -> pd.Series:
        reward_list = [
            a_r
            for s_a in self.staking_addresses
            for a_r in api.account_rewards(s_a, gather_pages=True)
            if a_r.epoch < self.end_epoch
        ]

        pool_result_list = {
            pool: api.pool_metadata(pool)
            for pool in frozenset([r.pool_id for r in reward_list])
        }
        reward_result_list = [
            self._reward_transaction(api=api, reward=a_r, pools=pool_result_list)
            for a_r in reward_list
        ]
        return pd.Series(
            name="Rewards",
            data=reward_result_list,
            index=[t.hash for t in reward_result_list],
        ).sort_index()


class AccountPandasDumper:
    """Hold logic to convert an instance of AccountData to a Pandas dataframe."""

    TRANSACTION_OFFSET = np.timedelta64(1000, "ns")

    def __init__(self, data: AccountData, known_dict: Any, args: argparse.Namespace):
        self.data = data
        self.known_dict = known_dict
        self.args = args
        self.address_names = (
            pd.concat(
                [
                    pd.Series(
                        known_dict.get("addresses", {}),
                    ),
                    pd.Series({a: " wallet" for a in self.data.own_addresses}),
                ]
            )
            if not args.raw_values
            else pd.Series()
        )
        self.policy_names = (
            pd.Series(known_dict.get("policies", {}))
            if not args.raw_values
            else pd.Series()
        )
        self.asset_names = pd.Series(
            {
                asset.asset_id: self._decode_asset_name(asset.raw_name)
                if not args.raw_values
                else self._truncate(asset.raw_name)
                for asset in self.data.assets
            }
        )
        self.muted_policies = pd.Series(known_dict.get("muted_policies", []))
        self.scripts = pd.Series(known_dict.get("scripts", {}))
        self.labels = pd.Series(known_dict.get("labels", {}))

    def _truncate(self, value: str) -> str:
        return (
            value
            if self.args.no_truncate or len(value) <= self.args.truncate_length
            else ("..." + value[-self.args.truncate_length :])
        )

    def _decode_asset_name(self, asset_raw_name: str) -> str:
        try:
            return bytes.fromhex(asset_raw_name).decode()
        except UnicodeDecodeError:
            pass
        return self._truncate(asset_raw_name)

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
                hex_value = isinstance(value, str) and self._is_hex_number(value)
                if (hex_name and hex_value) and not self.args.unmute:
                    continue
                out_att = self._truncate(att) if hex_name else att
                value = self._munge_metadata(value)
                if value:
                    result[out_att] = value
            return result
        elif isinstance(obj, str) and self._is_hex_number(obj) and not self.args.unmute:
            return {}
        else:
            return obj

    def _format_message(self, tx_obj: blockfrost.utils.Namespace) -> str:
        result: List[str] = []
        for metadata_key in tx_obj.metadata:
            label = self.labels.get(
                metadata_key.label, self._truncate(metadata_key.label)
            )
            val = self._munge_metadata(metadata_key.json_metadata)
            if (
                self._is_hex_number(label)
                and (not val or self._is_hex_number(val))
                and not self.args.unmute
            ):
                continue
            result.append(label)
            result.append(":")
            result.append(str(val))
        redeemer_scripts: Dict[str, List] = defaultdict(list)
        for redeemer in tx_obj.redeemers:
            if redeemer.purpose == "spend":
                redeemer_scripts["Spend:"].append(
                    self._format_script(redeemer.script_hash)
                )
            elif redeemer.purpose == "mint":
                redeemer_scripts["Mint:"].append(
                    self._format_policy(redeemer.script_hash)
                )
        for k, redeemer_script in redeemer_scripts.items():
            result.append(k)
            result.append(str(redeemer_script))
        if not result and all(
            [
                utxo.address in self.data.own_addresses
                for utxo in tx_obj.utxos.nonref_inputs + tx_obj.utxos.outputs
            ]
        ):
            result = ["(internal)"]
        return " ".join(result).removeprefix("Message : ")

    def _format_script(self, script: str) -> str:
        return self.scripts.get(script, self._truncate(script))

    def _format_policy(self, policy: str) -> Optional[str]:
        return self.policy_names.get(policy, self._truncate(policy))

    def _decimals_for_asset(self, asset: str) -> np.longlong:
        return np.longlong(self.data.assets[asset].metadata.decimals)

    def _asset_tuple(self, asset_id: str) -> Tuple:
        asset = self.data.assets[(asset_id,)].iloc[0]
        return (
            self._format_policy(asset.policy_id),
            self.asset_names.get(asset_id),
        )

    def _transaction_balance(self, transaction: blockfrost.utils.Namespace) -> Any:
        # Index: (policy,asset,address,address_name,own)
        result: MutableMapping[Tuple, np.longlong] = defaultdict(lambda: np.longlong(0))
        result[
            self._asset_tuple(self.data.LOVELACE_ASSET) + ("", " fees", True)
        ] = np.longlong(transaction.fees)
        result[
            self._asset_tuple(self.data.LOVELACE_ASSET) + ("", " deposit", True)
        ] = np.longlong(transaction.deposit)
        result[self._asset_tuple(self.data.LOVELACE_ASSET) + ("", " rewards", True)] = (
            np.negative(
                functools.reduce(
                    np.add,
                    [np.longlong(w.amount) for w in transaction.withdrawals],
                    np.longlong(0),
                )
            )
            if not transaction.reward_amount
            else np.longlong(transaction.reward_amount)
        )
        for utxo in transaction.utxos.nonref_inputs:
            if not utxo.collateral or not transaction.valid_contract:
                for amount in utxo.amount:
                    result[
                        self._asset_tuple(amount.unit)
                        + (
                            utxo.address,
                            self.address_names.get(
                                utxo.address,
                                self._truncate(utxo.address)
                                if self.args.raw_values
                                else "other",
                            ),
                            utxo.address in self.data.own_addresses,
                        )
                    ] -= np.longlong(amount.quantity)

        for utxo in transaction.utxos.outputs:
            for amount in utxo.amount:
                result[
                    self._asset_tuple(amount.unit)
                    + (
                        utxo.address,
                        self.address_names.get(
                            utxo.address,
                            self._truncate(utxo.address)
                            if self.args.raw_values
                            else "other",
                        ),
                        utxo.address in self.data.own_addresses,
                    )
                ] += np.longlong(amount.quantity)

        return result

    @classmethod
    def _extract_timestamp(cls, transaction: blockfrost.utils.Namespace) -> Any:
        return np.datetime64(
            datetime.datetime.fromtimestamp(transaction.block_time)
        ) + (int(transaction.index) * cls.TRANSACTION_OFFSET)

    def _drop_foreign_assets(self, balance: pd.DataFrame) -> None:
        # Drop assets that only touch foreign addresses
        balance.columns = pd.MultiIndex.from_tuples(balance.columns)
        assets_to_drop = frozenset(
            # Assets that touch other addresses
            x[:2]
            for x in balance.xs(False, level=-1, axis=1).columns
        ) - frozenset(
            # Assets that touch own addresses
            x[:2]
            for x in balance.xs(True, level=-1, axis=1).columns
        )
        balance.sort_index(inplace=True, axis=1)
        balance.drop(assets_to_drop, axis=1, inplace=True)

    def _drop_muted_policies(self, balance: pd.DataFrame) -> None:
        policies_to_drop = frozenset(
            [x[0] for x in balance.columns if x[0] in self.muted_policies]
        )
        balance.sort_index(inplace=True, axis=1)
        balance.drop(policies_to_drop, axis=1, inplace=True)

    def make_transaction_frame(self) -> pd.DataFrame:
        """Build a transaction spreadsheet."""

        # Add total line at the bottom
        # total = []
        # for column in outputs.columns:
        #     # Only NaN is float in the column
        #     total.append(
        #         functools.reduce(
        #             self.np.longlong_context.add,
        #             [a for a in outputs[column] if type(a) is type(np.longlong(0))],
        #             np.longlong(0),
        #         )
        #     )
        # outputs.loc["Total"] = total
        transactions = pd.concat(
            objs=[
                self.data.transactions,
                pd.Series() if self.args.no_rewards else self.data.reward_transactions,
            ],
        ).rename("transactions")
        timestamp = transactions.rename("timestamp").map(self._extract_timestamp)
        tx_hash = transactions.rename("hash").map(lambda x: x.hash)
        message = transactions.rename("message").map(self._format_message)
        balance = pd.DataFrame(
            data=[self._transaction_balance(x) for x in transactions],
            dtype=pd.Int64Dtype,
        )
        self._drop_foreign_assets(balance)
        if not self.args.unmute:
            self._drop_muted_policies(balance)
        if self.args.detail_level == 1:
            balance.drop(labels=False, axis=1, level=4, inplace=True)
        balance.columns = pd.MultiIndex.from_tuples(balance.columns)
        balance.sort_index(axis=1, level=0, sort_remaining=True, inplace=True)
        balance = balance.groupby(axis=1, level=(0, 1, 3)).sum(numeric_only=True)
        frame = pd.concat([timestamp, tx_hash, message], axis=1)
        frame.reset_index(drop=True, inplace=True)
        frame.columns = pd.MultiIndex.from_tuples(
            [
                ("metadata", c) + (len(balance.columns[0]) - 2) * ("",)
                for c in frame.columns
            ]
        )
        frame = frame.join(balance)
        frame.sort_values(by=frame.columns[0], inplace=True)
        return frame
