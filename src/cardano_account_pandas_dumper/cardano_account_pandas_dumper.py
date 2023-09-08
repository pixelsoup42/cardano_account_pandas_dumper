""" Cardano Account To Pandas Dumper."""
import datetime
import itertools
from collections import defaultdict
from decimal import Context, Decimal
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Set, Tuple
import pandas as pd
import numpy as np
from blockfrost import BlockFrostApi
from blockfrost.utils import Namespace


class AccountData:
    """Hold data retrieved from the API to allow checkpointing it."""

    LOVELACE_ASSET = "lovelace"
    LOVELACE_DECIMALS = 6
    TRANSACTION_OFFSET = np.timedelta64(
        1000, "ns"
    )  # Time for a stransaction is block_time + index * TRANSACTION_OFFSET

    def __init__(
        self,
        api: BlockFrostApi,
        staking_addresses: FrozenSet[str],
        to_block: Optional[int],
        rewards: bool,
    ) -> None:
        self.staking_addresses = staking_addresses
        if to_block is None:
            to_block = int(api.block_latest().height)
        self.to_block = to_block
        self.own_addresses: FrozenSet[str] = self._own_addresses(api)
        self.rewards = rewards
        if self.rewards:
            self.reward_transactions: pd.Series = self._reward_transactions(api)
        self.transactions: pd.Series = self._transaction_data(api)
        self.assets: pd.Series = self._assets_from_transactions(api)

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
        for addr in self.own_addresses:
            for outer_tx in api.address_transactions(
                addr,
                to_block=self.to_block,
                gather_pages=True,
            ):
                transaction = api.transaction(outer_tx.tx_hash)
                transaction.utxos = api.transaction_utxos(outer_tx.tx_hash)
                transaction.utxos.nonref_inputs = [
                    i for i in transaction.utxos.inputs if not i.reference
                ]
                transaction.metadata = api.transaction_metadata(outer_tx.tx_hash)
                transaction.redeemers = (
                    api.transaction_redeemers(outer_tx.tx_hash)
                    if transaction.redeemer_count
                    else []
                )
                transaction.withdrawals = (
                    api.transaction_withdrawals(outer_tx.tx_hash)
                    if transaction.withdrawal_count
                    else []
                )
                transaction.reward_amount = None

                result_list.append(transaction)
        index = pd.DatetimeIndex(
            [
                np.datetime64(datetime.datetime.fromtimestamp(t.block_time))
                + (int(t.index) * self.TRANSACTION_OFFSET)
                for t in result_list
            ],
        )
        return pd.Series(name="Transactions", data=result_list, index=index)

    def _assets_from_transactions(self, api: BlockFrostApi) -> pd.Series:
        all_asset_ids: Set[str] = set()
        for tx_obj in self.transactions:
            if hasattr(tx_obj, "utxos"):
                all_asset_ids.update(
                    [a.unit for i in tx_obj.utxos.inputs for a in i.amount]
                    + [a.unit for i in tx_obj.utxos.outputs for a in i.amount]
                )
        all_asset_ids.remove(AccountData.LOVELACE_ASSET)
        return pd.Series(
            name="Assets", data={asset: api.asset(asset) for asset in all_asset_ids}
        )

    @staticmethod
    def _reward_transaction(api: BlockFrostApi, reward: Namespace) -> Namespace:
        result = Namespace()
        result.tx_hash = None
        result.Metadata = Namespace()
        result.Metadata.message = f"Reward: {reward.type} - {reward.epoch}"
        result.reward_amount = reward.amount
        epoch = api.epoch(reward.epoch + 1)  # Time is right before start of next epoch.
        result.block_time = epoch.start_time
        result.index = -1
        result.fees = "0"
        result.deposit = "0"
        result.redeemers = []
        result.hash = None
        result.withdrawals = []
        result.utxos = Namespace()
        result.utxos.inputs = []
        result.utxos.outputs = []
        result.utxos.nonref_inputs = []
        return result

    def _reward_transactions(self, api: BlockFrostApi) -> pd.Series:
        result_list = [
            self._reward_transaction(api=api, reward=a_r)
            for s_a in self.staking_addresses
            for a_r in api.account_rewards(s_a, gather_pages=True)
        ]
        index = pd.DatetimeIndex(
            [
                np.datetime64(datetime.datetime.fromtimestamp(t.block_time))
                + (t.index * self.TRANSACTION_OFFSET)
                for t in result_list
            ],
        )
        return pd.Series(name="Rewards", data=result_list, index=index)


class AccountPandasDumper:
    """Hold logic to convert an instance of AccountData to a Pandas dataframe."""

    ADDRESSES_KEY = "addresses"
    POLICIES_KEY = "policies"
    MUTED_POLICIES_KEY = "muted_policies"
    SCRIPTS_KEY = "scripts"
    LABELS_KEY = "labels"
    ASSETS_KEY = "assets"

    def __init__(
        self,
        data: AccountData,
        known_dict: Mapping[str, Mapping[str, str]],
        detail_level: int,
        unmute: bool,
        truncate_length: Optional[int],
        raw_asset: bool,
        rewards: bool,
    ):
        self.known_dict = known_dict
        self.data = data
        self.truncate_length = truncate_length
        self.detail_level = detail_level
        self.unmute = unmute
        self.raw_asset = raw_asset
        self.rewards = rewards
        self.decimal_context = Context()

    def _format_asset(self, asset: str) -> Optional[str]:
        if asset == AccountData.LOVELACE_ASSET:
            return " ADA"
        if asset in self.known_dict[self.ASSETS_KEY]:
            return self.known_dict[self.ASSETS_KEY][asset]
        asset_obj = self.data.assets[asset]
        if asset_obj.metadata and asset_obj.metadata.name:
            return asset_obj.metadata.name
        if isinstance(asset_obj.asset_name, str):
            name_bytes = bytearray(
                [
                    b if b in range(32, 127) else 127
                    for b in bytes.fromhex(
                        asset_obj.asset_name.removeprefix(asset_obj.policy_id)
                    )
                ]
            )
            name_str = name_bytes.decode(encoding="ascii", errors="replace")
        else:
            name_str = "???"
        policy = self._format_policy(asset_obj.policy_id, self.unmute)
        return f"{policy}@{name_str}" if policy is not None else None

    def _truncate(self, value: str) -> str:
        return (
            (value[: self.truncate_length] + "...") if self.truncate_length else value
        )

    def _format_policy(self, policy: str, unmute: bool) -> Optional[str]:
        if self.known_dict[self.MUTED_POLICIES_KEY].get(policy, False) and not unmute:
            return None
        return self.known_dict[self.POLICIES_KEY].get(policy, self._truncate(policy))

    @staticmethod
    def _is_hex_number(num: Any) -> bool:
        return isinstance(num, str) and not bool(
            frozenset(num.lower().removeprefix("0x")).difference(
                frozenset("0123456789abcdef")
            )
        )

    def _munge_metadata(self, namespace_obj) -> Any:
        if isinstance(namespace_obj, Namespace):
            result = {}
            for att in dir(namespace_obj):
                if att.startswith("_") or att in (
                    "to_json",
                    "to_dict",
                ):
                    continue
                hex_name = self._is_hex_number(att)
                value = getattr(namespace_obj, att)
                hex_value = isinstance(value, str) and self._is_hex_number(value)
                if (hex_name and hex_value) and not self.unmute:
                    continue
                out_att = self._truncate(att) if hex_name else att
                value = self._munge_metadata(value)
                if value:
                    result[out_att] = value
            return result
        elif (
            isinstance(namespace_obj, str)
            and self._is_hex_number(namespace_obj)
            and not self.unmute
        ):
            return {}
        else:
            return namespace_obj

    def _format_message(self, tx_obj: Namespace) -> str:
        result: List[str] = []
        for metadata_key in tx_obj.metadata:
            if metadata_key.label in self.known_dict[self.LABELS_KEY]:
                label = self.known_dict[self.LABELS_KEY][metadata_key.label]
            else:
                label = metadata_key.label
            val = self._munge_metadata(metadata_key.json_metadata)
            if (
                self._is_hex_number(label)
                and (not val or self._is_hex_number(val))
                and not self.unmute
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
                    self._format_policy(redeemer.script_hash, True)
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
        return " ".join(result)

    def _format_script(self, script: str) -> str:
        return self.known_dict[self.SCRIPTS_KEY].get(
            script,
            self._truncate(script),
        )

    def _format_address(self, address: str) -> str:
        if address in self.data.own_addresses:
            return " own"
        return self.known_dict[self.ADDRESSES_KEY].get(
            address,
            "other",
        )

    def _decimals_for_asset(self, asset: str) -> int:
        if asset == self.data.LOVELACE_ASSET:
            return self.data.LOVELACE_DECIMALS
        if asset in self.data.assets and self.data.assets[asset].metadata:
            return self.data.assets[asset].metadata.decimals
        return 0

    def _utxo_amount_key(self, utxo: Namespace, amount: Namespace) -> Optional[Tuple]:
        if self.detail_level < 2 and utxo.address not in self.data.own_addresses:
            return None
        asset = self._format_asset(amount.unit)
        if asset is None:
            return None
        addr = self._format_address(utxo.address)

        return (
            (asset, addr, self._decimals_for_asset(amount.unit))
            if not self.raw_asset
            else (asset, amount.unit, addr, self._decimals_for_asset(amount.unit))
        )

    def transaction_dict(self) -> Iterable[Dict]:
        """Return a dict holding a Pandas row for transaction tx"""
        result_list = []
        for transaction in self.data.transactions:
            result: Dict = dict(
                [
                    (
                        ("  metadata", k, d)
                        if not self.raw_asset
                        else ("  metadata", k, "", d),
                        v,
                    )
                    for k, v, d in [
                        ("  hash", transaction.hash, 0),
                        ("  message", self._format_message(transaction), 0),
                        (" fees", int(transaction.fees), self.data.LOVELACE_DECIMALS),
                        (
                            "deposit",
                            int(transaction.deposit),
                            self.data.LOVELACE_DECIMALS,
                        ),
                        (
                            "rewards",
                            -sum(
                                [int(w.amount) for w in transaction.withdrawals],
                            )
                            if not transaction.reward_amount
                            else transaction.reward_amount,
                            self.data.LOVELACE_DECIMALS,
                        ),
                    ]
                ]
            )
            balance_result: Dict = defaultdict(lambda: Decimal(0))
                for i in transaction.utxos.nonref_inputs:
                    if not i.collateral or not transaction.valid_contract:
                        for amount in i.amount:
                            key = self._utxo_amount_key(i, amount)
                            if key is not None:
                                balance_result[key] -= Decimal(amount.quantity)

                for out in transaction.utxos.outputs:
                    for amount in out.amount:
                        key = self._utxo_amount_key(out, amount)
                        if key is not None:
                            balance_result[key] += Decimal(amount.quantity)
                result.update({k: v for k, v in balance_result.items() if v != 0})
            result_list.append(result)
        return result_list

    def make_transaction_array(self) -> pd.DataFrame:
        """Return a dataframe with each transaction until the specified block, included."""
        frame = pd.DataFrame(data=self.transaction_dict())

        # Scale according to decimal index row, and drop that row
        for col in frame.columns:
            if col[-1]:
                scale = self.decimal_context.power(10, -Decimal(col[-1]))
                frame[col] *= scale  # type: ignore
        frame.columns = pd.MultiIndex.from_tuples([c[:-1] for c in frame.columns])
        frame.sort_index(axis=1, level=0, sort_remaining=True, inplace=True)
        return frame
