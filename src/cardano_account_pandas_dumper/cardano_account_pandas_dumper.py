""" Cardano Account To Pandas Dumper."""
import datetime
import itertools
from collections import OrderedDict, defaultdict
from decimal import Context, Decimal
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Set, Tuple
import pandas
from blockfrost import BlockFrostApi
from blockfrost.utils import Namespace


class AccountData:
    """Hold data retrieved from the API to allow checkpointing it."""

    LOVELACE_ASSET = "lovelace"
    LOVELACE_DECIMALS = 6

    def __init__(
        self,
        api: BlockFrostApi,
        staking_addresses: FrozenSet[str],
        to_block: Optional[int],
    ) -> None:
        self.staking_addresses = staking_addresses
        if to_block is None:
            to_block = int(api.block_latest().height)
        self.to_block = to_block
        self.own_addresses = self._load_own_addresses(api)
        self.transactions = self._load_transaction_data(
            api, self._load_transaction_hashes(api)
        )
        self._collect_from_transactions(api)

    def _load_own_addresses(self, api: BlockFrostApi) -> FrozenSet[str]:
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

    def _load_transaction_hashes(self, api: BlockFrostApi) -> List[str]:
        return list(
            OrderedDict.fromkeys(
                [
                    t.tx_hash
                    for t in sorted(
                        itertools.chain(
                            *[
                                api.address_transactions(
                                    a,
                                    to_block=self.to_block,
                                    gather_pages=True,
                                )
                                for a in self.own_addresses
                            ],
                        ),
                        key=lambda x: (x.block_height, x.tx_index),
                    )
                ],
            )
        )

    @staticmethod
    def _load_transaction_data(
        api: BlockFrostApi, tx_hashes: List[str]
    ) -> OrderedDict[str, Namespace]:
        result = OrderedDict()
        for tx_hash in tx_hashes:
            transaction = api.transaction(tx_hash)
            transaction.utxos = api.transaction_utxos(tx_hash)
            transaction.metadata = api.transaction_metadata(tx_hash)
            transaction.redeemers = (
                api.transaction_redeemers(tx_hash) if transaction.redeemer_count else []
            )
            transaction.withdrawals = (
                api.transaction_withdrawals(tx_hash)
                if transaction.withdrawal_count
                else []
            )
            result[tx_hash] = transaction
        return result

    def _collect_from_transactions(self, api: BlockFrostApi) -> None:
        self.assets: Dict[str, Namespace] = {}

        all_asset_ids: Set[str] = set()
        self.addresses: Dict[str, Namespace] = {}
        all_addresses: Set[str] = set(self.own_addresses)
        for tx_obj in self.transactions.values():
            all_asset_ids.update(
                [a.unit for i in tx_obj.utxos.inputs for a in i.amount]
                + [a.unit for i in tx_obj.utxos.outputs for a in i.amount]
            )
            all_addresses.update(
                [i.address for i in tx_obj.utxos.inputs]
                + [i.address for i in tx_obj.utxos.outputs]
            )
        all_asset_ids.remove(self.LOVELACE_ASSET)
        for asset in all_asset_ids:
            self.assets[asset] = api.asset(asset)


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
    ):
        self.known_dict = known_dict
        self.data = data
        self.truncate_length = truncate_length
        self.detail_level = detail_level
        self.unmute = unmute
        self.raw_asset = raw_asset
        self.decimal_context = Context()
        for tx_obj in self.data.transactions.values():
            tx_obj.utxos.nonref_inputs = [
                i for i in tx_obj.utxos.inputs if not i.reference
            ]

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
            result = ["(Local)"]
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

    def transaction_dict(self, transaction: Namespace) -> Optional[Dict]:
        """Return a dict holding a Pandas row for transaction tx"""
        result: Dict = OrderedDict(
            [
                (
                    ("  metadata", k, d)
                    if not self.raw_asset
                    else ("  metadata", k, "", d),
                    v,
                )
                for k, v, d in [
                    (
                        "  block_time",
                        # Make sure time monotonically increases by adding tx index to block time
                        datetime.datetime.fromtimestamp(
                            int(transaction.block_time) + int(transaction.index)
                        ),
                        0,
                    ),
                    ("  hash", transaction.hash, 0),
                    ("  message", self._format_message(transaction), 0),
                    (" fees", int(transaction.fees), self.data.LOVELACE_DECIMALS),
                    ("deposit", int(transaction.deposit), self.data.LOVELACE_DECIMALS),
                    (
                        "withdrawal_sum",
                        sum(
                            [int(w.amount) for w in transaction.withdrawals],
                        ),
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
        return result

    def make_transaction_array(self, to_block: int) -> pandas.DataFrame:
        """Return a dataframe with each transaction until the specified block, included."""
        data = []
        for transaction in self.data.transactions.values():
            if to_block is not None and int(transaction.block_height) > to_block:
                break
            data.append(self.transaction_dict(transaction))
        frame = pandas.DataFrame(
            data=data,
        )

        # Scale according to decimal index row, and drop that row
        for col in frame.columns:
            if col[-1]:
                scale = self.decimal_context.power(10, -Decimal(col[-1]))
                frame[col] *= scale  # type: ignore
        frame.columns = pandas.MultiIndex.from_tuples([c[:-1] for c in frame.columns])
        frame.sort_index(axis=1, level=0, sort_remaining=True, inplace=True)
        return frame
