""" Main for Cardano Account Pandas Dumper. """
import argparse
import os
import pickle
import warnings
from json import JSONDecodeError

import jstyleson
from blockfrost import ApiError, BlockFrostApi

from .cardano_account_pandas_dumper import AccountData, AccountPandasDumper


def _create_arg_parser():
    result = argparse.ArgumentParser(prog="cardano_account_pandas_dumper")
    exclusive_group = result.add_mutually_exclusive_group()
    result.add_argument(
        "--blockfrost_project_id",
        nargs="?",
        default="mainnetRlrNKtjWWp7VkzwRBrragNvtSsKyOeA4",
        help="Blockfrost API key, create your own at https://blockfrost.io/dashboard.",
    )
    exclusive_group.add_argument(
        "--checkpoint_output",
        nargs="?",
        help="Path to checkpoint file to create, if any.",
        type=argparse.FileType("wb"),
    )
    result.add_argument(
        "staking_address",
        nargs="+",
        help="The staking addresses to report on.",
    )
    result.add_argument(
        "--to_block",
        help="Block number to end the search at, if unspecified look at whole history.",
        type=int,
    )
    result.add_argument(
        "--known_file",
        nargs="?",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "known.jsonc"),
        help="Path to JSONC file with known addresses, scripts, policies, ...",
        type=argparse.FileType("r"),
    )
    exclusive_group.add_argument(
        "--from_checkpoint",
        nargs="?",
        help="Path to checkpoint file to read, if any.",
        type=argparse.FileType("rb"),
    )
    result.add_argument(
        "--pandas_output",
        help="Path to pickled Pandas dataframe output file.",
        type=argparse.FileType("wb"),
    )
    result.add_argument(
        "--csv_output",
        help="Path to CSV output file.",
        type=argparse.FileType("wb"),
    )
    result.add_argument(
        "--detail_level",
        help="Level of detail of report (1=only own addresses, 2=other addresses as well).",
        default=1,
        type=int,
    )
    result.add_argument(
        "--unmute",
        help="Do not mute policies in the mute list and numerical-only metadata.",
        action="store_true",
    )
    result.add_argument(
        "--truncate_length",
        help="Length to truncate numerical identifiers to.",
        type=int,
        default=6,
    )
    result.add_argument(
        "--no_truncate",
        help="Do not truncate numerical identifiers.",
        action="store_true",
    )
    result.add_argument(
        "--raw_asset",
        help="Add header row with concatenation of policy_id and hex-encoded asset_name.",
        action="store_true",
    )
    result.add_argument(
        "--no_rewards",
        help="Do not add reward transactions.",
        action="store_true",
    )
    return result


def main():
    """Main function."""
    parser = _create_arg_parser()
    args = parser.parse_args()
    if not any([args.checkpoint_output, args.csv_output, args.pandas_output]):
        parser.exit(
            status=1,
            message="No output specified, neeed at least one of --checkpoint_output,"
            + " --csv_output, --pandas_output.\n",
        )
    known_dict_from_file = jstyleson.load(args.known_file) if args.known_file else {}
    staking_addresses_set = frozenset(args.staking_address)
    if args.from_checkpoint:
        try:
            data_from_api = pickle.load(args.from_checkpoint)
        except (pickle.UnpicklingError, OSError) as exception:
            parser.exit(status=2, message=f"Failed to read checkpoint: {exception}")
        if (
            staking_addresses_set
            and staking_addresses_set != data_from_api.staking_addresses
        ):
            parser.exit(
                message=(
                    f"Specified staking adresses {staking_addresses_set} differ "
                    + f"from checkpoint's {data_from_api.staking_addresses}"
                ),
                status=1,
            )
        if args.to_block is not None and args.to_block != data_from_api.to_block:
            parser.exit(
                message=(
                    f"--to_block {args.to_block} different "
                    + f"from checkpoint's {data_from_api.to_block}."
                ),
                status=1,
            )
    elif staking_addresses_set:
        try:
            api_instance = BlockFrostApi(project_id=args.blockfrost_project_id)
            data_from_api = AccountData(
                api=api_instance,
                staking_addresses=staking_addresses_set,
                to_block=args.to_block,
                rewards=not args.no_rewards,
            )
        except (ApiError, JSONDecodeError, OSError) as exception:
            parser.exit(
                status=2,
                message=(
                    f"Failed to read data from blockfrost.io: {exception},"
                    + " maybe create your own API key at https://blockfrost.io/dashboard and "
                    + "specify it with the --blockfrost_project_id flag."
                ),
            )
        if args.checkpoint_output:
            try:
                pickle.dump(obj=data_from_api, file=args.checkpoint_output)
                args.checkpoint_output.flush()
            except (pickle.PicklingError, OSError) as exception:
                warnings.warn(f"Failed to write checkpoint: {exception}")
    else:
        parser.exit(status=1, message="Staking address(es) required.")
    reporter = AccountPandasDumper(
        data=data_from_api,
        known_dict=known_dict_from_file,
        detail_level=args.detail_level,
        unmute=args.unmute,
        truncate_length=None if args.no_truncate else args.truncate_length,
        raw_asset=args.raw_asset or False,
        rewards=data_from_api.rewards,
    )
    dataframe = reporter.make_transaction_frame()
    if args.pandas_output:
        try:
            dataframe.to_pickle(args.pandas_output)
        except (pickle.PicklingError, OSError) as exception:
            warnings.warn(f"Failed to write pandas file: {exception}")
    if args.csv_output:
        try:
            dataframe.to_csv(args.csv_output, index=False)
        except OSError as exception:
            warnings.warn(f"Failed to write CSV file: {exception}")
    print("Done.")


if __name__ == "__main__":
    main()
