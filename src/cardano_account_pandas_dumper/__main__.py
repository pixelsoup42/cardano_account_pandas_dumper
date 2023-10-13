""" Main for Cardano Account Pandas Dumper. """
import argparse
import os
import pickle
import warnings
from json import JSONDecodeError

import jstyleson
import matplotlib.pyplot as plt
from blockfrost import ApiError, BlockFrostApi
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from .cardano_account_pandas_dumper import AccountData, AccountPandasDumper

# Error codes due to project key rate limiting or capping
PROJECT_KEY_ERROR_CODES = frozenset([402, 403, 418, 429])


def _create_arg_parser():
    result = argparse.ArgumentParser(
        prog="cardano_account_pandas_dumper",
        description="Retrieve transaction history for Cardano staking addresses.",
    )
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
        "--xlsx_output",
        help="Path to .xlsx output file.",
        type=argparse.FileType("wb"),
    )
    result.add_argument(
        "--csv_output",
        help="Path to CSV output file.",
        type=argparse.FileType("wb"),
    )
    result.add_argument(
        "--graph_output",
        help="Path to graphics output file.",
        type=str,

    )
    result.add_argument(
        "--detail_level",
        help="Level of detail of report (1=only own addresses, 2=other addresses as well).",
        default=1,
        type=int,
    )
    result.add_argument(
        "--unmute",
        help="Do not auto-mute anything, do not use muted policies.",
        action="store_true",
    )
    result.add_argument(
        "--truncate_length",
        help="Length to truncate numerical identifiers to, 0= do not truncate.",
        type=int,
        default=6,
    )
    result.add_argument(
        "--raw_values",
        help="Keep assets, policies and addresses as hex instead of looking up names.",
        action="store_true",
    )
    result.add_argument(
        "--with_rewards",
        help="Add synthetic transactions for rewards.",
        default=True,
        type=bool,
    )
    result.add_argument(
        "--with_total",
        help="Add line with totals for each column at the bottom of the spreadsheet.",
        default=True,
        type=bool,
    )
    return result


def main():
    """Main function."""
    parser = _create_arg_parser()
    args = parser.parse_args()
    invalid_staking_addresses = frozenset(
        [a for a in args.staking_address if not a.startswith("stake")]
    )
    if invalid_staking_addresses:
        parser.exit(
            status=1,
            message="Following addresses do not look like valid staking addresses: "
            + " ".join(invalid_staking_addresses),
        )
    if not any([args.checkpoint_output, args.csv_output, args.xlsx_output, args.graph_output]):
        parser.exit(
            status=1,
            message="No output specified, neeed at least one of --checkpoint_output,"
            + " --csv_output, --xlsx_output, --graph_output.\n",
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
    else:
        try:
            api_instance = BlockFrostApi(project_id=args.blockfrost_project_id)
            data_from_api = AccountData(
                api=api_instance,
                staking_addresses=staking_addresses_set,
                to_block=args.to_block,
                include_rewards=args.with_rewards,
            )
        except ApiError as api_exception:
            parser.exit(
                status=2,
                message=(
                    f"Failed to read data from blockfrost.io: {api_exception}."
                    + (
                        "\nMaybe create your own API key at https://blockfrost.io/dashboard and "
                        + "specify it with the --blockfrost_project_id flag."
                    )
                    if int(api_exception.status_code) in PROJECT_KEY_ERROR_CODES
                    else ""
                ),
            )
        except (JSONDecodeError, OSError) as exception:
            parser.exit(
                status=2,
                message=(f"Failed to read data from blockfrost.io: {exception},"),
            )
        if args.checkpoint_output:
            try:
                pickle.dump(obj=data_from_api, file=args.checkpoint_output)
                args.checkpoint_output.flush()
            except (pickle.PicklingError, OSError) as exception:
                warnings.warn(f"Failed to write checkpoint: {exception}")
    reporter = AccountPandasDumper(
        data=data_from_api,
        known_dict=known_dict_from_file,
        truncate_length=args.truncate_length,
        unmute=args.unmute,
        detail_level=args.detail_level,
    )
    if args.csv_output:
        try:
            reporter.make_transaction_frame(
                with_total=args.with_total,
                raw_values=args.raw_values,
            ).to_csv(
                args.csv_output,
            )
        except OSError as exception:
            warnings.warn(f"Failed to write CSV file: {exception}")
    if args.xlsx_output:
        try:
            frame = reporter.make_transaction_frame(
                with_total=args.with_total,
                text_cleaner=lambda x: ILLEGAL_CHARACTERS_RE.sub(
                    lambda y: "".join(
                        ["\\x0" + hex(ord(y.group(0))).removeprefix("0x")]
                    ),
                    x,
                ),
                raw_values=args.raw_values,
            )
            frame.to_excel(
                args.xlsx_output,
                sheet_name=f"Transactions to block {args.to_block}",
                freeze_panes=(
                    len(frame.columns[0]) + 1
                    if isinstance(type(frame.columns[0]), tuple)
                    else 2,
                    3,
                ),
            )
        except OSError as exception:
            warnings.warn(f"Failed to write .xlsx file: {exception}")
    if args.graph_output:
        reporter.plot_balance()
        try:
            plt.savefig(args.graph_output,metadata=reporter.get_graph_metadata(args.graph_output),pad_inches=0.5)
        except OSError as exception:
            warnings.warn(f"Failed to write graph file: {exception}")
    print("Done.")


if __name__ == "__main__":
    main()
