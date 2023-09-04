# Cardano Account Pandas Dumper

![Project Logo](logo.png)

## Description

Create a spreadsheet with the owned amount of any Cardano asset at the end of a specific block, and a record of the transactions that affected it.

Also, provide a reusable module that lets you turn the transaction history of specified staking addresses into a Pandas dataframe for further analysis and processing.

## Requirements

* Python 3.11, possibly works with lower versions, not tested.

## Installation

```sh
pipx install git+https://github.com/pixelsoup42/cardano_account_pandas_dumper
```

## Basic Usage

The simplest use case is to just run the tool, specifying the CSV output file name and the staking address(es) you are interested in:

```sh
cardano_account_pandas_dumper  --csv_output report.csv <staking_address1> <staking_address2> ...
```

You can then load `report.csv` in your favorite spreadsheet software (eg. Libreoffice Calc or Excel)

If you get a [blockfrost.io](https://blockfrost.io) API error, or if execution is very slow, please see the `blockfrost_project_id` command line flag below.

This basic usage just lists all transactions that affect the specified staking addresses, with the total of each owned asset at the end of the specified `to_block`.

## Advanced usage

```sh
cardano_account_pandas_dumper --detail_level 2 --csv_output report.csv <staking_address1> <staking_address2> ...
```

With `--detail_level 2`, the tool outputs not only the balance and UTXOs of the owned addresses, but also includes external contracts and addresses.

### Command line flags

`-h`, `--help`
:  show help message and exit

`--blockfrost_project_id BLOCKFROST_PROJECT_ID`
: Blockfrost API key, create your own at <https://blockfrost.io/dashboard>.
This tool comes with its own prepackaged API key that uses the Free plan, so it will be rate limited and capped.
It is likely to be overused and/or abused, so you're better off creating your own.
If you use this tool seriously you're probably better off getting a paid plan.

`--checkpoint_output CHECKPOINT_OUTPUT`
: Path to checkpoint file to create, if any.
This is useful for development, the checkpoint contains all the data fetched from the API so you can tweak the report format without having to call the API on every run, which would be very slow and consume quota.

`--to_block TO_BLOCK`
: Block number to end the search at, if unspecified the tool will look up the latest known block from the API.
For instance, block 8211670 matches EOY 2022 pretty closely.

`--known_file KNOWN_FILE`
: Path to JSONC file with known addresses, scripts, policies, ... See the [packaged file](./src/cardano_account_pandas_dumper/known.jsonc) for an example.

`--from_checkpoint FROM_CHECKPOINT`
: Path to checkpoint file to read, if any.
The checkpoint must have been created with the `--checkpoint_output` flag.

`--pandas_output PANDAS_OUTPUT`
: Path to pickled Pandas dataframe output file.
If you want to further process the data with Pandas, you can serialize the generated `DataFrame` into a file.

`--csv_output CSV_OUTPUT`
: Path to CSV output file.
This the flag most people will need, it specifies the CSV file to write the output to.
Each row is a transaction, each column is a combination of asset + address.
Addresses belonging to one of the specified staking addresses are labeled as `own`.
With `--detail_level=2`, known addresses are listed with their name, other addresses are labeled as `other`.

`--detail_level DETAIL_LEVEL`
: Level of detail of report (1=only own addresses, 2=other addresses as well).
By default (`--detail_level=1`), only addresses linked to the specified staking addresses will be shown.
With `--detail_level=2`, all addresses will be shown.

`--unmute`
: Do not mute policies in the mute list and numerical-only metadata.
Some DeFI apps like MinSwap are very spammy, by default some NFTs are muted to keep the output lean.
The muted policies are listed in the `known.jsonc` file. This flag disables muting and shows all assets.

`--truncate_length TRUNCATE_LENGTH`
: Length to truncate numerical identifiers to.
When a policy, address or asset is not known, it is listed as a numerical hex value.
For legibility, those values are truncated to a specific number of digits (6 by default).
This flag lets you specify another truncation length.

`--no_truncate`
: Do not truncate numerical identifiers.
If you need numerical hex values to not be truncated at all (see `--truncate_length`above), specify this flag.

`--raw_asset`
: Add header row with concatenation of policy_id and hex-encoded asset_name.
This is useful if you need to look up a specific asset.

## Calculations and precision

All calculations are done using Python decimals to preserve accuracy.
However when importing into a spreadsheet, the values are usually converted to floats and some rounding errors can occur.
This is a spreadsheet issue, there isn't much that can be done by this tool to avoid it.
If you want to preserve accuracy the best way is probably to write a serialized Pandas dataframe and write some code to process it.

## Possible improvements

* The first obvious possible improvement would be to replace the static `--known_file` that lists the known addresses, policies and scripts with a dynamic API.

The [blockfrost.io](https://blockfrost.io) API already provides some metadata for assets, but AFAIK not for addresses, scripts and policies.

The current list of addresses, scripts and policies was gleaned from external sources like [cardanoscan.io](https://cardanoscan.io) and [cexplorer.io](https://cexplorer.io).

Any suggestion to improve this would be greatly appreciated (please open a GitHub issue).

* If you have other improvements or bug fixes in mind, please open a GitHub issue or send a PR.

However the general philosophy of this tool is to remain as simple as possible, the preferred way to build on top of it is to write other modules that import it, or to consume the data files it produces.

## If you have use for this tool, please consider supporting the toolsmith

Writing good tools takes time, effort and talent. If this tool is useful to you, please consider supporting the toolsmith by donating to

> addr1q84h5zhcvaur9ey8792w0jm5swrcyz8uta9ldnq7h43k2mvu5x99y2s9skjyv82evr0rmjry0een8almmxm5c50kq3lsfuxqc4

(mention "Cardano Account Pandas Dumper" in the message).

or purchasing one of our cool [PixelSoup NFTs](https://www.jpg.store/PixelSoup?tab=listings) !

Donations and NFT purchases are both really appreciated, the advantage of an NFT purchase is that there is a nonzero probability of financial upside.

If you think this tool can be useful to others, please retweet [the announcement](https://twitter.com/PixelSoup42/status/1697305462721396957)

## Comparison with [cardano-accointing-exporter](https://github.com/pabstma/cardano-accointing-exporter)

After finishing this tool, I was made aware that another comparable project existed: [cardano-accointing-exporter](https://github.com/pabstma/cardano-accointing-exporter)

Here is a comparison table for both projects (please submit corrections if you think anything is wrong):

| Feature | [cardano_account_pandas_dumper](https://github.com/pixelsoup42/cardano_account_pandas_dumper) | [cardano-accointing-exporter](https://github.com/pabstma/cardano-accointing-exporter) |
| ------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------|
| CSV output |✔️|✔️|
| coingecko integration for fiat price |❌|✔️[^1]|
| Knows about assets other than ADA |✔️|❌|
| Knows about DeFI contract addresses |✔️[^2]|❌|
| Extracts useful information from tx metadata |✔️|❌|
| Decimal arithmetic for absolute precision |✔️|❌|
| .xlsx output |❌[^3]|✔️|
| Serialized pandas dataframe output |✔️|❌|
| Ready to use after one-liner install command |✔️|❌|
| Code is [Mypy](https://mypy-lang.org/) clean |✔️|❌|
| Lines of Python code in repo (2023-09-01)| 529 | 1011|
| Has a cool logo 😉 |✔️|❌|

<!-- markdownlint-disable MD053 -->

[^1]: Could not get this to work

[^2]: With `--detail_level=2`

[^3]: Deliberate, since this format is lossy
