"""CLI entry point for aretomo3-preprocess."""

import argparse
from aretomo3_preprocess.commands import (
    analyse, trim_ts, check_gain_transform, validate_mdoc,
    rename_ts, run_aretomo3, run_aretomo3_per_ts,
)


def main():
    ap = argparse.ArgumentParser(
        prog='aretomo3-preprocess',
        description='AreTomo3 tilt-series quality control and editing toolkit',
    )
    sub = ap.add_subparsers(dest='command', metavar='COMMAND')
    sub.required = True

    check_gain_transform.add_parser(sub)
    validate_mdoc.add_parser(sub)
    rename_ts.add_parser(sub)
    run_aretomo3.add_parser(sub)
    run_aretomo3_per_ts.add_parser(sub)
    analyse.add_parser(sub)
    trim_ts.add_parser(sub)

    args = ap.parse_args()
    args.func(args)
