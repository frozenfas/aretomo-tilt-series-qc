"""CLI entry point for aretomo3-preprocess."""

import argparse
from aretomo3_preprocess.commands import analyse, trim_ts, check_gain_transform, validate_mdoc


def main():
    ap = argparse.ArgumentParser(
        prog='aretomo3-preprocess',
        description='AreTomo3 tilt-series quality control and editing toolkit',
    )
    sub = ap.add_subparsers(dest='command', metavar='COMMAND')
    sub.required = True

    check_gain_transform.add_parser(sub)
    validate_mdoc.add_parser(sub)
    analyse.add_parser(sub)
    trim_ts.add_parser(sub)

    args = ap.parse_args()
    args.func(args)
