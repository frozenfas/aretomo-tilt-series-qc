"""CLI entry point for aretomo3-editor."""

import argparse
from aretomo3_editor.commands import analyse, trim_ts


def main():
    ap = argparse.ArgumentParser(
        prog='aretomo3-editor',
        description='AreTomo3 tilt-series quality control and editing toolkit',
    )
    sub = ap.add_subparsers(dest='command', metavar='COMMAND')
    sub.required = True

    analyse.add_parser(sub)
    trim_ts.add_parser(sub)

    args = ap.parse_args()
    args.func(args)
