"""CLI entry point for aretomo3-preprocess."""

import argparse
from aretomo3_preprocess.commands import (
    analyse, trim_ts, check_gain_transform, validate_mdoc,
    rename_ts, run_aretomo3, run_aretomo3_per_ts, cryocare, enrich, select_ts,
    imod_mtffilter, topaz_denoise3d, topaz_train, deep_dewedge, deep_dewedge_mw,
    aln_edit, pytom_match,
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
    enrich.add_parser(sub)
    run_aretomo3.add_parser(sub)
    run_aretomo3_per_ts.add_parser(sub)
    analyse.add_parser(sub)
    select_ts.add_parser(sub)
    aln_edit.add_parser(sub)
    trim_ts.add_parser(sub)
    cryocare.add_parser(sub)
    imod_mtffilter.add_parser(sub)
    topaz_denoise3d.add_parser(sub)
    topaz_train.add_parser(sub)
    deep_dewedge.add_parser(sub)
    deep_dewedge_mw.add_parser(sub)
    pytom_match.add_parser(sub)

    args = ap.parse_args()
    args.func(args)
