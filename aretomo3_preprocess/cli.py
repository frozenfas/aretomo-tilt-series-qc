"""CLI entry point for aretomo3-preprocess."""

import sys
import argparse
from aretomo3_preprocess.commands import (
    analyse, trim_ts, check_gain_transform, validate_mdoc,
    rename_ts, run_aretomo3, run_aretomo3_per_ts, cryocare, enrich, select_ts,
    imod_mtffilter, topaz_denoise3d, topaz_train, deep_dewedge, deep_dewedge_mw,
    aln_edit, pytom_match, membrain_seg, slabify, simple_box_mask, gapstop_match,
    relion5_convert, easymode_seg, pytom_ribo_auto, project_summary,
    ctf_handedness,
)

# Groups shown on the top-level --help listing, purely a navigation aid --
# dispatch itself doesn't care about grouping at all (every subparser is
# still registered flat on `sub`, `aretomo3-preprocess <command> --help`
# always works via that command's own subparser regardless of this list).
# Update this alongside the add_parser() calls below when adding a command.
_HELP_GROUPS = [
    ('Utilities', [
        'check-gain-transform', 'validate-mdoc', 'rename-ts', 'enrich',
        'select-ts', 'aln-edit', 'trim-ts', 'project-summary',
    ]),
    ('TS Alignment', [
        'run-aretomo3', 'run-aretomo3-per-ts', 'analyse',
    ]),
    ('Denoising', [
        'topaz-denoise3d', 'topaz-train', 'cryocare-train', 'cryocare-predict',
        'ddw-train', 'ddw-refine', 'ddw-train-mw', 'ddw-refine-mw', 'imod-mtffilter',
    ]),
    ('Particle Picking', [
        'pytom-match', 'pytom-ribo-auto', 'gapstop-match',
    ]),
    ('STA / RELION5 prep', [
        'membrain-seg', 'slabify', 'simple-box-mask', 'relion5-convert', 'easymode-seg',
        'ctf-handedness',
    ]),
]


def _print_grouped_help(ap, sub):
    """Top-level `aretomo3-preprocess` / `-h` / `--help` listing, grouped
    by pipeline stage instead of argparse's default flat alphabetical-ish
    dump. `aretomo3-preprocess <command> --help` is unaffected -- that's
    handled entirely by argparse via the command's own subparser."""
    help_by_name = {a.dest: a.help for a in sub._choices_actions}
    grouped = {name for _, names in _HELP_GROUPS for name in names}
    ungrouped = [n for n in help_by_name if n not in grouped]

    print(f'usage: {ap.prog} COMMAND [options]')
    print()
    print(ap.description)
    for title, names in _HELP_GROUPS:
        print(f'\n{title}:')
        for name in names:
            text = help_by_name.get(name, '')
            print(f'  {name:<24}{text}')
    if ungrouped:
        # Safety net so a command added to cli.py but not yet sorted into
        # _HELP_GROUPS above still shows up, instead of silently vanishing.
        print('\nOther:')
        for name in sorted(ungrouped):
            print(f'  {name:<24}{help_by_name.get(name, "")}')
    print(f"\nRun '{ap.prog} <command> --help' for command-specific options.")


def main():
    ap = argparse.ArgumentParser(
        prog='aretomo3-preprocess',
        description='AreTomo3 tilt-series quality control and editing toolkit',
        add_help=False,
    )
    ap.add_argument('-h', '--help', action='store_true', help=argparse.SUPPRESS)
    sub = ap.add_subparsers(dest='command', metavar='COMMAND')
    sub.required = False

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
    membrain_seg.add_parser(sub)
    slabify.add_parser(sub)
    simple_box_mask.add_parser(sub)
    gapstop_match.add_parser(sub)
    relion5_convert.add_parser(sub)
    easymode_seg.add_parser(sub)
    pytom_ribo_auto.add_parser(sub)
    project_summary.add_parser(sub)
    ctf_handedness.add_parser(sub)

    # No command given at all, or top-level -h/--help (as opposed to
    # `<command> --help`, which argparse already routes to that command's
    # own subparser before we'd ever see it here) -> grouped listing.
    # Exit codes match argparse's own convention: 0 for help explicitly
    # requested, 2 for a missing required argument (bare invocation).
    if len(sys.argv) == 1:
        _print_grouped_help(ap, sub)
        sys.exit(2)
    if sys.argv[1] in ('-h', '--help'):
        _print_grouped_help(ap, sub)
        sys.exit(0)

    args = ap.parse_args()
    if args.command is None:
        _print_grouped_help(ap, sub)
        sys.exit(2)
    args.func(args)
