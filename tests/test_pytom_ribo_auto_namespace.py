"""
Tests for pytom_match.py:default_args() and pytom_ribo_auto.py's
_build_pm_namespace() -- addresses the audit finding that
pytom_ribo_auto.py hand-built three separate argparse.Namespace objects
duplicating pytom_match.py's entire CLI surface by hand (a future
attribute added to pytom_match.py's args without a getattr(..., default)
guard at these call sites would silently break) and reached into
pytom_match.py's "private" _find_tomogram directly.

Synthetic fixtures (no mounted data required) so these always run.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aretomo3_preprocess.commands import pytom_match, pytom_ribo_auto


def test_default_args_returns_every_cli_field():
    d = pytom_match.default_args()
    # Spot-check a representative sample across input/matching/extraction/
    # run-control groups rather than the full 50+ field list.
    for key in ('input', 'vol_suffix', 'select_ts', 'template', 'mask',
                'voxel_size', 'particle_diameter', 'gpu', 'angular_search',
                'relion5_compat', 'imod', 'output', 'dry_run', 'extract_only'):
        assert key in d, f'{key} missing from default_args()'
    assert 'help' not in d


def test_default_args_matches_real_parser_defaults():
    d = pytom_match.default_args()
    assert d['output'] == 'pytom_match'
    assert d['vol_suffix'] == ''
    assert d['dry_run'] is False
    assert d['relion5_compat'] is False
    assert d['input'] is None


def test_build_pm_namespace_starts_from_defaults():
    ns = pytom_ribo_auto._build_pm_namespace(input='/some/dir')
    assert ns.input == '/some/dir'
    # Untouched fields come from pytom_match's own defaults, not omitted.
    assert ns.vol_suffix == ''
    assert ns.dry_run is False
    assert ns.output == 'pytom_match'


def test_build_pm_namespace_overrides_win():
    ns = pytom_ribo_auto._build_pm_namespace(output='custom_out', relion5_compat=True)
    assert ns.output == 'custom_out'
    assert ns.relion5_compat is True


def test_build_pm_namespace_has_all_fields_run_might_read():
    """A minimal sanity check that the built Namespace has the same field
    set as a real parsed pytom-match invocation would -- run() reading
    any of these via plain attribute access must not AttributeError."""
    ns = pytom_ribo_auto._build_pm_namespace()
    real_fields = set(pytom_match.default_args().keys())
    assert set(vars(ns).keys()) == real_fields


def test_find_tomogram_is_public():
    """pytom_ribo_auto.py used to reach into pytom_match._find_tomogram
    directly (a private-by-convention internal); now a public API."""
    assert hasattr(pytom_match, 'find_tomogram')
    assert not hasattr(pytom_match, '_find_tomogram')
