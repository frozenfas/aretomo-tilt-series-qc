"""
Tests for trim_ts.py's _resolve_effective_threshold() -- addresses the
audit finding that --threshold was printed in the run summary but never
used to filter anything: the real overlap-based exclusion (is_flagged)
was already baked into alignment_data.json by analyse's own --threshold
at analysis time. Re-running trim-ts --threshold 70 against data built
with analyse --threshold 80 used to print "70%" while actually filtering
at 80%, with no indication of the mismatch.

Synthetic fixtures (no mounted data required) so these always run.
"""
import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aretomo3_preprocess.commands.trim_ts import _resolve_effective_threshold


def _args(threshold):
    return argparse.Namespace(threshold=threshold)


def test_no_recorded_analyse_run_falls_back_to_cli_arg_silently():
    effective, recorded, warning = _resolve_effective_threshold(
        _args(70.0), {}, Path('alignment_data.json'))
    assert effective == 70.0
    assert recorded is None
    assert warning is None


def test_matching_threshold_no_warning():
    project = {'analyse': {'args': {'threshold': 80.0}}}
    effective, recorded, warning = _resolve_effective_threshold(
        _args(80.0), project, Path('alignment_data.json'))
    assert effective == 80.0
    assert recorded == 80.0
    assert warning is None


def test_mismatched_threshold_warns_and_uses_recorded_value():
    """The exact scenario the audit flagged: trim-ts --threshold 70
    against data actually built with analyse --threshold 80."""
    project = {'analyse': {'args': {'threshold': 80.0}}}
    effective, recorded, warning = _resolve_effective_threshold(
        _args(70.0), project, Path('alignment_data.json'))
    assert effective == 80.0  # the value actually in effect, not 70
    assert recorded == 80.0
    assert warning is not None
    assert '70' in warning and '80' in warning


def test_default_cli_value_against_matching_recorded_value_no_warning():
    """The common case: user never touches --threshold (default 80.0) and
    analyse was also run at its own default -- no spurious warning."""
    project = {'analyse': {'args': {'threshold': 80.0}}}
    effective, recorded, warning = _resolve_effective_threshold(
        _args(80.0), project, Path('alignment_data.json'))
    assert warning is None
