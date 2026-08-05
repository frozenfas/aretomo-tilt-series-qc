"""
Tests for ctf_handedness.py:_resolve_aln_dir() -- addresses the audit
finding that when --analysis is given explicitly but --aln-dir is
omitted, the default fell back to the CURRENT working directory's
most-recently-run analyse invocation's --input, which may be a
DIFFERENT AreTomo3 run (different -TiltCor, different AlphaOffset) than
the one --analysis actually points at. Exactly the run-mismatch class
this file's own extensive comments already warn about, previously
unguarded in the one place it could actually happen silently.

Synthetic fixtures (no mounted data required) so these always run.
"""
import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aretomo3_preprocess.commands.ctf_handedness import _resolve_aln_dir


def _args(aln_dir=None):
    return argparse.Namespace(aln_dir=aln_dir)


def test_explicit_aln_dir_always_wins(tmp_path):
    analysis_dir = tmp_path / 'run002_analysis'
    analysis_dir.mkdir()
    aln_dir, warning = _resolve_aln_dir(_args(aln_dir='/explicit/path'), analysis_dir, {})
    assert aln_dir == Path('/explicit/path')
    assert warning is None


def test_prefers_analysis_dirs_own_backup_snapshot(tmp_path):
    """The exact fix: --analysis's own aretomo3_project.json backup
    (written by update_section's backup_dir mechanism) is self-describing
    and guaranteed to match, so it's preferred over the CWD's live
    project.json even when both are present and disagree."""
    analysis_dir = tmp_path / 'run002_analysis'
    analysis_dir.mkdir()
    (analysis_dir / 'aretomo3_project.json').write_text(json.dumps({
        'analyse': {'args': {'input': '/run002/correct_input'}},
    }))
    cwd_proj = {'analyse': {'args': {'input': '/run005/different_input'}}}

    aln_dir, warning = _resolve_aln_dir(_args(), analysis_dir, cwd_proj)
    assert aln_dir == Path('/run002/correct_input')
    assert warning is None


def test_falls_back_to_cwd_project_with_warning_when_no_backup(tmp_path):
    """No aretomo3_project.json backup in --analysis's own dir (e.g. an
    older run, or alignment_data.json copied in from elsewhere) --
    falls back to the CWD's project.json, but with an explicit warning
    since it can't be verified to actually match."""
    analysis_dir = tmp_path / 'run002_analysis'
    analysis_dir.mkdir()
    cwd_proj = {'analyse': {'args': {'input': '/run005/some_input'}}}

    aln_dir, warning = _resolve_aln_dir(_args(), analysis_dir, cwd_proj)
    assert aln_dir == Path('/run005/some_input')
    assert warning is not None
    assert '--aln-dir' in warning


def test_no_source_at_all_returns_none(tmp_path):
    analysis_dir = tmp_path / 'run002_analysis'
    analysis_dir.mkdir()
    aln_dir, warning = _resolve_aln_dir(_args(), analysis_dir, {})
    assert aln_dir is None
    assert warning is None


def test_corrupt_backup_json_falls_back_gracefully(tmp_path):
    analysis_dir = tmp_path / 'run002_analysis'
    analysis_dir.mkdir()
    (analysis_dir / 'aretomo3_project.json').write_text('{not valid json')
    cwd_proj = {'analyse': {'args': {'input': '/fallback/input'}}}

    aln_dir, warning = _resolve_aln_dir(_args(), analysis_dir, cwd_proj)
    assert aln_dir == Path('/fallback/input')
    assert warning is not None


def test_backup_json_present_but_no_analyse_section_falls_back(tmp_path):
    """A backup snapshot exists but predates the analyse run (or is from
    some other command entirely) -- no analyse.args.input in it, falls
    back the same as if the file didn't exist."""
    analysis_dir = tmp_path / 'run002_analysis'
    analysis_dir.mkdir()
    (analysis_dir / 'aretomo3_project.json').write_text(json.dumps({'gain_check': {}}))
    cwd_proj = {'analyse': {'args': {'input': '/fallback/input'}}}

    aln_dir, warning = _resolve_aln_dir(_args(), analysis_dir, cwd_proj)
    assert aln_dir == Path('/fallback/input')
    assert warning is not None
