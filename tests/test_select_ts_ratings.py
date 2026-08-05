"""
Tests for select_ts.py's _load_ratings() timestamped-ratings-CSV fix.

Audit finding: analyse.py's own HTML report globs for the newest
ts_ratings*.csv by mtime (explicitly to support timestamped export
copies), but select_ts.py used to check only the literal 'ts_ratings.csv'
filename. A timestamped export (e.g. ts_ratings_2026-08-01.csv) showed up
correctly in the HTML report but was silently ignored by
--select-by-rating, which fell through to "treat every TS as unrated" and
excluded all of them. Fixed by having both use
shared/discovery.py:most_recent_glob().

Synthetic fixtures (no mounted data required) so these always run.
"""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aretomo3_preprocess.commands.select_ts import _load_ratings


def _write_ratings(path: Path, rows):
    lines = ['ts_name,rating\n']
    lines += [f'{ts},{r}\n' for ts, r in rows]
    path.write_text(''.join(lines))


def test_loads_literal_filename_when_thats_all_that_exists(tmp_path):
    _write_ratings(tmp_path / 'ts_ratings.csv', [('ts-001', 4), ('ts-002', 2)])
    ratings = _load_ratings(tmp_path)
    assert ratings == {'ts-001': 4, 'ts-002': 2}


def test_no_ratings_file_returns_empty_dict(tmp_path):
    assert _load_ratings(tmp_path) == {}


def test_timestamped_export_is_found_when_literal_name_absent(tmp_path):
    """The exact scenario the audit flagged: only a timestamped export
    exists (e.g. downloaded from the HTML report's Export button), no
    plain ts_ratings.csv."""
    _write_ratings(tmp_path / 'ts_ratings_2026-08-01.csv', [('ts-001', 5)])
    ratings = _load_ratings(tmp_path)
    assert ratings == {'ts-001': 5}


def test_prefers_newest_when_multiple_ratings_files_exist(tmp_path):
    _write_ratings(tmp_path / 'ts_ratings.csv', [('ts-001', 1)])
    time.sleep(0.01)
    _write_ratings(tmp_path / 'ts_ratings_newer.csv', [('ts-001', 5)])

    ratings = _load_ratings(tmp_path)
    assert ratings == {'ts-001': 5}


def test_explicit_ratings_file_takes_precedence(tmp_path):
    _write_ratings(tmp_path / 'ts_ratings.csv', [('ts-001', 1)])
    explicit = tmp_path / 'custom.csv'
    _write_ratings(explicit, [('ts-001', 3)])

    ratings = _load_ratings(tmp_path, ratings_file=str(explicit))
    assert ratings == {'ts-001': 3}
