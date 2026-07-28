"""
Tests for project_json.py's atomic-write fix: a killed/interrupted write
must never leave the live aretomo3_project.json truncated, and a corrupt
file must fail with a clear diagnostic instead of a bare traceback.
"""
import sys
import json
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aretomo3_preprocess.shared import project_json as pj


def test_round_trip_and_no_stray_tmp_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / pj.PROJECT_FILENAME

    pj.update_section('foo', {'a': 1}, path=path)
    data = pj.load_or_create(path)

    assert data['foo'] == {'a': 1}
    assert list(tmp_path.glob('*.tmp*')) == [], \
        'no leftover temp file should remain after a successful write'


def test_update_section_preserves_other_sections(tmp_path):
    path = tmp_path / pj.PROJECT_FILENAME
    pj.update_section('foo', {'a': 1}, path=path)
    pj.update_section('bar', {'b': 2}, path=path)

    data = pj.load(path)
    assert data['foo'] == {'a': 1}
    assert data['bar'] == {'b': 2}


def test_write_survives_simulated_interruption(tmp_path):
    """
    The old plain open(path,'w')+json.dump would leave a truncated file
    behind if interrupted mid-write. With the atomic tmp-file+os.replace
    approach, a failure while writing the temp file must leave the
    original, valid file completely untouched.
    """
    path = tmp_path / pj.PROJECT_FILENAME
    pj.update_section('foo', {'a': 1}, path=path)
    original_bytes = path.read_bytes()

    class BoomJSON:
        @staticmethod
        def dump(data, fh, indent=2):
            fh.write('{"partial": tr')  # write a bit, then blow up
            raise OSError('simulated disk full')

    real_json_dump = json.dump
    json.dump = BoomJSON.dump
    try:
        with pytest.raises(OSError):
            pj._write({'foo': {'a': 2}}, path)
    finally:
        json.dump = real_json_dump

    # Original file must be untouched -- the failed write only ever
    # touched a temp file, never the live path.
    assert path.read_bytes() == original_bytes
    # ...and no leftover temp file from the failed attempt.
    assert list(tmp_path.glob('*.tmp*')) == []


def test_corrupt_json_gives_clear_error_not_traceback(tmp_path, capsys):
    path = tmp_path / pj.PROJECT_FILENAME
    path.write_text('{"project": {"working_dir": "x"}, "foo": {"a": 1')  # truncated

    with pytest.raises(SystemExit):
        pj._read(path)

    out = capsys.readouterr().out
    assert 'corrupt JSON' in out
    assert 'backup' in out.lower()
