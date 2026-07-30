"""
output_guard.py — shared helper for safe output directory handling.

Prevents accidental overwriting of existing output directories by prompting
the user to choose a new path or pass --clean to remove the existing one.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def check_output_dir(out_dir: Path, clean: bool = False, dry_run: bool = False) -> Path:
    """
    Check whether an output directory already exists and handle it safely.

    Behaviour
    ---------
    - Does not exist          → return out_dir unchanged (caller creates it)
    - Exists + clean=True     → remove it and return out_dir
    - Exists + clean=False    → prompt user interactively:
        · Enter a new path    → return that path (recursively checked)
        · Enter 'clean'       → remove existing dir, return out_dir
        · Ctrl-C / empty      → abort with sys.exit(1)
    - dry_run=True            → only warn, never delete or prompt

    Parameters
    ----------
    out_dir  : Path  Desired output directory
    clean    : bool  If True, silently remove existing directory
    dry_run  : bool  If True, skip deletion/prompting (just warn)

    Returns
    -------
    Path — the resolved output directory to use
    """
    out_dir = Path(out_dir).resolve()

    if not out_dir.exists():
        return out_dir

    # ── Directory exists ──────────────────────────────────────────────────────
    if dry_run:
        # Suppress — caller will include this in the bottom summary
        return out_dir

    if clean:
        print(f'--clean: removing existing output directory: {out_dir}')
        shutil.rmtree(out_dir)
        return out_dir

    # ── Interactive prompt ────────────────────────────────────────────────────
    print()
    print(f'  Output directory already exists:')
    print(f'    {out_dir}')
    print()
    print(f'  Options:')
    print(f'    · Enter a new output path')
    print(f'    · Type "clean" to remove the existing directory and reuse this path')
    print(f'    · Press Ctrl-C to abort')
    print()

    if not sys.stdin.isatty():
        print('ERROR: output directory exists and stdin is not a terminal. '
              'Re-run with --clean to remove it, or choose a different --output path.')
        sys.exit(1)

    while True:
        try:
            answer = input('  > ').strip()
        except (KeyboardInterrupt, EOFError):
            print('\nAborted.')
            sys.exit(1)

        if not answer:
            continue

        if answer.lower() == 'clean':
            print(f'Removing: {out_dir}')
            shutil.rmtree(out_dir)
            return out_dir

        # Treat as a new path — recurse to check it too
        new_dir = Path(answer).resolve()
        return check_output_dir(new_dir, clean=False, dry_run=False)


def check_disk_space(out_dir: Path, min_free_gb: float = 20.0) -> list:
    """
    Warn if free space on out_dir's filesystem is low.

    A precise "enough space for this run" estimate depends on the specific
    job (VolZ/AtBin/dtype/split-sum for AreTomo3; volume count/dtype for a
    segmentation batch) and is easy to get wrong; this only flags an
    absolute low-water mark as a sanity check, not a prediction. (A batch
    tomogram run genuinely crashed from exactly this cause -- see
    project_json.py's atomic-write fix and run_aretomo3.py's post-run
    completeness check -- so even a rough warning here is worth having.)

    Returns a list of warning strings (empty if space looks fine or the
    check itself fails, e.g. the path doesn't exist yet).
    """
    out_dir = Path(out_dir)
    check_dir = out_dir if out_dir.exists() else out_dir.parent
    try:
        usage = shutil.disk_usage(check_dir)
    except OSError:
        return []
    free_gb = usage.free / 1e9
    if free_gb < min_free_gb:
        return [f'Low disk space on {check_dir} filesystem: {free_gb:.1f} GB free '
                f'-- a batch run can consume far more than this']
    return []
