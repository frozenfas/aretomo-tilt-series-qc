"""
Shared project state file — aretomo3_project.json

All commands read from and write to this file in the current working directory.
On first use the working_dir is recorded; subsequent runs verify the cwd matches
so the file is never accidentally updated from the wrong location.

A backup copy is written to each command's output directory after every update,
providing a restore point if the live file is accidentally edited or corrupted.

Usage (from a command's run() function)
-----------------------------------------
    from aretomo3_editor.shared.project_json import update_section, args_to_dict

    update_section(
        section    = 'gain_check',
        values     = {
            'command':   ' '.join(sys.argv),
            'args':      args_to_dict(args),
            'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
            # ... command-specific results ...
        },
        backup_dir = out_dir,   # also copied here for safe-keeping
    )

Reverting from a backup
------------------------
    cp gain_check/aretomo3_project.json .
"""

import sys
import json
import shutil
import datetime
from pathlib import Path


PROJECT_FILENAME = 'aretomo3_project.json'


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as fh:
        return json.load(fh)


def _write(data: dict, path: Path):
    with open(path, 'w') as fh:
        json.dump(data, fh, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_or_create(path: Path = None) -> dict:
    """
    Load the project JSON from the current working directory.

    First call  — file does not exist:
        Creates aretomo3_project.json with working_dir = cwd and returns it.

    Subsequent calls — file exists:
        Checks that the stored working_dir matches the current directory.
        Aborts with a clear error message if there is a mismatch, so the
        file is never accidentally updated from the wrong location.

    Parameters
    ----------
    path : Path, optional
        Explicit path to the project file.  Defaults to
        ``Path.cwd() / 'aretomo3_project.json'``.

    Returns
    -------
    dict : the full project data
    """
    if path is None:
        path = Path.cwd() / PROJECT_FILENAME

    cwd   = str(Path.cwd().resolve())
    data  = _read(path)

    if not data:
        # First run — initialise
        data = {
            'project': {
                'working_dir':  cwd,
                'created':      datetime.datetime.now().isoformat(timespec='seconds'),
                'last_updated': datetime.datetime.now().isoformat(timespec='seconds'),
            }
        }
        _write(data, path)
        print(f'Created project file: {path.name}')
        return data

    # Verify working directory
    stored          = data.get('project', {}).get('working_dir', '')
    stored_resolved = str(Path(stored).resolve()) if stored else ''

    if stored_resolved and stored_resolved != cwd:
        print(f'ERROR: {PROJECT_FILENAME} belongs to a different project directory.')
        print(f'       Expected : {stored}')
        print(f'       Current  : {cwd}')
        print(f'       Run the command from the project directory, or delete')
        print(f'       {PROJECT_FILENAME} to start a new project here.')
        sys.exit(1)

    return data


def update_section(section: str, values: dict,
                   backup_dir: Path = None,
                   path: Path = None):
    """
    Merge values into a named section of the project JSON, then save.

    Only the named section is replaced; all other sections are preserved.
    A backup copy is written to backup_dir (the command's output directory)
    so you can always revert: ``cp <output_dir>/aretomo3_project.json .``

    Parameters
    ----------
    section    : str   top-level key, e.g. ``'gain_check'``, ``'analyse'``
    values     : dict  data to store under ``data[section]``
    backup_dir : Path  if given, copy the updated file here after writing
    path       : Path  explicit path to project file (default: cwd / filename)
    """
    if path is None:
        path = Path.cwd() / PROJECT_FILENAME

    data                         = load_or_create(path)
    data[section]                = values
    data['project']['last_updated'] = (
        datetime.datetime.now().isoformat(timespec='seconds')
    )

    _write(data, path)
    print(f'Project file updated: {path.name}  [{section}]')

    if backup_dir is not None:
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_dir / PROJECT_FILENAME)
        print(f'Backup written to  : {backup_dir / PROJECT_FILENAME}')


def args_to_dict(args) -> dict:
    """
    Convert an argparse Namespace to a JSON-serialisable dict.

    Excludes the ``func`` attribute (a non-serialisable function reference).
    Converts Path objects to strings.
    """
    result = {}
    for k, v in vars(args).items():
        if k == 'func':
            continue
        result[k] = str(v) if isinstance(v, Path) else v
    return result
