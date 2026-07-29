# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`aretomo3-preprocess` is a pre-processing/QC pipeline for AreTomo3 cryo-ET
tilt-series data: raw TIFF/EER movies → motion correction → alignment →
reconstruction, plus wrappers for several downstream tools (pytom-match,
GAPSTOP, MemBrain-seg, Topaz, DeepDeWedge, cryoCARE, RELION5 conversion).
Early-stage, written with Claude Code — review outputs before relying on them.

## Setup and common commands

```bash
conda env create -f environment.yml   # env name in the file: aretomo3-preprocess
conda activate aretomo3-preprocess     # on the dev machine the working env is actually named `aretomo-parse`
pip install -e .
aretomo3-preprocess --help
```

Run the full pipeline via the `aretomo3-preprocess` CLI (installed console
script, or `./aretomo3-preprocess` to run from the repo without installing).
See `README.md` for the full 10-stage walkthrough (`check-gain-transform` →
`validate-mdoc` → `rename-ts` → `run-aretomo3 --cmd 0` → `enrich` → `analyse`
→ `run-aretomo3 --cmd 1` → `analyse` → `select-ts` → `run-aretomo3 --cmd 2`
→ optional `cryocare`/`pytom-match`/`gapstop-match`/etc.).

Tests:
```bash
pytest tests/ -v                         # full suite
pytest tests/test_alpha_offset.py -v      # single file
pytest tests/test_parsing.py::TestParseAln::test_header_values -v   # single test
```
`test_parsing.py` needs real fixture data mounted at
`/mnt/McQueen-002/sconnell/TEST-ARETOMO-PARSE/relion` (auto-skips otherwise via
`skip_if_no_data`/`skip_if_no_mrc`). `test_alpha_offset.py` and
`test_project_json.py` use synthetic fixtures (`tmp_path`) and always run.
There is no lint/type-check config in this repo.

## Architecture

**Command pattern.** Every subcommand lives in `aretomo3_preprocess/commands/*.py`
and exposes `add_parser(subparsers)` + `run(args)`; `cli.py` just registers
each module's parser. When adding a new command, follow this same shape.

**Shared state: `aretomo3_project.json`.** Written to the working directory by
almost every command via `shared/project_json.py` (`load_or_create`,
`update_section`, `update_section_once`). Each command reads what earlier
commands recorded (e.g. `run-aretomo3` reads `gain_check` and `input_stacks`;
`select-ts`/`trim-ts` read `analyse`'s section) so later stages don't need to
repeat CLI args. Writes are atomic (temp file + `os.replace`) — a killed or
concurrent write can no longer truncate the live file; `update_section` also
copies a backup into the command's own output dir on every write, which is
the recovery path if the file is ever hand-corrupted.

**`alignment_data.json`** (written by `analyse` into its output dir) is the
parsed, per-TS/per-frame ground truth — width/height, `alpha_offset`,
`dark_frames`, per-frame `tilt`/`tx`/`ty`/CTF/dose/mdoc fields — that
`trim-ts`, `select-ts`, and `run-aretomo3 --filter-overlap` consume. Its
`frames`/`dark_frames` tilt values are always the raw nominal `.aln` values
(see alpha_offset convention below); don't add corrections when writing to
this structure, `trim-ts` depends on it staying nominal to match IMOD's
`order_list.csv`.

It is fully rebuilt from source `.aln`/`_TLT.txt`/mdoc data on every
`analyse` run (not merged/incremental — this is intentional, it's what lets
`--reuse-plots`/`--refit-lamellae` re-target an existing `--output` dir), so
re-running `analyse` can't destroy the underlying AreTomo3/IMOD output it
was built from. What it *can* do is go stale under already-generated
downstream output with no warning: re-running `analyse` with a different
`--threshold` (or against re-processed `.aln` files) into the same
`--output` dir silently changes which frames are dark/flagged, but any
`trim-ts`/`select-ts` output already generated from the old version isn't
invalidated or regenerated automatically. Re-run `trim-ts`/`select-ts` after
any `analyse` re-run rather than assuming their output is still current.

**Cross-referencing frames — use the identifiers already in the data,
don't re-derive a match.** Every frame dict in `alignment_data.json` (and
the `.aln`/`_TLT.txt`/mdoc files it's built from) carries AreTomo3's own
exact, 1-indexed identifiers for cross-referencing across files:
- `frames[].sec` and `dark_frames[].frame_b` — SEC number in the tilt-sorted
  stack, straight from the `.aln` data rows / `# DarkFrame = frame_a
  frame_b tilt` header lines. This is exactly the same tilt-sorted ordering
  IMOD's `order_list.csv` reconstructs (see `trim_ts.py`'s
  `tilt_sorted_sections`), so `sec`/`frame_b` map directly (1-indexed) onto
  `newstack` section numbers — no angle comparison needed.
- `acq_order` / `z_value` — acquisition-order position (`z_value = acq_order
  - 1`), the key mdoc entries and `order_list.csv`'s `ImageNumber` column
  are indexed by.

Matching frames by nearest tilt angle (or any other approximate/
tolerance-based comparison) instead of one of these exact keys is a bug,
not a stylistic choice: `trim_ts.py`'s `find_sections_by_tilt` did exactly
that, with no de-duplication, so two frames with close-but-different tilts
could silently collapse onto one section — replaced 2026-07-29 with a
direct `sec`/`frame_b` index lookup (`sections_from_sec_numbers`). Before
writing new cross-file matching logic, check whether `sec`/`frame_b`/
`acq_order`/`z_value` already gets you there exactly.

**`shared/` modules** — parsing and cross-command utilities, not one-off
helpers:
- `parsers.py` — `parse_aln_file`/`parse_ctf_file`/`parse_tlt_file`/`parse_mdoc_file`. Always reuse these instead of hand-rolling `.aln`/`.mdoc` parsing.
- `project_json.py` / `project_state.py` — the state file API above, plus resolving `--select-ts`.
- `discovery.py` — volume discovery (`ts-*_Vol.mrc` + legacy `ts-*.mrc` fallback), MRC header dims, `--include`/`--exclude` glob filtering. Used by `membrain_seg.py`, `slabify.py`, `pytom_match.py`, `gapstop_match.py`, `simple_box_mask.py`.
- `denoise_training.py` — EVN/ODD pair discovery, `ts-select.csv` defocus loading, defocus-stratified sampling. Used by `cryocare.py`, `deep_dewedge.py`, `deep_dewedge_mw.py`, `topaz_train.py`.
- `volume_qc.py` — shared HTML/plot generation for QC reports (slabs, orthoslices, picks overlays).
- `output_guard.py`, `geometry.py`, `colours.py` — smaller single-purpose helpers.

If you're about to copy a helper function into a new command file, check
`shared/` first — this codebase has a history of the same function drifting
into 5+ near-identical copies before being consolidated.

**External tool wrappers.** Most non-core commands (`pytom_match.py`,
`gapstop_match.py`, `membrain_seg.py`, `cryocare.py`, `deep_dewedge*.py`,
`topaz_*.py`, `imod_mtffilter.py`) shell out to a separately-installed tool at
a hardcoded default binary path (e.g. `/opt/miniconda3/envs/gapstop/bin/gapstop`,
`/opt/AreTomo3/AreTomo3`), overridable via a CLI flag. When one of these
wrappers needs a fix, check whether the same class of bug exists in its
siblings — `pytom_match.py` and `gapstop_match.py` in particular are
near-twins (both do template matching / particle extraction) and have
historically diverged when a fix landed in one but not the other.

## The alpha_offset convention (read before touching tilt angles)

AreTomo3 never bakes `AlphaOffset` (the milling-pretilt correction) into the
TILT column — not in the `.aln` data rows, not in IMOD's `_st.tlt`, whether
the offset came from AreTomo3's own `-TiltCor` auto-estimate or was set by
hand via `aln-edit`. It is always a separate, header-only value
(`aln['alpha_offset']` from `parse_aln_file`) that each consumer must add
explicitly to get the specimen-referenced (rather than raw nominal
stage-tilt) angle:

- **`relion5_convert.py`, `pytom_match.py`, `gapstop_match.py`** correctly add
  it — this matters for real reconstruction/picking geometry.
- **`analyse.py`'s QC plots/tables deliberately do NOT add it** — nominal
  tilt is stable across re-runs (alpha_offset can change between
  refinements), and this view is for frame-pattern QC, not reconstruction.
- **`aln-edit`** only ever rewrites the `AlphaOffset` header line, never the
  TILT column, matching AreTomo3's own convention.
- **`DarkFrame` header lines are never alpha-corrected either** — their
  `tilt` field is always raw nominal, matching `_TLT.txt`'s `nominal_tilt`.

Also note: AreTomo3 ≥ 2.3.0 changed mdoc parsing — it requires a 5th field
(`ExposureTime`) in strict order after `SubFramePath` per tilt section;
missing or out-of-order fields don't error, they silently merge sections
(wrong tilt count, no warning). `validate-mdoc` and `run-aretomo3`'s
preflight check both simulate this (and the pre-2.3.0 4-field parser) to
catch it ahead of time — see `validate_mdoc.py:check_parser_conformance`.

**Git tags `v1`/`v2`** mark this convention fix: `v1` is the pre-fix state
(what already-completed projects were processed with — don't reprocess them
under v2 assumptions without checking); `v2` is current `main`, with the
fix plus the mdoc-conformance preflight, atomic `project.json` writes, and
the `shared/` consolidation above.

`docs/version2_ideas.md` is a separate, older set of forward-looking design
notes (never-overwrite-output rules, universal `--dry-run`, JSON schema
ownership) — not yet implemented, unrelated to the `v1`/`v2` git tags.
