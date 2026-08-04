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

## Tilt-related file formats (read before touching tilt angles)

AreTomo3/IMOD produce several files that all carry per-frame tilt-angle
information, and they are **not interchangeable** — mixing them up, or
assuming one behaves like another, is the direct cause of the alpha_offset
bug documented below. This section is the canonical reference; update it
whenever a new tilt-angle source is added or an assumption about an
existing one changes.

- **`.aln`** (AreTomo3-native, one per TS, e.g. `ts-XXX.aln`): header lines
  (`# AlphaOffset = ...`, `# BetaOffset = ...`, `# RawSize = ...`, `#
  DarkFrame = frame_a frame_b tilt`, ...) followed by one data row per
  aligned frame: `SEC ROT GMAG TX TY SMEAN SFIT SCALE BASE TILT`
  (whitespace-separated, parsed by `parsers.py:parse_aln_file` purely by
  field count/position — not fixed-width, safe to reformat). Rows are in
  **tilt-sorted order**, 1-indexed by `SEC`, matching the section order of
  the corresponding raw stack (`<cmd0_outdir>/ts-XXX.mrc`) and every other
  tilt-sorted file below. `SEC`/`DarkFrame`'s `frame_b` are AreTomo3's own
  exact cross-referencing keys — see "Cross-referencing frames" above.
  **The TILT column's nominal-vs-corrected status depends on `-TiltCor`**
  (see next section) — it is not always raw nominal.
- **IMOD `_st.tlt`** (`ts-XXX_Imod/ts-XXX_st.tlt`, written when
  `run-aretomo3 --out-imod 1`, the pipeline default): one tilt angle per
  line, same tilt-sorted `SEC` order as `.aln`, and its values are always
  **identical** to `.aln`'s own TILT column (verified directly) — because
  it's derived from it, it inherits the same nominal-vs-corrected status.
- **IMOD `_st.xf`** (`ts-XXX_Imod/ts-XXX_st.xf`): the alignment transform,
  one line per frame (same `SEC` order) of `A11 A12 A21 A22 DX DY` —
  rotation+shift only, **no tilt-angle information at all**. Used for
  `newstack -xform` (building aligned/half stacks) and for deriving the
  tilt-axis rotation angle (see `analyse.py`'s `tilt_axis_deg` /
  `frames[0]['rot']`, which actually comes from `.aln`'s own ROT column,
  not `.xf` — the two should agree since AreTomo3 writes both from the
  same alignment solution).
- **AreTomo3's own `_TLT.txt`** (`ts-XXX_TLT.txt`, one per TS): 3 columns
  per tilt-sorted `SEC` row — `nominal_tilt acq_order dose_e_per_A2`
  (`parsers.py:parse_tlt_file`). **Always raw nominal stage tilt,
  regardless of `-TiltCor`** — confirmed directly against the source
  mdoc's own `TiltAngle` for the same frame (exact match). This is the one
  tilt source in this list whose nominal-vs-corrected status never varies.
- **mdoc `TiltAngle`**: the true ground truth, recorded by SerialEM at
  acquisition time, never touched by AreTomo3. Indexed by `ZValue`
  (0-indexed acquisition order) = `acq_order - 1`.

This codebase prefers IMOD-format files over AreTomo3-native ones where
both carry the same information, because IMOD's formats are simpler and
more consistently specified across tool versions — the alpha_offset bug
below is a direct example of an AreTomo3-native behavior (what `-TiltCor`
does to `.aln`'s TILT column) changing/being under-documented in a way an
IMOD-preferring design would have been more robust against.
`relion5_convert.py` already follows this (prefers IMOD `_st.tlt` over
`_TLT.txt`, falling back only if the IMOD file is short); commands that
still read `.aln`'s own TILT column directly (`pytom_match.py`,
`gapstop_match.py`) are known exceptions, not yet migrated — flagged here
rather than fixed silently, since it's a larger change than the
correctness fix below and deserves its own review.

## The alpha_offset convention (corrected 2026-08 — read before touching tilt angles)

**AreTomo3's `-TiltCor` DOES bake `AlphaOffset` (the milling-pretilt
correction) directly into the TILT column** — both `.aln`'s data rows and
`DarkFrame` header lines, and therefore IMOD's `_st.tlt` too, since it's
derived from `.aln`. This corrects an earlier version of this document,
which claimed AreTomo3 never bakes it in; that claim was tested and found
wrong (see Test methodology below). AreTomo3's own `_TLT.txt` is the one
exception — it stays raw nominal regardless of `-TiltCor` (see previous
section).

- **`-TiltCor 0`** (disabled): `AlphaOffset` header reads `0.00`; `.aln`
  TILT column (and `DarkFrame` tilt fields, and IMOD `_st.tlt`) are raw
  nominal, matching the source mdoc's `TiltAngle` exactly.
- **`-TiltCor 1`** (or a fixed value — AreTomo3 also accepts a manual
  offset here): `AlphaOffset` header reads the estimated/given correction;
  **every** TILT value (`.aln` data rows, `DarkFrame` tilt fields, IMOD
  `_st.tlt`) is shifted from what `-TiltCor 0` would have produced by
  exactly that `AlphaOffset` — the header value is a redundant duplicate
  of a correction that's already applied, not a separate correction still
  owed.

**Consequence: consumers reading `.aln`'s TILT column or IMOD's `_st.tlt`
must never add `alpha_offset` on top — doing so double-applies the
correction.** This is the opposite of what this document said before the
2026-08 fix; if you're reading an old comment or an older commit that says
"AreTomo3 never bakes AlphaOffset into the TILT column," it's describing
the pre-fix (wrong) understanding.

- **`relion5_convert.py`**: primary path uses IMOD `_st.tlt`
  (`itlt_list`) directly, no `+ alpha_offset` — already corrected when
  `-TiltCor` was on. Fallback path (only hit if `itlt_list` is
  unexpectedly short) uses `_TLT.txt`'s `nominal_tilt` and *does* add
  `alpha_offset` explicitly, since `_TLT.txt` is always raw nominal.
- **`pytom_match.py`, `gapstop_match.py`**: use `.aln`'s own TILT column
  (`frames[].tilt`) directly, no `+ alpha_offset`.
- **`analyse.py`'s QC plots/tables** use whatever's in `.aln` as-is
  (unchanged behavior) — this is for frame-pattern QC, not reconstruction
  geometry, so it was never adding `alpha_offset` and still doesn't. Note
  the original rationale here ("nominal tilt is stable across re-runs")
  only fully holds for `-TiltCor 0` data; for `-TiltCor 1` data these
  values can shift between re-runs if AreTomo3's own `AlphaOffset`
  estimate changes, same as any other `.aln`-derived value would.
- **`analyse.py`'s `_validate_ts` consistency check** (compares
  `_TLT.txt`'s `nominal_tilt` against `.aln`'s TILT) now correctly checks
  `nominal_tilt + alpha_offset ≈ TILT` instead of plain equality — the old
  check was a real, active bug: it flagged every single frame as
  inconsistent on any `-TiltCor 1` TS (confirmed: 47/47 frames on one real
  TS), a false positive on what's actually normal, expected AreTomo3
  output.
- **`aln-edit`** now bakes its offset into the TILT column too (`.aln`
  data rows *and* the matching IMOD `ts-XXX_Imod/ts-XXX_st.tlt`, if
  present), matching AreTomo3's own `-TiltCor 1` behavior, instead of only
  updating the `AlphaOffset` header. This keeps "never add alpha_offset
  again" true after a manual correction too — the old header-only
  behavior would have left a correction recorded but never actually
  applied anywhere, since no consumer adds it anymore.
- **`ctf_handedness.py`** reads `alpha_offset` directly from `--aln-dir`'s
  own `.aln` (not from `--analysis`'s `alignment_data.json`, which can be
  a *different* AreTomo3 run processed with a different `-TiltCor`
  setting) to correctly label its plots' tilt axis as nominal or
  alpha-offset-corrected per TS. See its `_make_plot()` docstring for the
  full story, including how this resolved a real, previously-confusing
  observation (a defocus-vs-tilt delta plot crossing zero at nominal tilt
  ≈ 0 instead of the naively-expected ≈ `-AlphaOffset` — expected once you
  know the plotted `.tlt` was already alpha-offset-corrected).

### Test methodology (how this was actually confirmed, not just reasoned about)

Compared the same real TS processed twice — `run001` (cmd0, `-TiltCor 0`,
confirmed in `run001.log`) vs `run002` (cmd1, `-TiltCor 1`, confirmed in
`run002-cmd1.log`) — across every TS present in both (172/172 on the
dataset used):
- For each TS, matched `.aln` TILT values by `SEC` (frame counts and dark-
  frame counts were identical between the two runs for every TS checked,
  so `SEC` correspondence is exact, not approximate).
- `run002`'s TILT values were shifted from `run001`'s by exactly that TS's
  own `AlphaOffset`, uniformly across every frame (spread across frames
  < 0.1°) — confirmed for all 172/172 TS, zero exceptions.
- Same test repeated for `DarkFrame` header lines' `tilt` field: 693/693
  dark-frame entries across the dataset matched `nominal_tilt +
  alpha_offset` exactly.
- Cross-checked against the true ground truth (mdoc `TiltAngle`,
  untouched by AreTomo3) for one TS's first-acquired frame: `run001`
  (`-TiltCor 0`) matched the mdoc exactly; `run002` (`-TiltCor 1`) did
  not, by exactly that TS's `AlphaOffset`.
- Independently corroborated against the community-standard
  `Phaips/aretomo3torelion5` conversion script (referenced by the
  TomoGuide tutorials): it reads tilt angles straight from `.tlt` with no
  separate `AlphaOffset`/`-TiltCor` handling anywhere in it — consistent
  with `.tlt` already being the correct, final value to use as-is.

Also note: AreTomo3 ≥ 2.3.0 changed mdoc parsing — it requires a 5th field
(`ExposureTime`) in strict order after `SubFramePath` per tilt section;
missing or out-of-order fields don't error, they silently merge sections
(wrong tilt count, no warning). `validate-mdoc` and `run-aretomo3`'s
preflight check both simulate this (and the pre-2.3.0 4-field parser) to
catch it ahead of time — see `validate_mdoc.py:check_parser_conformance`.

**Git tags `v1`/`v2`** mark an earlier alpha_offset convention fix: `v1` is
the pre-fix state (what already-completed projects were processed with —
don't reprocess them under v2 assumptions without checking); `v2` is where
`relion5_convert.py`/`pytom_match.py`/`gapstop_match.py` started adding
`alpha_offset` explicitly, plus the mdoc-conformance preflight, atomic
`project.json` writes, and the `shared/` consolidation above. **The
2026-08 correction documented above (TILT column already has it baked in
under `-TiltCor 1`) postdates `v2`** — code checked out at the `v2` tag
still has the double-counting bug for `-TiltCor 1` data; only current
`main` has the fix.

`docs/version2_ideas.md` is a separate, older set of forward-looking design
notes (never-overwrite-output rules, universal `--dry-run`, JSON schema
ownership) — not yet implemented, unrelated to the `v1`/`v2` git tags.
