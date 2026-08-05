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

**`frame_lookup` (project.json section) — the canonical SEC ↔ acq_order/
z_value bridge, build once, don't re-derive.** Cross-referencing a frame
between `SEC` and `acq_order`/`z_value` used to be re-derived independently
in at least 7 files (`analyse.py`, `check_gain_transform.py`, `enrich.py`,
`gapstop_match.py`, `pytom_match.py`, `relion5_convert.py`, `select_ts.py`)
— exactly the kind of repeat re-derivation that produced the
`find_sections_by_tilt` bug above. `frame_lookup` (added 2026-08) closes
this off for new code:

- **Schema**: `project.json['frame_lookup']['per_ts'][ts_name] = {n_total,
  frames, dark_secs, validated, frames_dir}`. `frames` is `{str(sec):
  {z_value, sub_frame_path}}` for *every* SEC (aligned + dark). `dark_secs`
  is the list of dark SECs. `validated` records whether `_TLT.txt`'s
  `nominal_tilt + alpha_offset` agreed with `.aln`'s TILT column (both
  frames and `DarkFrame` lines) to within 0.05° when this entry was built
  — a consistency check via tilt-angle agreement, not the cross-referencing
  mechanism itself (that's still exact SEC/`frame_b` keys throughout).
- **Filename and frames_dir are captured at build time, not composed at
  read time.** An earlier version of this design deliberately left
  `sub_frame_path` out of `frame_lookup` (to avoid duplicating
  `mdoc_data`) and composed it lazily in `resolve_frame()` via
  `get_ts_to_original_stem()` + `mdoc_data` on every call. Reconsidered
  after hitting real fragility from it: a real-project test where
  `mdoc_data` had been registered from a non-standard directory (not the
  actual `rename-ts` symlink dir) made `resolve_frame()` silently return
  `sub_frame_path=None` even though `frame_lookup`'s own data was
  perfectly valid — a live dependency on `rename_ts`/`mdoc_data` being
  correct *every time a frame is resolved*, not just once. Baking the
  filename in when `register_frame_lookup()` runs (composing with
  whatever `mdoc_data` is registered *then*) makes a registered entry
  self-contained — a consumer that only needs ONE frame at a time (e.g.
  `enrich.py`'s SEC-for-a-given-`acq_order` lookup) can read only
  `frame_lookup`, no live join needed. `relion5_convert.py` itself still
  reads `mdoc_data` directly (all frames for a TS in one dict, since it
  needs every frame's `target_defocus` too, a field `frame_lookup`
  deliberately doesn't carry — see the migration note below) rather than
  calling `resolve_frame()` per-frame in its per-tilt loop, which would
  mean one project.json re-read per tilt instead of one per TS. If
  `mdoc_data` wasn't registered yet, `sub_frame_path`/`frames_dir` are
  `None` for that run — re-run `enrich --frame-lookup --force` after
  registering `mdoc_data` to backfill them.
- **frames_dir + the co-location assumption.** AreTomo3 requires raw movie
  files to live in the same directory as their mdoc — confirmed both by
  `rename-ts`'s own symlinks being created "alongside the originals" (see
  its docstring) and, directly, by `validate-mdoc --check-frames` (below).
  `frames_dir` is recorded once per TS in `mdoc_data.per_ts[stem]` by
  `validate-mdoc` (the validated mdoc's own resolved parent directory —
  never whatever directory `SubFramePath` itself encodes, which is
  typically a stale acquisition-PC Windows/UNC path), and copied into
  `frame_lookup` by `register_frame_lookup()`. `resolve_frame()` composes
  `frames_dir` with `sub_frame_path`'s filename (basename only — its own
  directory is discarded) into a `frame_path` key: a full, real, usable
  movie path with no further plumbing needed downstream.
- **`validate-mdoc --check-frames`** verifies every `SubFramePath`-
  referenced movie actually exists next to the mdoc — a real, fixable
  problem (a missing/moved/not-yet-transferred movie) that no other check
  catches, since the mdoc's own fields can be perfectly well-formed while
  referencing a movie that isn't there. On by default at the CLI (use
  `--no-check-frames` for workflows that validate mdocs before movies
  finish transferring); `validate_file()`'s own function-level default
  stays `False` so other callers (e.g. `run-aretomo3`'s preflight, which
  doesn't pass `check_frames`) are unaffected. Not fixable by any other
  `--fix-*` flag, so it always keeps `success=False` regardless of which
  other fixes succeed on the same run.
- **Built by** `shared/project_state.py:register_frame_lookup(out_dir)`,
  parsing that directory's `ts-*.aln` (for `dark_frames`, `alpha_offset`)
  + matching `ts-*_TLT.txt` (for the SEC↔z_value bridge, covering all SECs
  in one parse) per TS, composed with the already-registered `mdoc_data`
  for filenames and `frames_dir`. Wired into `run-aretomo3 --cmd 0`'s
  completion (same integration point as `input_stacks`' auto-fill) and
  into `enrich --frame-lookup <dir>` (the manual/force-overwrite escape
  hatch, same pattern as `enrich`'s other sections). Safe to call
  repeatedly — merges into existing `per_ts` entries.
- **Read via** `get_frame_lookup(ts_name)` (raw section) or
  `resolve_frame(ts_name, sec=... | z_value=... | acq_order=...)` (give
  exactly one identifier, get `{sec, z_value, acq_order, is_dark,
  sub_frame_path, frame_path}` back). Returns `None` — never raises —
  when the TS isn't registered or the given id isn't found.
- **Migration is intentionally partial**: the other 6 files listed above
  (all except `enrich.py`, whose own defocus-lookup responsibility was
  removed rather than migrated — see the `defocus_data` note below) still
  re-derive independently — known candidates for a future pass, not
  migrated in this change (same reasoning as `pytom_match.py`/
  `gapstop_match.py` not yet preferring IMOD `_st.tlt`, flagged in the
  alpha_offset section below rather than migrated all at once).
  `relion5_convert.py`'s `_load_mdoc_from_project()` is a partial
  exception: it still reads `mdoc_data` directly rather than
  `frame_lookup` (see above for why), but its `frames_dir` resolution now
  prefers `mdoc_data.per_ts[stem].frames_dir` directly, falling back to
  the older `rename_ts.lookup`-derived path only for projects validated
  before that field existed.

**`defocus_data` (project.json section) — removed, deliberately not
replaced by anything cached.** `enrich --defocus-data` used to parse
per-TS reference defocus from `_CTF.txt`/`_TLT.txt` into a project.json
section, read by `imod_mtffilter.py` (its only consumer — confirmed via a
full-repo `get_defocus_data()` grep before removing it) as a fallback
behind `ts-select.csv`'s optional `ref_defocus_um` column. Removed
because a cached defocus value is a *worse* fit for this codebase's
"don't let state go silently stale" concerns than `frame_lookup`'s
SEC↔z_value/filename data: defocus estimates change every time AreTomo3/
CTFFIND is re-run, unlike SEC↔z_value correspondence or filenames (which
are structurally fixed once acquired). `imod_mtffilter.py` now calls
`shared/parsers.py:compute_reference_defocus(ctf_dir)` directly on its own
already-required `--input` directory, computed fresh every invocation,
never persisted anywhere — see its docstring for the full priority order
(`ts-select.csv` override, if present, still wins first; then this; then
`--defocus`).

**Design principle — build a centralized lookup table for any cross-file
index correspondence used by more than one command, rather than letting
each consumer re-derive it — but only for data that's genuinely stable
once written (SEC↔z_value, filenames).** For data that changes across
re-runs of the same tool (defocus estimates, anything CTF-refinement-
dependent), compute fresh from the relevant already-required input
directory instead of caching in project.json — a cache of that kind of
data has no way to know it's gone stale. `frame_lookup` is the concrete
example of the first case; `compute_reference_defocus` (used directly,
never cached) is the concrete example of the second.

**`select_ts.py` had a third, independent reference-defocus computation**
— `_compute_ts_stats()` read `alignment_data.json`'s own cached
per-frame `mean_defocus_um` (whatever `analyse` last wrote), a second
"generation" of the same underlying CTFFIND data separate from
`compute_reference_defocus`, with its own staleness window. Confirmed on
real data (156 TS) that the two do silently disagree in practice — every
TS showed a small but consistent difference between the cached and
freshly-recomputed value. Fixed with an optional `--input DIR` flag: when
given, `compute_reference_defocus(DIR)` is computed once for the whole
batch and takes priority over `alignment_data.json`'s cached value per TS
(falling back to it when a TS isn't found in the fresh scan). Omitted by
default — unlike `imod_mtffilter.py` (which always has a required
`--input` already in hand), `select_ts.py`'s only required input is
`--analysis`, so this needed a genuinely new, opt-in flag rather than a
free substitution.

**`shared/` modules** — parsing and cross-command utilities, not one-off
helpers:
- `parsers.py` — `parse_aln_file`/`parse_ctf_file`/`parse_tlt_file`/`parse_mdoc_file`/`check_nominal_tilt_consistency`/`compute_reference_defocus`. Always reuse these instead of hand-rolling `.aln`/`.mdoc` parsing.
- `project_json.py` / `project_state.py` — the state file API above, plus resolving `--select-ts`, plus `frame_lookup`'s `register_frame_lookup`/`get_frame_lookup`/`resolve_frame`.
- `discovery.py` — volume discovery (`ts-*_Vol.mrc` + legacy `ts-*.mrc` fallback), MRC header dims (`mrc_dims`) and pixel size (`mrc_pixel_size`, struct-based, no `mrcfile` dependency), ts-name-from-volume-filename extraction (`ts_name_from_vol`, EVN/ODD-aware), SerialEM fraction-movie-filename parsing (`parse_fraction_filename` — acq_order/tilt_angle from `..._NNN_TILT_YYYYMMDD_HHMMSS_fractions.tif[f]`; consolidates `check_gain_transform.py`'s and `validate_mdoc.py`'s own independent regexes, which used to accept different strings — one required exactly `.tiff` with a looser tilt token, the other accepted `.tif`/`.tiff` with a strict decimal token), newest-by-mtime glob lookup (`most_recent_glob`, used for `ts_ratings*.csv`/`ts_comments*.csv` by both `analyse.py` and `select_ts.py`), per-TS threshold CSV loading (`load_threshold_csv`), `--include`/`--exclude` glob filtering. Used by `membrain_seg.py`, `slabify.py`, `pytom_match.py`, `gapstop_match.py`, `simple_box_mask.py`, `ctf_handedness.py`, `imod_mtffilter.py`, `pytom_ribo_auto.py`, `topaz_denoise3d.py`, `check_gain_transform.py`, `validate_mdoc.py`, `analyse.py`, `select_ts.py`.
- `denoise_training.py` — EVN/ODD pair discovery, `ts-select.csv` defocus loading, defocus-stratified sampling. Used by `cryocare.py`, `deep_dewedge.py`, `deep_dewedge_mw.py`, `topaz_train.py`.
- `volume_qc.py` — shared HTML/plot generation for QC reports (slabs, orthoslices, picks overlays).
- `output_guard.py`, `geometry.py`, `colours.py` — smaller single-purpose helpers.

If you're about to copy a helper function into a new command file, check
`shared/` first — this codebase has a history of the same function drifting
into 5+ near-identical copies before being consolidated.

**Three mdoc parsers exist on purpose — not unreconciled duplication, but
each answering a genuinely different question:**
1. **`validate_mdoc.py`'s `_simulate_aretomo3`** — a hand-rolled state
   machine that faithfully mimics AreTomo3's own destructive C++ mdoc
   parser (including its bugs, e.g. field-order sensitivity), to predict
   whether AreTomo3 itself will successfully load a given file. This one
   *cannot* be replaced by a "real" parser without losing the entire
   point of the check.
2. **`shared/parsers.py:parse_mdoc_file`** — the real, robust parse via
   the `mdocfile` library. The only one of the three that actually
   populates `mdoc_data` in project.json; every downstream command reads
   *this* parser's output, never the other two's.
3. **`run_aretomo3.py:_read_mdoc_metadata`** — a cheap first-occurrence
   regex scan for exactly 3 scalar fields (PixelSpacing/Voltage/
   SubFramePath), used in a loop over every mdoc in a batch (potentially
   hundreds of files) purely for cross-file consistency warnings. Using
   `parse_mdoc_file` here would mean parsing every `ZValue` section of
   every file just to read 3 values that don't vary within one mdoc —
   real, unnecessary overhead at batch scale, not laziness.

**The one real risk this split creates** — a file that "passes" #1's
simulation but that #2 parses differently, meaning project.json's
`mdoc_data` might not reflect what AreTomo3 will actually do with the
file — is what `validate_mdoc.py:check_mdocfile_agreement()` (added
2026-08) directly guards against: when a file passes the simulation +
order checks, it's cross-checked against `parse_mdoc_file`'s own section
count, and a disagreement blocks `success` the same way a missing frame
(`--check-frames`) does. Verified zero disagreements across 484 real
mdocs before making this blocking, so it's a defensive guard against a
previously-flagged theoretical risk, not a currently-observed failure
mode.

**`register_mdoc_data()` (`project_state.py`) — the one place that
merges a parsed mdoc into `mdoc_data`.** `enrich.py --mdoc-data` and
`validate_mdoc.py`'s own save-on-pass step used to each hand-roll this
"parse → build `per_ts` entry → merge with existing → write" logic
independently, and had drifted apart on two real edge cases: only
`enrich.py`'s version stripped stale `ts-\d+`-keyed entries (debris from
an older key-resolution strategy — a bare `ts-123` key is never a real
original stem), and only `validate_mdoc.py`'s version recorded
`frames_dir`. Both call sites now share one implementation, so neither
gap can silently reappear in just one of the two paths. Keys each entry
by `path.resolve().stem` — correct whether `path` is a renamed
`ts-XXX.mdoc` symlink (resolves through to the original `Position_N`
stem at the filesystem level) or an original mdoc file passed directly,
with no `rename_ts.lookup` project.json dependency needed either way
(stronger than the old `get_ts_to_original_stem()`-based lookup
`validate_mdoc.py` used, which would silently fall back to the wrong key
if `rename_ts.lookup` wasn't registered yet).

**External tool wrappers.** Most non-core commands (`pytom_match.py`,
`gapstop_match.py`, `membrain_seg.py`, `cryocare.py`, `deep_dewedge*.py`,
`topaz_*.py`, `imod_mtffilter.py`) shell out to a separately-installed tool at
a hardcoded default binary path (e.g. `/opt/miniconda3/envs/gapstop/bin/gapstop`,
`/opt/AreTomo3/AreTomo3`), overridable via a CLI flag. When one of these
wrappers needs a fix, check whether the same class of bug exists in its
siblings — `pytom_match.py` and `gapstop_match.py` in particular are
near-twins (both do template matching / particle extraction) and have
historically diverged when a fix landed in one but not the other. Two
confirmed instances: their `_read_ts_metadata()`'s missing-`_TLT.txt`-SEC
guard (gapstop's was an uncaught `KeyError` where pytom's already raised
a caught `ValueError` — fixed 2026-08), and `_write_wedge_list()`'s
missing-CTF-entry handling (gapstop silently wrote `NaN` defocus for just
that frame instead of failing the TS, where pytom already raised — also
fixed 2026-08, same fail-loud-not-silent-corruption principle). Their
`_load_threshold_csv()` was also a byte-for-byte duplicate, now
`shared/discovery.py:load_threshold_csv()`.

**`pytom_ribo_auto.py` drives `pytom_match.py` programmatically, not via
its CLI** — it calls `pytom_match.run(pm_ns)` directly with a hand-built
`argparse.Namespace`, three separate times (the handedness check, a
`--reextract`-only path, and the main full run). Each of those used to
hand-duplicate `pytom_match.py`'s entire CLI surface, restating most
fields at whatever their already-current default happened to be, just to
have a "complete" Namespace — a real, previously-unguarded risk: a new
field `pytom_match.py`'s `run()` reads via plain attribute access (not
`getattr(..., default)`) would silently `AttributeError` at every
hand-built call site until each was updated to match, and the three
would drift independently in the meantime (same "diverged twin" pattern
as `pytom_match.py`/`gapstop_match.py` above, just across a command
boundary instead of within one file). Fixed via
`pytom_match.py:default_args()` — introspects the real parser's own
argparse actions rather than a hand-maintained copy — and
`pytom_ribo_auto.py:_build_pm_namespace(**overrides)`, which starts from
that and applies only the genuine overrides each call site needs. Also
made `pytom_match.py:find_tomogram()` (was `_find_tomogram`) a proper
public function, since `pytom_ribo_auto.py` calls it directly too.

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

**Same nominal-vs-measured distinction applies to defocus, not just tilt.**
`parse_mdoc_file`'s `nominal_defocus` (mdoc's `Defocus` field) and
`target_defocus` (mdoc's `TargetDefocus`) are both straight from
SerialEM, never touched by AreTomo3/CTFFIND — same category as
`nominal_tilt`. The actual measured defocus for a frame is
`parse_ctf_file`'s `mean_defocus_um` (CTFFIND's fit, via `_CTF.txt`) — a
different thing, keep them conceptually separate. `nominal_defocus` was
renamed from `mdoc_defocus` for exactly this reason (2026-08): a bare
`defocus`-ish name next to a real CTFFIND measurement risks the same
ambiguity that already caused a real bug for tilt. General rule for any
future field: if it's read directly from the mdoc, unprocessed, prefix it
`nominal_`; if it's something this codebase measured or computed, don't.

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
