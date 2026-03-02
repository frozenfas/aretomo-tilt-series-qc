# aretomo3-preprocess

Pre-processing pipeline for AreTomo3 cryo-ET tilt-series data: from raw TIFF movies to reconstructed tomograms.

> **Early development** — written with Claude Code. Review outputs before relying on them.

---

## Installation

```bash
git clone https://github.com/frozenfas/aretomo-tilt-series-qc.git
cd aretomo-tilt-series-qc
conda env create -f environment.yml
conda activate aretomo-parse
pip install -e .
```

The executable is `aretomo3-preprocess`. Check it works:

```bash
aretomo3-preprocess --help
```

---

## Pipeline overview

```
Stage 0  check-gain-transform   Determine -RotGain / -FlipGain for AreTomo3     [OPTIONAL]
Stage 1  validate-mdoc          Check and repair mdoc files (ExposureDose etc.)
Stage 2  rename-ts              Create ts-XXXX.mdoc symlinks from Position_*.mdoc
Stage 3  run-aretomo3 --cmd 0   Motion correction + initial alignment
Stage 4  enrich                 Add mdoc metadata to alignment JSON              [OPTIONAL]
Stage 5  analyse                Parse .aln outputs; QC plots + HTML
Stage 6  run-aretomo3 --cmd 1   Re-alignment with global or per-lamella TiltAxis/AlignZ
Stage 7  analyse (re-run)       Re-analyse after cmd=1; update QC plots
Stage 8  select-ts              Filter TS by quality; produces ts_selection.csv
Stage 9  run-aretomo3 --cmd 2   Final reconstruction using selected TS
Stage 10 cryocare               Train and apply cryoCARE denoising               [OPTIONAL]
```

All commands write results to `aretomo3_project.json` in the working directory so subsequent commands can auto-fill their arguments.

---

## Full pipeline example

### Stage 0 — Check gain transform *(optional)*

```bash
aretomo3-preprocess check-gain-transform \
    --gain   estimated_gain.mrc \
    --frames frames/ \
    --output gain_check \
    --n-acquisitions 12 \
    --n-movies 250 \
    2>&1 | tee check-gain-transform.log
```

Open `gain_check/report.html`. Note the recommended `-RotGain` / `-FlipGain` flags.

---

### Stage 1 — Validate mdoc files

```bash
# Dry run first — see what needs fixing
aretomo3-preprocess validate-mdoc frames/Position*.mdoc

# Fix missing ExposureDose (most common Tomo5 issue)
aretomo3-preprocess validate-mdoc frames/Position*.mdoc \
    --fix-dose --dose 4.16

# Fix too-short mdocs (restarted acquisition)
aretomo3-preprocess validate-mdoc frames/Position_6*.mdoc \
    --fix-subframes Position_6_SubFramePaths/

# Revalidate — should be 100% PASS
aretomo3-preprocess validate-mdoc frames/Position*.mdoc
```

---

### Stage 2 — Rename mdoc files

```bash
# Dry run to preview
aretomo3-preprocess rename-ts --input frames --dry-run

# Create symlinks: ts-0001.mdoc → Position_1.mdoc etc.
aretomo3-preprocess rename-ts --input frames
```

For multi-session datasets use `--start N` to offset numbering and avoid collisions.

---

### Stage 3 — Run AreTomo3 cmd=0 (motion correction + alignment)

```bash
aretomo3-preprocess run-aretomo3 \
    --in-prefix   frames/ts- \
    --in-suffix   mdoc \
    --output      run001 \
    --cmd         0 \
    --gain        estimated_gain.mrc \
    --flip-gain   1 \
    --rot-gain    0 \
    --apix        1.63 \
    --fm-dose     0.52 \
    --split-sum   1 \
    --gpu         0 1 2 3 \
    --aretomo3    /opt/AreTomo3/AreTomo3 \
    2>&1 | tee run001.log
```

After cmd=0, output stacks are registered in `project.json` so later commands know where they are.

---

### Stage 4 — Enrich alignment data with mdoc metadata *(optional)*

Attaches mdoc fields (dose, datetime, stage position, defocus target) to the
alignment data for use by `analyse` and `select-ts`.

```bash
aretomo3-preprocess enrich \
    --input   run001 \
    --mdocdir frames
```

---

### Stage 5 — Analyse initial alignment results

```bash
aretomo3-preprocess analyse \
    --input     run001 \
    --output    run001_analysis \
    --threshold 80 \
    --mdocdir   frames \
    --angpix    1.63 \
    2>&1 | tee analyse.log
```

Open `run001_analysis/index.html`. The report shows per-TS histograms and a
global summary including the **recommended TiltAxis** (median in-plane rotation)
and **suggested AlignZ** (median sample thickness in pixels), which are saved to
`project.json` and used automatically in the next step.

---

### Stage 6 — Re-align with refined TiltAxis/AlignZ (cmd=1)

cmd=1 skips motion correction and re-runs alignment + reconstruction using the
optimised global or per-lamella parameters from the Stage 5 analysis.

**Option A — single global TiltAxis/AlignZ** (auto-loaded from `project.json`):

```bash
aretomo3-preprocess run-aretomo3 \
    --in-prefix run001/ts- \
    --in-suffix mrc \
    --in-skips  _CTF _Vol _EVN _ODD \
    --output    run002 \
    --cmd       1 \
    --analysis  run001_analysis \
    --split-sum 1 \
    --at-bin    4 8 \
    --apix      1.63 \
    --gpu       0 1 2 3 \
    --aretomo3  /opt/AreTomo3/AreTomo3 \
    2>&1 | tee run002.log
# --tilt-axis and --align-z are auto-filled from run001_analysis/project.json
```

**Option B — per-lamella TiltAxis/AlignZ** (use `run-aretomo3-per-ts` instead):

```bash
aretomo3-preprocess run-aretomo3-per-ts \
    --mrcdir   run001 \
    --output   run002 \
    --analysis run001_analysis \
    --cmd      1 \
    --split-sum 1 \
    --at-bin   4 8 \
    --apix     1.63 \
    --gpu      0 1 2 3 \
    --aretomo3 /opt/AreTomo3/AreTomo3 \
    2>&1 | tee run002.log
# TiltAxis and AlignZ are read per-lamella from run001_analysis/lamella_positions.csv
```

Use Option B when the dataset has multiple lamellae with significantly different
tilt-axis orientations.

---

### Stage 7 — Analyse refined alignment results

```bash
aretomo3-preprocess analyse \
    --input     run002 \
    --output    run002_analysis \
    --threshold 80 \
    --mdocdir   frames \
    --angpix    1.63 \
    2>&1 | tee analyse2.log
```

---

### Stage 8 — Select quality tilt series

```bash
# Use defaults: exclude TS with fewer than 2 pos OR 2 neg frames
aretomo3-preprocess select-ts --analysis run002_analysis

# Add quality filters
aretomo3-preprocess select-ts \
    --analysis       run002_analysis \
    --min-neg-frames 2 \
    --min-pos-frames 2 \
    --min-frames     10 \
    --max-thickness  300 \
    --min-defocus    1.0 \
    --max-defocus    6.0 \
    --output         ts_selection.csv
```

Writes `ts_selection.csv` and saves the selection to `project.json`. Re-run
`analyse` afterwards to see the "Selected only" toggle in the HTML viewer.

---

### Stage 9 — Final reconstruction of selected TS (cmd=2)

cmd=2 uses the existing `.aln` files from run002 and re-runs only the WBP
reconstruction, optionally filtering out low-overlap frames first.

```bash
aretomo3-preprocess run-aretomo3 \
    --in-prefix      run002/ts- \
    --in-suffix      mrc \
    --output         run003 \
    --cmd            2 \
    --analysis       run002_analysis \
    --filter-overlap 80 \
    --at-bin         4 8 \
    --split-sum      1 \
    --apix           1.63 \
    --gpu            0 1 2 3 \
    --aretomo3       /opt/AreTomo3/AreTomo3 \
    2>&1 | tee run003.log
# --select-ts is auto-loaded from project.json
```

---

### Stage 10 — cryoCARE denoising *(optional)*

```bash
aretomo3-preprocess cryocare train \
    --input     run002 \
    --output    cryocare_train \
    --n-vols    8 \
    --gpu       0
# --select-ts is auto-loaded from project.json
```

---

## Command reference

| Command | Stage | Description |
|---|---|---|
| `check-gain-transform` | 0 *(optional)* | Test 4 gain orientations; outputs -RotGain/-FlipGain |
| `validate-mdoc` | 1 | Check mdoc files; fix missing ExposureDose or too-short files |
| `rename-ts` | 2 | Create ts-XXXX.mdoc symlinks from Position_*.mdoc |
| `run-aretomo3 --cmd 0` | 3 | Batch motion correction + initial alignment |
| `enrich` | 4 *(optional)* | Attach mdoc metadata (dose, stage pos, defocus) to .aln output |
| `analyse` | 5 / 7 | Parse .aln outputs; QC plots and interactive HTML |
| `run-aretomo3 --cmd 1` | 6 | Re-alignment with refined TiltAxis/AlignZ (global) |
| `run-aretomo3-per-ts --cmd 1` | 6 | Re-alignment with per-lamella TiltAxis/AlignZ |
| `select-ts` | 8 | Filter TS by quality criteria; saves selection to project.json |
| `run-aretomo3 --cmd 2` | 9 | Final reconstruction of selected TS |
| `trim-ts` | post | Trim IMOD support files to match a subset of tilt series |
| `cryocare` | 10 *(optional)* | Train and apply cryoCARE denoising |

---

## select-ts filters

| Argument | Default | Description |
|---|---|---|
| `--min-neg-frames` | 2 | Min frames with tilt < −threshold (AreTomo3 crash prevention) |
| `--min-pos-frames` | 2 | Min frames with tilt > +threshold (AreTomo3 crash prevention) |
| `--tilt-threshold` | 2.0° | ±boundary between pos/neg counting |
| `--min-frames` / `--max-frames` | — | Total aligned frame count |
| `--min-alpha` / `--max-alpha` | — | Alpha offset (°) |
| `--min-thickness` / `--max-thickness` | — | Estimated sample thickness (nm) |
| `--min-defocus` / `--max-defocus` | — | Reference-frame defocus (μm) |
| `--min-rot` / `--max-rot` | — | Mean in-plane rotation angle (°) |

The CSV output (`ts_selection.csv`) can be passed explicitly as `--select-ts ts_selection.csv`
to any downstream command, or it is auto-loaded from `project.json` when omitted.

---

## Project state file

`aretomo3_project.json` in the working directory records each command run:

```json
{
  "gain_check":  { "best_transform": "flipud", "aretomo3_rot_gain": 0, ... },
  "validate_mdoc": { "n_pass": 156, "n_fail": 0, ... },
  "rename_ts":   { "n_symlinks": 156, "lookup": { "ts-0001.mdoc": "..." } },
  "input_stacks": { "n_stacks": 156, "stacks": { "ts-001": { "path": "..." } } },
  "analyse":     { "output_dir": "run001_analysis", "n_tilt_series": 156, ... },
  "select_ts":   { "n_selected": 149, "n_excluded": 7, "ts_names": [...] },
  ...
}
```

---

## Tests

```bash
pytest tests/ -v
```

Test data lives at `/mnt/McQueen-002/sconnell/TEST-ARETOMO-PARSE/relion` and tests are skipped if that path is not mounted.

---

## License

MIT — see [LICENSE](LICENSE).
