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
Stage 0  check-gain-transform   Determine -RotGain / -FlipGain for AreTomo3
Stage 1  validate-mdoc          Check and repair mdoc files (ExposureDose etc.)
Stage 2  rename-ts              Create ts-XXXX.mdoc symlinks from Position_*.mdoc
Stage 3  run-aretomo3 --cmd 0   Motion correction + alignment (AreTomo3 batch)
Stage 4  enrich                 Add mdoc metadata to alignment JSON
Stage 5  analyse                Parse .aln outputs; produce QC plots + HTML
Stage 6  select-ts              Filter TS by quality; produces ts_selection.csv
Stage 7  run-aretomo3 --cmd 2   Final reconstruction using selected TS
Stage 8  cryocare               Train and apply cryoCARE denoising (optional)
```

All commands write results to `aretomo3_project.json` in the working directory so subsequent commands can auto-fill their arguments.

---

## Full pipeline example

### Stage 0 — Check gain transform (optional)

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

### Stage 4 — Enrich alignment data with mdoc metadata

```bash
aretomo3-preprocess enrich \
    --input  run001 \
    --mdocdir frames
```

---

### Stage 5 — Analyse alignment results

```bash
aretomo3-preprocess analyse \
    --input     run001 \
    --output    run001_analysis \
    --threshold 80 \
    --mdocdir   frames \
    --angpix    1.63 \
    2>&1 | tee analyse.log
```

Open `run001_analysis/index.html` for the interactive report.

---

### Stage 6 — Select quality tilt series

```bash
# Use defaults: exclude TS with fewer than 2 pos OR 2 neg frames
aretomo3-preprocess select-ts --analysis run001_analysis

# Add quality filters
aretomo3-preprocess select-ts \
    --analysis       run001_analysis \
    --min-neg-frames 2 \
    --min-pos-frames 2 \
    --min-frames     10 \
    --max-thickness  300 \
    --min-defocus    1.0 \
    --max-defocus    6.0 \
    --output         ts_selection.csv
```

Writes `ts_selection.csv` and saves selected TS names to `project.json`. Re-run `analyse` afterwards to see the "Selected only" toggle in the HTML viewer.

---

### Stage 7 — Run AreTomo3 cmd=2 (reconstruction only, selected TS)

```bash
aretomo3-preprocess run-aretomo3 \
    --in-prefix run001/ts- \
    --in-suffix mrc \
    --output    run002 \
    --cmd       2 \
    --analysis  run001_analysis \
    --filter-overlap 80 \
    --at-bin    4 8 \
    --split-sum 1 \
    --apix      1.63 \
    --gpu       0 1 2 3 \
    --aretomo3  /opt/AreTomo3/AreTomo3 \
    2>&1 | tee run002.log
# --select-ts is auto-loaded from project.json
```

---

### Stage 8 — cryoCARE denoising (optional)

```bash
aretomo3-preprocess cryocare train \
    --input     run002 \
    --output    cryocare_train \
    --n-vols    8 \
    --gpu       0
# --select-ts is auto-loaded from project.json
```

---

## Per-lamella reconstruction (run-aretomo3-per-ts)

For datasets with multiple lamellae at different tilt-axis angles, use the
per-TS command after running `analyse` (which clusters by stage position):

```bash
aretomo3-preprocess run-aretomo3-per-ts \
    --mrcdir  run001 \
    --output  run002 \
    --analysis run001_analysis \
    --cmd     1 \
    --at-bin  4 8 \
    --apix    1.63 \
    --gpu     0 1 2 3 \
    --aretomo3 /opt/AreTomo3/AreTomo3 \
    2>&1 | tee run002.log
# --select-ts is auto-loaded from project.json
```

---

## Command reference

| Command | Stage | Description |
|---|---|---|
| `check-gain-transform` | 0 | Test 4 gain orientations; outputs -RotGain/-FlipGain |
| `validate-mdoc` | 1 | Check mdoc files; fix missing ExposureDose or too-short files |
| `rename-ts` | 2 | Create ts-XXXX.mdoc symlinks from Position_*.mdoc |
| `enrich` | 3.5 | Attach mdoc metadata to alignment .aln output |
| `run-aretomo3` | 3/7 | Batch AreTomo3 wrapper (all cmds) |
| `run-aretomo3-per-ts` | 3/7 | Per-TS AreTomo3 wrapper with per-lamella parameters |
| `analyse` | 5 | Parse .aln outputs; QC plots and interactive HTML |
| `select-ts` | 6 | Filter TS by quality criteria; saves selection to project.json |
| `trim-ts` | post | Trim IMOD support files to match a subset of tilt series |
| `cryocare` | 8 | Train and apply cryoCARE denoising |

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
