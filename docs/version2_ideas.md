# aretomo3-preprocess v2 — Ideas

## General design principles

### 1. Never silently overwrite output
If an output directory already exists, prompt the user before proceeding:
- Print a clear warning listing what would be overwritten
- Require explicit confirmation (or an `--overwrite` flag to skip the prompt)
- In `--dry-run` mode, warn that the directory exists but do not prompt

### 2. All commands must have `--dry-run`
Every command that writes files, creates symlinks, or modifies anything on disk
must support `--dry-run`:
- **Nothing is written at all** — no output files, no symlinks, no JSON updates, no logs, no CSVs
- Print exactly what would be created/modified/deleted
- Output should be identical in format to the real run (just prefixed or labelled)

### 3. Clear JSON schema documentation with ownership rules

Every field in `aretomo3_project.json` and `alignment_data.json` must be
documented: what it contains, which file/command it comes from, and which
command is allowed to write it.

Write rules:
- Each top-level section in `aretomo3_project.json` is **owned by exactly one command**
- A command may only write its own section
- If a section already exists, the command must warn the user and prompt before overwriting
  (consistent with design principle 1)
- `alignment_data.json` is owned by `analyse` — other commands must not modify it

---

## JSON schema

### `aretomo3_project.json`  (RELION working directory)

| Section | Owner command | Contents | Overwrites? |
|---|---|---|---|
| `project` | project init | `working_dir`, `created`, `last_updated` | `last_updated` always updated; others never overwritten |
| `mdoc_data` | `validate-mdoc` | per-mdoc frame metadata parsed from `.mdoc` files | prompt if exists |
| `rename_ts` | `rename-ts` | grid lookup tables, ts→grid mapping | new grids appended; existing grid numbers never overwritten |
| `input_stacks` | `select-ts` / `run-aretomo3` | paths to input `.mrc` stacks and `_TLT.txt` | prompt if exists |
| `run_aretomo3` | `run-aretomo3` | command, args, timestamp, output dir, log, returncode | prompt if exists |
| `analyse` | `analyse` | summary stats, output dir, timestamp | prompt if exists |

#### `mdoc_data.per_ts[name].frames[acq_order]` fields

| Field | Source |
|---|---|
| `tilt_angle` | mdoc `TiltAngle` |
| `sub_frame_path` | mdoc `SubFramePath` |
| `mdoc_defocus` | mdoc `Defocus` (µm, as written) |
| `target_defocus` | mdoc `TargetDefocus` |
| `datetime` | mdoc `DateTime` |
| `stage_x/y/z` | mdoc `StagePosition` / `StageZ` |
| `exposure_time` | mdoc `ExposureTime` |
| `num_subframes` | mdoc `NumSubFrames` |

---

### `alignment_data.json`  (`<output_dir>/analyse/`)

**Owner:** `analyse`
**Written once per `analyse` run; prompt before overwrite.**

#### Per-tilt-series fields

| Field | Source |
|---|---|
| `file` | input `.mrc` path |
| `width`, `height` | `.mrc` header |
| `total_frames` | `_TLT.txt` row count |
| `alpha_offset` | `.aln` header |
| `beta_offset` | `.aln` header |
| `thickness` | `.aln` header (pixels) |
| `thickness_nm` | computed: `thickness × angpix / 10` |
| `angpix` | `--angpix` arg or mdoc `PixelSpacing` |
| `num_patches` | `.aln` header |
| `dark_frames` | `.aln` `DarkFrame` lines |

#### Per-frame fields (aligned frames only, sorted by tilt angle)

| Field | Source |
|---|---|
| `sec` | `.aln` col 0 (1-indexed section) |
| `rot`, `gmag`, `tx`, `ty`, `smean`, `sfit`, `scale`, `base` | `.aln` cols 1–8 |
| `tilt` | `.aln` col 9 (corrected tilt = nominal + alpha_offset) |
| `overlap_pct` | computed from `tx`/`ty` and image geometry |
| `is_reference` | computed (frame with smallest `|tilt|`) |
| `is_flagged` | computed (`overlap_pct < threshold`) |
| `defocus1_A`, `defocus2_A` | `_CTF.txt` cols 1–2 |
| `mean_defocus_A`, `mean_defocus_um` | `_CTF.txt` col 3 + conversion |
| `astig_A`, `astig_um` | `_CTF.txt` col 4 + conversion |
| `astig_angle_deg` | `_CTF.txt` col 5 |
| `cc` | `_CTF.txt` col 6 |
| `fit_spacing_A` | `_CTF.txt` col 7 |
| `nominal_tilt` | `_TLT.txt` col 0 |
| `acq_order` | `_TLT.txt` col 1 |
| `dose_e_per_A2` | `_TLT.txt` col 2 |
| `z_value` | computed: `acq_order - 1` |
| `cumulative_dose_e_per_A2` | computed: cumulative sum sorted by `acq_order` |
| `tilt_angle` | mdoc (via `z_value`) |
| `sub_frame_path` | mdoc |
| `mdoc_defocus`, `target_defocus` | mdoc |
| `datetime` | mdoc |
| `stage_x`, `stage_y`, `stage_z` | mdoc |
| `exposure_time`, `num_subframes` | mdoc |

---

## Motion metrics from AreTomo3 cmd=0

AreTomo3 cmd=0 outputs per-tilt, per-frame motion trajectories in
`ts-xxx_Log/ts-xxx_MC_GL.csv` with columns:

```
frame_idx  tilt_idx  tilt_angle  pixel_size  shift_x  shift_y
```

Shifts are cumulative (frame 0 = 0,0), units are pixels.

**Proposed metrics per tilt:**
- **Total path length** (Å): sum of frame-to-frame displacements × pixel_size
  — most sensitive to erratic/non-smooth motion
- **Net displacement** (Å): sqrt(x_last² + y_last²) × pixel_size
  — total drift across the movie
- **Early motion** (Å): displacement in the first frame interval
  — beam-induced spike at start of exposure
- **Non-linearity ratio**: total_path / net_displacement
  — ratio > 1 indicates wobble/reversal rather than smooth drift

**Motivation:** DiamondLightSource cryoem-services flags tilts with motion model
coefficients > 1000 (from RELION star files). We can derive an equivalent metric
directly from AreTomo3's MC_GL output.

**Note:** High-tilt acquisitions (>~30°) show near-zero motion because sample is
out of view — these are not informative for motion-based QC.

**Implementation sketch:**
- Add `parse_mc_gl_file()` to `shared/parsers.py`
- Incorporate into `enrich` command (adds motion columns to per-tilt JSON/CSV)
- Optionally flag tilts exceeding a threshold (analogous to AreTomo dark frames)
