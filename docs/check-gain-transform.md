# check-gain-transform

Determine the correct AreTomo3 gain correction parameters (`-RotGain` /
`-FlipGain`) by testing candidate transformations against a subset of your
TIFF movie data.

## When to use this

Run `check-gain-transform` **before** `analyse`.  The results are saved to
`results.json` and can be embedded into the analysis HTML report via the
`--gain-check-dir` flag.

Typical pipeline:

```
aretomo3-editor check-gain-transform \
    --gain  relion/gain_20260213T101027.mrc \
    --frames relion/frames/ \
    --output relion/gain_check/

# → inspect gain_check/report.html, note the recommended flags

aretomo3-editor analyse \
    --input  relion/run001/ \
    --output relion/run001_analysis/ \
    --gain-check-dir relion/gain_check/

# → open run001_analysis/index.html  (Gain Check tab shown first)
```

---

## Scope and limitations

| Property | Supported |
|---|---|
| Detector | K3 (rectangular sensor, e.g. 5760 × 4092) |
| Movie format | TIFF (`*_fractions.tiff`) |
| Gain format | MRC, 32-bit float (mode 2) |
| Software conventions | AreTomo3 |
| Square sensors (Falcon 3/4) | Not yet — see [Future plans](#future-plans) |
| EER movies | Not yet — see [Future plans](#future-plans) |

This command encodes **AreTomo3-specific** flip/rotation semantics.  The
same raw transforms may have different meanings in RELION or cryoSPARC
(different row conventions), so results from this command should **not** be
applied directly to other software.

---

## Why K3 only (for now)

The K3 detector is **rectangular** (width ≠ height).  This means only 4 of
the 8 possible flip/rotation combinations preserve the image dimensions:

| Transform | AreTomo3 flags | numpy equivalent |
|---|---|---|
| none | `-RotGain 0 -FlipGain 0` | — |
| flipud | `-RotGain 0 -FlipGain 1` | `np.flipud` |
| fliplr | `-RotGain 0 -FlipGain 2` | `np.fliplr` |
| rot180 | `-RotGain 2 -FlipGain 0` | `np.rot90(x, 2)` |

`rot90` and `rot270` would swap the width and height, making the gain
incompatible with the movie frames.  For a **square** sensor (Falcon), all 8
transforms are valid, requiring a broader search and dedicated test data.

### AreTomo3 source reference

Flip semantics confirmed from AreTomo3 source (`GFlip2D.cu`,
`CLoadRefs.cpp`):

- `FlipGain 1` → `GFlip2D::Vertical()` → flip around horizontal axis =
  `np.flipud`
- `FlipGain 2` → `GFlip2D::Horizontal()` → flip around vertical axis =
  `np.fliplr`
- `RotGain 1/2/3` → 90° CCW / 180° / 270° CCW

Application order: **rotate → flip → invert**.  Gain is **multiplied** to
each raw frame (`frame × gain`), not divided.  TIFF files are read in the
same orientation by both AreTomo3 and `tifffile.imread()` (no implicit
Y-flip).

---

## Gain file requirements

The gain MRC must be **32-bit float (mode 2)**.  Integer or other modes are
rejected at startup with a clear error message.  If your gain is in a
different format, convert it first with `e2proc2d.py` or a similar tool.

---

## Movie selection

Movies are selected by **acquisition order** parsed from the filename, not
by tilt angle.  For pre-tilted lamella the first-acquired frame may be at a
significant tilt angle, but it still has the most uniform illumination
relative to the gain detector response.

**Filename pattern expected:**
```
Position_1_001_14.00_20260213_171849_fractions.tiff
              ^^^  ^^^^^
              acq  tilt_angle
```

The first `--n-acquisitions` movies (default 12) sorted by acquisition
order are used.  The tilt range of the selected movies is printed so you can
verify the selection makes sense.

### Why not filter by tilt angle?

At high tilt angles the sample is thicker (more beam path through material),
so images are darker and less uniform.  This would bias the flatness metric.
Using acquisition order instead selects the most stable early frames without
needing to assume anything about the pre-tilt of the stage.

---

## Algorithm

1. Load gain MRC; validate mode (float32) and geometry (rectangular = K3).
2. Apply each of 4 transforms to produce `gain_none`, `gain_flipud`,
   `gain_fliplr`, `gain_rot180`.
3. For each selected movie:
   - Load TIFF stack; sum sub-frames → `raw_sum` (float64, prevents
     integer wrap-around).
   - For each transform: `corrected_sum += raw_sum × gain_t`.
4. After accumulation, compute the **coefficient of variation** (CV =
   std / mean) for each corrected sum.  Lower CV = flatter = better.
5. Also compute SSIM between the normalised corrected image and a uniform
   reference (informational, not used for the decision).
6. Best transform = lowest CV.

---

## Arguments

| Argument | Short | Default | Description |
|---|---|---|---|
| `--gain` | `-g` | required | Path to gain MRC file (float32, mode 2) |
| `--frames` | `-f` | required | Directory containing `*_fractions.tiff` |
| `--output` | `-o` | `gain_check/` | Output directory |
| `--n-acquisitions` | `-n` | `12` | Number of lowest-acq-order movies to use |

---

## Outputs

| File | Description |
|---|---|
| `results.json` | Best transform, AreTomo3 flags, all scores, metadata |
| `corrected_averages.png` | 2 × 2 grid of normalised corrected images per transform |
| `cv_vs_nmovies.png` | CV convergence vs movies accumulated, one line per transform |
| `report.html` | Standalone HTML viewer (open directly in a browser) |

### results.json structure

```json
{
  "gain_file":          "/path/to/gain.mrc",
  "frames_dir":         "/path/to/frames/",
  "input_type":         "tiff_mrc_aretomo3",
  "n_movies_tested":    12,
  "acq_range":          [1, 12],
  "tilt_range_deg":     [2.0, 26.0],
  "best_transform":     "flipud",
  "aretomo3_rot_gain":  0,
  "aretomo3_flip_gain": 1,
  "scores": {
    "none":   {"cv": 0.142, "ssim": 0.71},
    "flipud": {"cv": 0.031, "ssim": 0.97},
    "fliplr": {"cv": 0.139, "ssim": 0.72},
    "rot180": {"cv": 0.140, "ssim": 0.72}
  },
  "timestamp": "2026-02-21T10:00:00"
}
```

---

## Interpreting the results

**CV (coefficient of variation):** the ratio of pixel standard deviation to
mean intensity of the accumulated gain-corrected image.  A perfectly uniform
image has CV = 0.  The correct transform should produce a noticeably lower CV
than the alternatives (often 3–5× lower).

**SSIM vs flat:** structural similarity between the corrected image and a
perfectly uniform reference.  Closer to 1.0 = better.  This is a secondary
metric; CV is the decision criterion.

**CV convergence plot:** if the correct transform is genuinely better the
curves should separate clearly within the first few movies and converge
stably.  If the curves are close together or noisy, consider increasing
`--n-acquisitions`.

---

## Integration with `analyse`

Pass the output directory to `analyse --gain-check-dir`:

```bash
aretomo3-editor analyse \
    --input  run001/ \
    --output run001_analysis/ \
    --gain-check-dir gain_check/
```

The `results.json` is embedded in the HTML report as a "Gain Transform Check"
tab shown before the tilt-series viewer.  The two gain-check PNGs are copied
into the analysis output directory so the HTML is fully self-contained.

---

## Future plans

### Square-sensor cameras (Falcon 3/4)

For square detectors all 8 flip/rotation combinations are valid, requiring a
broader search.  A stub (`_run_square_camera`) is present in the source but
raises `NotImplementedError` — it will be implemented once Falcon test data
is available.

### EER movie support

EER (Electron Event Representation) is the native format for Falcon cameras.
EER rendering (dose fractionation, gain application) differs from TIFF and
requires a separate reader.  AreTomo3 EER conventions must also be verified
against the source before implementation.

### cryoSPARC / RELION conventions

These packages may apply gain with different row-order conventions (implicit
Y-flip relative to the raw detector).  A separate subcommand or `--software`
flag would be needed to test those conventions correctly.
