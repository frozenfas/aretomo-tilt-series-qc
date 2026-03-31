# Installing gapstop

Template matching wrapper for cryo-ET using the STOPGAP per-tilt wedge model,
accelerated with JAX on GPU.

- Paper: https://www.nature.com/articles/s41467-024-47839-8
- GitLab: https://gitlab.mpcdf.mpg.de/bturo/gapstop_tm
- Docs: https://bturo.pages.mpcdf.de/gapstop_tm/

Installed at: `/opt/miniconda3/envs/gapstop/`

## Requirements

- CUDA 12.x (system has CUDA 12.6 at `/usr/local/cuda-12.6/`)
- NVIDIA GPU (system has 4× RTX A4000)

## Installation (run as admin)

```bash
conda create -n gapstop python=3.11 -y
conda activate gapstop

# MPI support
conda install -c conda-forge mpi4py openmpi -y

# JAX with CUDA 12
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# gapstop-tm
pip install "gapstop-tm @ git+https://gitlab.mpcdf.mpg.de/bturo/gapstop_tm.git"

# cryoCAT (required for wedge list generation)
pip install "cryocat @ git+https://github.com/turonova/cryoCAT.git"

# starfile
pip install starfile
```

## Verification

```bash
/opt/miniconda3/envs/gapstop/bin/gapstop --help
/opt/miniconda3/envs/gapstop/bin/python -c "import jax; print(jax.devices())"
/opt/miniconda3/envs/gapstop/bin/python -c "import cryocat; print('cryocat OK')"
```

Expected output for `jax.devices()`: `[CudaDevice(id=0), CudaDevice(id=1), CudaDevice(id=2), CudaDevice(id=3)]`

## Usage

gapstop takes a STAR parameter file as input rather than command-line arguments:

```bash
gapstop run_tm tm_param.star          # single node
gapstop run_tm -n 4 tm_param.star     # 4 GPU tiles
```

See `aretomo3-preprocess gapstop-match` (not yet implemented) for a wrapper
that generates the parameter file and wedge list from AreTomo3 output.
