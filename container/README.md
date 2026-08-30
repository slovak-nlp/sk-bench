# Container Scripts

Scripts for building and running the sk-bench evaluation environment inside a container with CUDA support. Works with both [Apptainer](https://apptainer.org/) and [Singularity](https://docs.sylabs.io/guides/latest/user-guide/) — the runtime is auto-detected. The container is used to run `eval_vllm.py` evaluations in isolated, reproducible environments, including on SLURM HPC clusters.

## Directory Overview

```
container/
├── container-env.sh                 # Shared runtime detection (sourced by other scripts)
├── mycontainer.def                  # Container definition (CUDA 12.8.1 + Ubuntu 24.04)
├── build.sh                         # Build the container image
├── shell.sh                         # Open an interactive shell inside the container
├── install-vllm.sh                  # Install Python dependencies inside the container
├── run-eval.sh                      # Run evaluation inside the container
├── eval_entry.sh                    # Evaluation entry point (called inside the container)
├── rerun.sh                         # Re-run evaluation from saved detail files
└── slurm/
    ├── README.md                    # How to submit evaluation jobs on HPC
    ├── submit-perun.slurm           # SLURM job for Perun HPC cluster
    └── submit-devana.slurm          # SLURM job for Devana cluster
```

## Container Runtime

All scripts source [`container-env.sh`](container-env.sh), which auto-detects the container runtime:

1. If `CONTAINER_CMD` is already set in the environment, use it as-is.
2. Try `apptainer` (preferred).
3. Fall back to `singularity`, attempting `module load singularity` on HPC clusters.

To force a specific runtime:
```bash
CONTAINER_CMD=singularity bash container/run-eval.sh Qwen/Qwen3-0.6B
```

## Setup

### 1. Build the container image

Run from the repository root:

```bash
bash container/build.sh
```

This creates `~/apphome/cuda.sif` from `mycontainer.def` (CUDA 12.8.1 + cuDNN on Ubuntu 24.04).

### 2. Install dependencies inside the container

Open a shell in the container and run the install script:

```bash
bash container/shell.sh
# Inside the container:
bash container/install-vllm.sh
```

This installs into `~/apphome/venv`:
- [uv](https://github.com/astral-sh/uv) package manager
- [vllm](https://github.com/vllm-project/vllm)
- [lighteval](https://github.com/huggingface/lighteval) (from source, with `multilingual` and `vllm` extras)

After installation, log in inside the container:
```bash
wandb login
hf auth login
```

## Running Evaluations

### Locally (with GPU)

Run from the repository root:

```bash
bash container/run-eval.sh <model-name> [additional-args]

# Example:
bash container/run-eval.sh Qwen/Qwen3-0.6B --tensor-parallel-size 8
```

The script:
- Activates the virtual environment in `~/apphome/venv`
- Runs `eval_vllm.py` with W&B logging (offline mode)
- Passes any additional arguments through to the evaluation script

### Via SLURM

See [`slurm/README.md`](slurm/README.md) for how to submit evaluation jobs on the Perun and Devana clusters (edit the script's model/arguments, then `sbatch`).

## Interactive Shells

```bash
# Open an interactive shell in the container (local machine):
bash container/shell.sh
```

The container shell has GPU access and the virtual environment on `PATH`.

## Configuration

The container home directory defaults to `~/apphome`. Override it by setting the `APPHOME` environment variable:

```bash
APPHOME=/scratch/myuser/apphome bash container/run-eval.sh Qwen/Qwen3-0.6B
```

## Script Reference

### [`build.sh`](build.sh)
Builds the container image from [`mycontainer.def`](mycontainer.def) into `~/apphome/cuda.sif`. Uses Docker bootstrap with NVIDIA CUDA 12.8.1 + cuDNN on Ubuntu 24.04.

### [`shell.sh`](shell.sh)
Opens an interactive container shell with GPU access and the virtual environment on `PATH`. Must be run from the **repository root**.

### [`install-vllm.sh`](install-vllm.sh)
Installs Python dependencies into the container's virtual environment (`$VIRTUAL_ENV`). Must be run **inside** the container (e.g., via `shell.sh`). Uses `uv sync` to install from `pyproject.toml`.

### [`run-eval.sh`](run-eval.sh)
Runs evaluation inside the container with GPU access. Usage:
```bash
bash container/run-eval.sh <model-name> [additional-args]
```
Must be run from the **repository root**. Binds the repository to `/home/sk-bench` inside the container.

### [`eval_entry.sh`](eval_entry.sh)
Entry point executed **inside** the container. Activates the virtual environment and runs `eval_vllm.py` with W&B logging enabled (offline mode). Not called directly — invoked by `run-eval.sh`.

### [`rerun.sh`](rerun.sh)
Re-runs evaluation from previously saved detail files. Called directly or from a SLURM script. Uses `scripts/rerun_from_details` inside the container.

### SLURM Scripts

| Script | Cluster | GPUs | Time | Notes |
|---|---|---|---|---|
| [`slurm/submit-perun.slurm`](slurm/submit-perun.slurm) | Perun | 4 | 200h | GPU partition, perun account |
| [`slurm/submit-devana.slurm`](slurm/submit-devana.slurm) | Devana | 1 | 4h | Loads singularity module |
