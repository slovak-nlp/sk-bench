
# SLURM Jobs

SLURM submission scripts for running evaluations on HPC clusters. Each script wraps `container/run-eval.sh` and forwards model + arguments to it.

**Always submit from the repository root**, not from this directory:

```bash
sbatch container/slurm/submit-<cluster>.slurm
```

## Clusters

| Script | Cluster | GPUs | Time | Notes |
|---|---|---|---|---|
| [`submit-perun.slurm`](submit-perun.slurm) | [Perun](https://hpc.tuke.sk/en) | 4 | up to 200h | GPU partition; requires active account/QOS |
| [`submit-devana.slurm`](submit-devana.slurm) | [Devana](https://hpc.sav.sk/en/supercomputers/devana/) | 1 | 4h | GPU partition; loads `singularity` module |

## Prerequisites

- SSH access to the cluster and an active account (plus a QOS on Perun).
- Repository cloned to a shared/scratch location.
- Container image already built (`container/build.sh`) and dependencies installed (`container/shell.sh` then `container/install-vllm.sh`) — see [`container/README.md`](../README.md).

## Usage

Edit the script's `#SBATCH` directives (account, partition, time, resources) and its final `srun` line to set the model and evaluation arguments, then submit:

## Changing the model / running other models

The model and its evaluation arguments are defined on the script's last `srun` line, in the form:

```bash
srun bash container/run-eval.sh <MODEL> [--eval-args ...]
```

To evaluate a different model, replace `<MODEL>` with the new identifier (and the arguments that follow with whatever the model needs). Any argument accepted by `eval_vllm.py` can be appended here.

For example, swapping the Perun script's default to another open-source model with tensor parallelism for its 4 GPUs:

```bash
srun --cpu-bind=none bash container/run-eval.sh meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 --max-tokens 8192 --max-batched-tokens 16384 \
    --tasks-file tasks_list_zero.txt
```

For multi-GPU tokenizers that require remote code, keep `--trust-remote-code` (as in the default Perun `srun` line). For reasoning/thinking models, keep `--thinking-mode`.

To make the model easy to change without editing the `srun` line, define it once as a shell variable near the top of the script (consult your cluster's SLURM docs) and reference it on the `srun` line, e.g.:

```bash
MODEL=Qwen/Qwen3-0.6B
...
srun bash container/run-eval.sh "$MODEL" --tensor-parallel-size 1
```

### Perun (`submit-perun.slurm`)

### Perun (`submit-perun.slurm`)

High-GPU nodes, long runtimes — suitable for full-benchmark evaluations on large models.

```bash
sbatch container/slurm/submit-perun.slurm
```

The default job runs `MiniMaxAI/MiniMax-M2.7` on 4 GPUs over `tasks_list_zero.txt`:

```bash
srun --cpu-bind=none bash container/run-eval.sh MiniMaxAI/MiniMax-M2.7 \
    --tensor-parallel-size 4 --max-tokens 8192 --max-batched-tokens 16384 \
    --thinking-mode --tasks-file tasks_list_zero.txt --trust-remote-code
```

### Devana (`submit-devana.slurm`)

Single GPU, shorter time limit — good for smoke tests and smaller models.

```bash
sbatch container/slurm/submit-devana.slurm
```

The default job runs `google/gemma-3-1b-it` on 1 GPU:

```bash
srun bash container/run-eval.sh google/gemma-3-1b-it --tensor-parallel-size 1
```

Logs are written to `logs/%x_%j.out` / `logs/%x_%j.err` (set via `#SBATCH --output` / `--error`).
