#!/bin/bash
#SBATCH --account=<account>
#SBATCH --job-name="run-eval-sk"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mem=48G

# Activate scratch (ONE LINE!)
# source .activate_scratch

module load singularity

echo "Submit from the repository root, not from the container directory."
# Run the evaluation script with the specified model and parameters
srun bash container/run-eval.sh google/gemma-3-1b-it --tensor-parallel-size 1
