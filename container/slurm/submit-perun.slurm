#!/bin/bash
#SBATCH --job-name="run-eval-sk"
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=GPU
#SBATCH --account=<account>
#SBATCH --qos=<account>
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=200:00:00
#SBATCH --ntasks=1

echo "Submit from the repository root, not from the container directory."
# Run the evaluation script with the specified model and parameters
srun --cpu-bind=none bash container/run-eval.sh MiniMaxAI/MiniMax-M2.7 --tensor-parallel-size 4 --max-tokens 8192 --max-batched-tokens 16384 --thinking-mode --tasks-file tasks_list_zero.txt --trust-remote-code
