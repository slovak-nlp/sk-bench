#!/usr/bin/env python3
"""Recompute benchmark JSON results from saved lighteval details parquet files.

This script walks a details directory shaped like:

    <output_dir>/details/<model_name>/<timestamp>/details_<task>_<timestamp>.parquet

It can also scan a broader root that contains one or more nested `details/`
trees and still recover the model name from the path.

For each timestamp folder it reconstructs the task list from the parquet file
names, loads the saved model responses from details, recomputes the metrics, and
writes the standard results JSON under:

    <output_dir>/results/<model_name>/results_<timestamp>.json

The model itself is not loaded.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from lighteval.models.model_input import GenerationParameters
from lighteval.models.vllm.vllm_model import VLLMModelConfig

from src.prompt_manager_override import patch_prompt_manager_chat_template
from src.tasks.task_registry import TASK_REGISTRY
from src.pipeline_patch import Pipeline, ParallelismManager, PipelineParameters
from src.evaluation_tracker_patch import EvaluationTracker
import os
import torch
torch.cuda.is_available = lambda: False


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate saved lighteval parquet details without loading a model."
    )
    parser.add_argument(
        "details_dir",
        type=str,
        help="Root directory containing the saved details/ tree.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where the recomputed results JSON files will be written. Defaults to the parent of details_dir.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on the number of evaluation samples per task.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Model dtype stored in the metadata JSON. Defaults to bfloat16.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size stored in the metadata JSON. Defaults to 1.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization stored in the metadata JSON. Defaults to 0.9.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Store trust_remote_code=True in the metadata JSON.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional max_new_tokens value stored in the metadata JSON.",
    )
    parser.add_argument(
        "--max-batched-tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens batched together for generation metadata.",
    )
    parser.add_argument(
        "--save-details",
        action="store_true",
        help="Also write the details parquet files again. Off by default.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        help="Optional reasoning effort for thinking models (e.g., none, low, medium, high).",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=1000,
        help="Number of bootstrap iterations for stderr estimates. Use 0 to skip stderr.",
    )
    parser.add_argument(
        "--cot-prompt",
        type=str,
        default=None,
        help="Optional chain-of-thought prompt file.",
    )
    reasoning_tags_group = parser.add_mutually_exclusive_group()
    reasoning_tags_group.add_argument(
        "--remove-reasoning-tags",
        action="store_true",
        default=True,
        help="Remove reasoning tags from responses (default).",
    )
    reasoning_tags_group.add_argument(
        "--no-remove-reasoning-tags",
        action="store_false",
        dest="remove_reasoning_tags",
        help="Keep reasoning tags in responses.",
    )
    parser.add_argument(
        "--reasoning-tags",
        type=str,
        default="[('<think>', '</think>')]",
        help="Reasoning tags to strip from responses.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of parallel worker processes to use. "
            "Defaults to SLURM_CPUS_ON_NODE or 1 if not set."
        ),
    )
    return parser.parse_args()


def parse_detail_filename(parquet_path: Path) -> tuple[str, str] | None:
    """Return (task_name, date_id) parsed from a details parquet filename."""
    stem = parquet_path.stem
    if not stem.startswith("details_"):
        return None

    body = stem.removeprefix("details_")
    if "_" not in body:
        return None

    task_name, date_id = body.rsplit("_", 1)
    return task_name, date_id


def discover_runs(details_root: Path, model_name_hint: str | None = None) -> dict[tuple[str, str], set[str]]:
    """Group details parquet files by (model_name, timestamp) and collect task names."""
    runs: dict[tuple[str, str], set[str]] = defaultdict(set)

    for parquet_path in details_root.rglob("details_*.parquet"):
        parsed = parse_detail_filename(parquet_path)
        if parsed is None:
            continue

        task_name, date_id = parsed
        run_dir = parquet_path.parent
        if run_dir.name != date_id:
            logger.warning("Skipping %s because the filename timestamp does not match its folder", parquet_path)
            continue

        try:
            model_rel_path = run_dir.parent.relative_to(details_root)
        except ValueError:
            logger.warning("Skipping %s because it is not under %s", parquet_path, details_root)
            continue

        if model_name_hint is not None:
            model_name = model_name_hint
        else:
            # Allow callers to point the script at a broader root that contains a
            # nested `details/` directory and preserve any extra path segments
            # between that anchor and the timestamp folder.
            if model_rel_path.parts and model_rel_path.parts[0] == "details":
                model_rel_path = Path(*model_rel_path.parts[1:])

            model_name = "/".join(model_rel_path.parts)
        if not model_name:
            logger.warning("Skipping %s because the model name could not be derived", parquet_path)
            continue

        runs[(model_name, date_id)].add(task_name)

    return runs


def build_model_config(
    model_name: str,
    dtype: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    trust_remote_code: bool,
    max_tokens: int | None,
    max_batched_tokens: int,
) -> VLLMModelConfig:
    generation_parameters = GenerationParameters(max_new_tokens=max_tokens) if max_tokens is not None else GenerationParameters()

    return VLLMModelConfig(
        model_name=model_name,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=trust_remote_code,
        generation_parameters=generation_parameters,
        max_num_batched_tokens=max_batched_tokens,
    )


def re_evaluate_run(
    output_dir: Path,
    model_name: str,
    date_id: str,
    task_names: set[str],
    args: argparse.Namespace,
) -> None:
    # Map parquet task names (which often include a shot suffix like '|0')
    # to registered task base names (registry stores names without the '|<shot>' suffix).
    mapped_base_tasks: dict[str, str] = {}
    missing_tasks: list[str] = []

    for t in sorted(task_names):
        base = t.split("|", 1)[0]
        if TASK_REGISTRY.get(base) is not None:
            mapped_base_tasks[t] = t
        else:
            missing_tasks.append(t)

    if missing_tasks:
        logger.warning("Skipping %d unknown tasks for %s/%s: %s", len(missing_tasks), model_name, date_id, missing_tasks)

    valid_tasks = sorted(set(mapped_base_tasks.values()))
    if not valid_tasks:
        logger.warning("No valid tasks found for %s/%s, skipping", model_name, date_id)
        return

    tasks = ",".join(valid_tasks)
    logger.info("Re-evaluating %s/%s with %d tasks", model_name, date_id, len(valid_tasks))

    evaluation_tracker = EvaluationTracker(
        output_dir=str(output_dir),
        save_details=args.save_details,
        use_wandb=False,
    )

    pipeline_parameters = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        custom_tasks_directory="./src/tasks/__init__.py",
        max_samples=args.max_samples,
        cot_prompt=args.cot_prompt,
        remove_reasoning_tags=args.remove_reasoning_tags,
        reasoning_tags=args.reasoning_tags,
        load_responses_from_details_date_id=date_id,
        bootstrap_iters=args.bootstrap_iters,
    )

    model_config = build_model_config(
        model_name=model_name,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        max_tokens=args.max_tokens,
        max_batched_tokens=args.max_batched_tokens,
    )

    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_parameters,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()


def _worker_run(output_dir_str: str, model_name: str, date_id: str, task_names: set[str], args: argparse.Namespace):
    """Module-level worker function suitable for multiprocessing."""
    out_dir = Path(output_dir_str)
    # safely remove .litell_cache to avoid multiprocessing issues with concurrent access
    cache_dir = ".litell_cache"
    if os.path.exists(cache_dir):
        try:
            if os.path.isdir(cache_dir):
                import shutil

                shutil.rmtree(cache_dir)
            else:
                os.remove(cache_dir)
        except Exception as e:
            logger.warning("Failed to remove cache directory %s: %s", cache_dir, e)
    re_evaluate_run(output_dir=out_dir, model_name=model_name, date_id=date_id, task_names=task_names, args=args)
    return model_name, date_id


def main() -> None:
    args = parse_arguments()
    if args.reasoning_effort is not None:
        os.environ["REASONING_EFFORT"] = args.reasoning_effort
    patch_prompt_manager_chat_template()

    # export HF_DATASETS_OFFLINE=1
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    details_root = Path(args.details_dir).resolve()
    if not details_root.is_dir():
        raise FileNotFoundError(f"Details directory not found: {details_root}")

    model_name_hint = None
    output_root = details_root
    if details_root.parent.parent.name == "details":
        model_name_hint = "/".join(details_root.relative_to(details_root.parent.parent).parts)
        output_root = details_root.parent.parent.parent
    elif details_root.name == "details":
        output_root = details_root.parent

    output_dir = Path(args.output_dir).resolve() if args.output_dir else output_root
    logger.info("Scanning details under %s", details_root)
    logger.info("Writing results to %s", output_dir)

    runs = discover_runs(details_root, model_name_hint=model_name_hint)
    if not runs:
        raise FileNotFoundError(f"No details parquet files found under {details_root}")

    # Default ignored models (preserve the original hard-coded list) and merge CLI ignores
    ignore_set: set[str] = {
    }

    if ignore_set:
        logger.info("Ignoring %d model(s): %s", len(ignore_set), sorted(ignore_set))

    # Build job list
    jobs = [((model_name, date_id), task_names) for (model_name, date_id), task_names in sorted(runs.items()) if model_name not in ignore_set]
    if not jobs:
        logger.info("No runs to process after applying ignore list.")
        return

    # Determine number of workers
    if args.workers and args.workers > 0:
        max_workers = args.workers
    else:
        max_workers = int(os.environ.get("SLURM_CPUS_ON_NODE", os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))

    max_workers = max(1, min(len(jobs), max_workers))

    # Use multiprocessing spawn to avoid fork-related issues
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    if max_workers > 1:
        logger.info("Parallelizing %d runs with %d worker(s)", len(jobs), max_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            futures = {}
            for (model_name, date_id), task_names in jobs:
                futures[exe.submit(_worker_run, str(output_dir), model_name, date_id, task_names, args)] = (model_name, date_id)

            for fut in as_completed(futures):
                model_name, date_id = futures[fut]
                try:
                    res = fut.result()
                    logger.info("Finished %s/%s", res[0], res[1])
                except Exception:
                    logger.exception("Run failed for %s/%s", model_name, date_id)
    else:
        # Sequential fallback
        for (model_name, date_id), task_names in jobs:
            re_evaluate_run(output_dir=output_dir, model_name=model_name, date_id=date_id, task_names=task_names, args=args)


if __name__ == "__main__":
    main()
