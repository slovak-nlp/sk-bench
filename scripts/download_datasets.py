#!/usr/bin/env python3
"""
Script to download all HuggingFace datasets defined in the tasks folder.

Imports the actual lighteval task configurations and extracts dataset information
from them, ensuring we download exactly what's being used.
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset


@dataclass
class DatasetInfo:
    """Information about a dataset."""

    repo_id: str
    subset: Optional[str] = None
    revision: Optional[str] = None
    splits: tuple = ()
    source_file: str = ""


def extract_datasets_from_module(module) -> list[DatasetInfo]:
    """Extract dataset info from a task module's TASKS_TABLE."""
    datasets: list[DatasetInfo] = []

    if not hasattr(module, "TASKS_TABLE"):
        return datasets

    for task in module.TASKS_TABLE:
        # LightevalTaskConfig has hf_repo, hf_subset, hf_revision attributes
        repo_id = getattr(task, "hf_repo", None)
        if not repo_id:
            continue

        subset = getattr(task, "hf_subset", None)
        revision = getattr(task, "hf_revision", None)
        splits = getattr(task, "hf_avail_splits", ())

        datasets.append(
            DatasetInfo(
                repo_id=repo_id,
                subset=subset,
                revision=revision,
                splits=tuple(splits) if splits else (),
                source_file=module.__name__,
            )
        )

    return datasets


def collect_all_datasets() -> list[DatasetInfo]:
    """Collect all datasets from registered tasks."""
    from src.tasks.task_registry import TASK_REGISTRY

    datasets = []
    seen = set()

    # Get all registered tasks from the registry (registry is a dict)
    all_tasks = list(TASK_REGISTRY.registry.values())

    for task in all_tasks:
        repo_id = getattr(task, "hf_repo", None)
        if not repo_id:
            continue

        subset = getattr(task, "hf_subset", None)
        revision = getattr(task, "hf_revision", None)
        splits = getattr(task, "hf_avail_splits", ())

        key = (repo_id, subset)
        if key not in seen:
            seen.add(key)
            datasets.append(
                DatasetInfo(
                    repo_id=repo_id,
                    subset=subset,
                    revision=revision,
                    splits=tuple(splits) if splits else (),
                    source_file="registered",
                )
            )

    return datasets


def download_dataset(dataset_info: DatasetInfo) -> tuple[str, bool, str]:
    """Download a single dataset to HF cache. Returns (dataset_name, success, message)."""
    repo_id = dataset_info.repo_id
    subset = dataset_info.subset
    dataset_name = f"{repo_id}/{subset}" if subset else repo_id

    try:
        print(f"Downloading {dataset_name}...", file=sys.stderr)

        kwargs = {
            "trust_remote_code": True,
        }
        if subset:
            kwargs["name"] = subset  # type: ignore
        if dataset_info.revision:
            kwargs["revision"] = dataset_info.revision  # type: ignore

        # Just load the dataset - it will be cached automatically by HF
        ds = load_dataset(repo_id, **kwargs)
        # Force materialization by accessing the data
        _ = ds.column_names

        print(f"Successfully cached {dataset_name}", file=sys.stderr)
        return (dataset_name, True, "cached")

    except Exception as e:
        error_msg = f"Failed to download {dataset_name}: {str(e)}"
        print(error_msg, file=sys.stderr)
        return (dataset_name, False, error_msg)


def main():
    """Main function to download all datasets to HF cache."""
    print("Discovering datasets from registered tasks...", file=sys.stderr)

    # Collect all datasets from registered tasks
    all_datasets = collect_all_datasets()

    print(f"Found {len(all_datasets)} unique datasets to download", file=sys.stderr)
    print("\nDatasets to download:", file=sys.stderr)
    for ds in all_datasets:
        subset_str = f"/{ds.subset}" if ds.subset else ""
        splits_str = f" (splits: {', '.join(ds.splits)})" if ds.splits else ""
        print(f"  - {ds.repo_id}{subset_str}{splits_str}", file=sys.stderr)

    # Download datasets
    print("\nStarting downloads...", file=sys.stderr)
    results = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_dataset, ds): ds for ds in all_datasets}

        for future in as_completed(futures):
            results.append(future.result())

    # Summary
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful

    print(f"\n{'=' * 50}", file=sys.stderr)
    print(f"Download complete: {successful} successful, {failed} failed", file=sys.stderr)

    if failed > 0:
        print("\nFailed datasets:", file=sys.stderr)
        for name, success, msg in results:
            if not success:
                print(f"  - {name}: {msg}", file=sys.stderr)
        sys.exit(1)

    print("\nDatasets cached in HuggingFace default cache directory", file=sys.stderr)


if __name__ == "__main__":
    main()
