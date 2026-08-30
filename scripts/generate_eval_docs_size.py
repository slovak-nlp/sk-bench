#!/usr/bin/env python3
"""
Script to generate eval_docs_size.json from LightEval benchmark configurations.

Extracts eval_docs_size for each task and writes it as JSON with task_name as key.
"""

import sys
from pathlib import Path


# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def write_json_eval_docs_size(datasets_info: list[dict], output_file: str):
    """Write JSON file with task_name as key and eval_docs_size as value.

    This format is suitable for averaging results across tasks.
    """
    import json

    result = {}
    for info in datasets_info:
        task_name = info["task_name"]
        eval_docs_size = info["eval_docs_size"]
        result[task_name] = eval_docs_size

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote eval_docs_size mapping to {output_file}")
    return result


def main():
    """Main function to collect and export eval_docs_size."""
    from lighteval.tasks.lighteval_task import LightevalTask
    from src.tasks.task_registry import TASK_REGISTRY

    print("Collecting dataset information from registered tasks...")

    # Collect all tasks from registry
    all_tasks = list(TASK_REGISTRY.registry.values())

    datasets_info = []

    for task_config in all_tasks:
        lighteval_task = LightevalTask(task_config)

        repo_id = getattr(task_config, "hf_repo", None)
        if not repo_id:
            continue

        dataset_info = {
            "task_name": task_config.name,
            "eval_docs_size": len(lighteval_task.eval_docs()),
        }
        datasets_info.append(dataset_info)

    # Write JSON file
    write_json_eval_docs_size(datasets_info, "eval_docs_size.json")

    print(f"Found {len(datasets_info)} task configurations")


if __name__ == "__main__":
    main()
