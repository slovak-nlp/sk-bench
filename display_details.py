import argparse
import glob
import json
import os


def truncate(text: str, max_len: int) -> str:
    """Return text truncated to max_len characters with an ellipsis if needed."""
    return text[:max_len] + ("..." if len(text) > max_len else "")


def as_list(value) -> list:
    """Normalize a value that may be None, a string, or a list into a list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def find_model_path(base: str, model_name: str) -> str | None:
    """Walk base to find the directory matching model_name.

    Prefers an exact relative-path match (e.g. ``org/model``) over a
    basename-only match so that models with the same name under different
    organisations are not confused.
    """
    basename_match = None
    for root, _dirs, _files in os.walk(base):
        rel = os.path.relpath(root, base)
        if rel == model_name:
            return root
        if basename_match is None and os.path.basename(root) == model_name:
            basename_match = root
    return basename_match


def fmt_metric(value) -> str:
    """Format a metric value: 4 decimal places for numbers, str() otherwise."""
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return f"{value:.4f}"
    return str(value)


def show_details(output_dir: str, model_name: str, task: str | None, timestamp: str = "latest") -> None:  # noqa
    base = os.path.join(output_dir, "details")

    if not os.path.isdir(base):
        print(f"Results directory not found: {base}")
        return

    model_path = find_model_path(base, model_name)
    if model_path is None:
        print(f"Model '{model_name}' not found under {base}")
        return

    if timestamp == "latest":
        timestamps = sorted(d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d)))
        if not timestamps:
            print("No timestamps found for this model.")
            return
        timestamp = timestamps[-1]

    ts_path = os.path.join(model_path, timestamp)
    if not os.path.isdir(ts_path):
        available = sorted(d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d)))
        print(f"Timestamp '{timestamp}' not found. Available: {available}")
        return

    if task:
        parquet_files = glob.glob(os.path.join(ts_path, f"details_{task}_{timestamp}.parquet"))
        if not parquet_files:
            # Fallback: match by derived task name after stripping prefix/suffix
            all_parquets = glob.glob(os.path.join(ts_path, "details_*.parquet"))
            parquet_files = [
                p
                for p in all_parquets
                if os.path.basename(p).removesuffix(".parquet").removeprefix("details_").removesuffix(f"_{timestamp}")
                == task
            ]
    else:
        parquet_files = glob.glob(os.path.join(ts_path, "details_*.parquet"))

    if not parquet_files:
        print(f"No parquet files found for model={model_name} task={task} timestamp={timestamp}")
        print(f"Searched in: {ts_path}")
        return

    from datasets import load_dataset

    for parquet_path in sorted(parquet_files):
        details = load_dataset("parquet", data_files=parquet_path, split="train")

        derived_task = (
            os.path.basename(parquet_path)
            .removesuffix(".parquet")
            .removeprefix("details_")
            .removesuffix(f"_{timestamp}")
        )

        print(f"\n{'=' * 80}")
        print(f"Model:     {model_name}")
        print(f"Task:      {derived_task}")
        print(f"Timestamp: {timestamp}")
        print(f"Records:   {len(details)}")
        print(f"{'=' * 80}")

        for i, detail in enumerate(details):
            doc = detail.get("doc") or {}
            metric = detail.get("metric") or {}
            model_response = detail.get("model_response") or {}

            print(f"\n--- Record {i} (id={doc.get('id', '?')}) ---")

            query = doc.get("query") or doc.get("original_query") or ""
            if query:
                print(f"Query:\n  {truncate(query, 500)}")

            instruction = doc.get("instruction") or ""
            if instruction:
                print(f"Instruction:\n  {truncate(instruction, 300)}")

            texts = as_list(model_response.get("text_post_processed") or model_response.get("text"))
            for j, t in enumerate(texts):
                label = f"Response[{j}]" if len(texts) > 1 else "Response"
                print(f"{label}:\n  {truncate(t, 800)}")

            reasonings = as_list(model_response.get("reasonings"))
            for j, r in enumerate(reasonings):
                if r:
                    label = f"Reasoning[{j}]" if len(reasonings) > 1 else "Reasoning"
                    print(f"{label}:\n  {truncate(r, 400)}")

            if metric:
                print("Metrics:")
                for k, v in metric.items():
                    print(f"  {k}: {v}")

            truncated_count = model_response.get("truncated_tokens_count")
            padded_count = model_response.get("padded_tokens_count")
            if truncated_count is not None or padded_count is not None:
                print(f"Tokens — truncated: {truncated_count}, padded: {padded_count}")


def find_results_json(output_dir: str, model_name: str, timestamp: str = "latest") -> str | None:
    """Return path to the results JSON file for the given model."""
    base = os.path.join(output_dir, "results")
    model_path = find_model_path(base, model_name)
    if model_path is None:
        print(f"Model '{model_name}' not found under {base}")
        return None

    json_files = sorted(glob.glob(os.path.join(model_path, "results_*.json")))
    if not json_files:
        print(f"No results JSON files found for model '{model_name}' in {model_path}")
        return None

    if timestamp == "latest":
        return json_files[-1]

    for f in json_files:
        if timestamp in os.path.basename(f):
            return f

    print(f"Timestamp '{timestamp}' not found; available: {[os.path.basename(f) for f in json_files]}")
    return None


def show_metrics(output_dir: str, model_name: str, timestamp: str = "latest") -> None:
    json_path = find_results_json(output_dir, model_name, timestamp)
    if json_path is None:
        return

    with open(json_path) as f:
        data = json.load(f)

    results = data.get("results", {})
    if not results:
        print("No results found in the file.")
        return

    model_label = data.get("config_general", {}).get("model_name", model_name)
    print(f"\nModel: {model_label}")
    print(f"File:  {os.path.basename(json_path)}\n")

    # Collect unique metric names (exclude *_stderr keys), preserving first-seen order
    all_metrics = list(
        dict.fromkeys(key for task_metrics in results.values() for key in task_metrics if not key.endswith("_stderr"))
    )

    tasks = sorted(results.keys())

    rows = [
        [task] + [fmt_metric(results[task][m]) if m in results[task] else "-" for m in all_metrics] for task in tasks
    ]

    headers = ["Task"] + all_metrics
    col_widths = [max(len(headers[i]), max((len(row[i]) for row in rows), default=0)) for i in range(len(headers))]

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells))

    print(fmt_row(headers))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt_row(row))


def list_available_results(output_dir: str = "results") -> None:
    base_path = os.path.join(output_dir, "results")

    if not os.path.exists(base_path):
        print("No results directory found.")
        return

    for root, _dirs, _files in os.walk(base_path):
        json_files = glob.glob(os.path.join(root, "results_*.json"))
        if not json_files:
            continue

        model_name = os.path.relpath(root, base_path)

        for json_file in sorted(json_files):
            timestamp = os.path.basename(json_file).removeprefix("results_").removesuffix(".json")
            with open(json_file) as f:
                data = json.load(f)
            tasks = sorted(t for t in data.get("results", {}) if t != "all")
            print(f"Model: {model_name}  |  Timestamp: {timestamp}  |  Tasks: {len(tasks)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display lighteval benchmark result details.")
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", help="List available results")
    list_parser.add_argument("--output-dir", default="results", help="Results directory (default: results)")

    show_parser = subparsers.add_parser("show", help="Show details for a model/task")
    show_parser.add_argument("model", help="Model name (e.g. Qwen/Qwen3-0.6B)")
    show_parser.add_argument(
        "task", nargs="?", default=None, help="Task name (e.g. ifeval_sk.ifeval|0); omit to show all tasks"
    )
    show_parser.add_argument("--timestamp", default="latest", help="Timestamp (default: latest)")
    show_parser.add_argument("--output-dir", default="results", help="Results directory (default: results)")

    metrics_parser = subparsers.add_parser("metrics", help="Print a metrics table for a model")
    metrics_parser.add_argument("model", help="Model name (e.g. Qwen/Qwen3-0.6B)")
    metrics_parser.add_argument("--timestamp", default="latest", help="Timestamp (default: latest)")
    metrics_parser.add_argument("--output-dir", default="results", help="Results directory (default: results)")

    args = parser.parse_args()

    if args.command == "list":
        list_available_results(args.output_dir)
    elif args.command == "show":
        show_details(args.output_dir, args.model, args.task, args.timestamp)
    elif args.command == "metrics":
        show_metrics(args.output_dir, args.model, args.timestamp)
    else:
        parser.print_help()
