#!/usr/bin/env python3
"""
Merge lighteval result JSONs per model and compute aggregated scores.

For each model (identified via config_general.model_name):
  1. Merges all JSON files found in INPUT_DIR
  2. Averages metrics across few-shot seeds for every subset
  3. Filters out unwanted metrics and applies name normalisation
  4. Computes a weighted average per task family  (<family>_average)
  5. Computes an unweighted average across all task families  (all)
  6. Writes one output JSON per model to OUTPUT_DIR

Usage:
    python aggregate_results.py [INPUT_DIR [OUTPUT_DIR]]

Defaults:  INPUT_DIR=./results   OUTPUT_DIR=./aggregated
"""

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional


# ── configuration ──────────────────────────────────────────────────────────────

# Metrics to drop entirely (and their _stderr companions)
METRICS_TO_FILTER = {"em", "em_norm"}

# Raw metric name → normalised name used in output
# Metrics absent from this dict keep their original name.
METRIC_NORMALIZATION = {
    "em_conservative_norm": "acc",
    "em_conservative": "acc",
    "em_relaxed_multi_label": "acc",
    "oneruler_accuracy": "acc",
    "exact_match_slk_prefix": "qem",
    "f1_slk": "qf1",
}

# Datasets where em_norm_relaxed should be normalized to acc
ANALOGIES_DATASETS = {"analogies_sk"}

# Sample counts per subset key (task.prompt:subtask, no |fewshot suffix).
# Loaded from eval_docs_size.json at module load time.
def _load_subset_sizes() -> dict[str, int]:
    """Load subset sizes from eval_docs_size.json in the project root."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    json_file = project_root / "eval_docs_size.json"

    if json_file.exists():
        with open(json_file) as f:
            return json.load(f)
    return {}


SUBSET_SIZES = _load_subset_sizes()

# ── helpers ────────────────────────────────────────────────────────────────────

def normalize_metric(name: str, dataset: str | None = None) -> Optional[str]:
    """
    Return the normalised metric name, or None if the metric should be dropped.
    Handles both plain metric names and their *_stderr companions.

    Args:
        name: The metric name to normalize (may include _stderr suffix)
        dataset: Optional dataset name for dataset-specific normalization rules
    """
    is_stderr = name.endswith("_stderr")
    base = name[: -len("_stderr")] if is_stderr else name

    if base in METRICS_TO_FILTER:
        return None

    normalised_base = METRIC_NORMALIZATION.get(base, base)

    # Special case: for analogies datasets, normalize em_norm_relaxed to acc
    if dataset and dataset in ANALOGIES_DATASETS and base == "em_norm_relaxed":
        normalised_base = "acc"

    return f"{normalised_base}_stderr" if is_stderr else normalised_base


def get_task_group(subset_key: str) -> Optional[str]:  # noqa: E701, C901
    """
    Map a subset key to its benchmark task-group name.

    Rules are checked in order; the first match wins.
    Returns None for subsets that do not belong to any defined group
    (they are kept in the results but excluded from all averages).

    Key format:  <dataset>.<prompt_variant>:<subtask>
    Examples:
        'mmlu.prompt2_mqa_single_correct_en:abstract_algebra'
        'e_tests.prompt3_qa_en:biologia'
        'sklep.prompt3_nli_3_classes_en:nli'
        'oneruler_sk.oneruler_english:cwe_4096'
    """
    k = subset_key

    # ── single-dataset groups ───────────────────────────────────────────────
    if k.startswith("analogies_sk."):      return "analogies_sk"
    if k.startswith("arc."):               return "arc"
    if k.startswith("belebele."):          return "belebele"
    if k.startswith("chiby."):             return "chiby"
    if k.startswith("movie_review."):      return "movie_review"
    if k.startswith("driving_test."):      return "driving_test"
    if k.startswith("financial_exams."):   return "financial_exams"
    if k.startswith("ifeval_sk."):         return "ifeval_sk"
    if k.startswith("mmlu."):              return "mmlu"
    if k.startswith("globalmmlu-lite."):   return "globalmmlu_lite"
    if k.startswith("multi_wiki_qa."):     return "multi_wiki_qa"
    if k.startswith("nli_sk."):            return "nli_sk"
    if k.startswith("oneruler_sk."):       return "oneruler"      # merges english + slovak
    if k.startswith("scio."):              return "scio"
    if k.startswith("slohry."):            return "slohry"
    if k.startswith("slovak_trivia."):     return "slovak_trivia"
    if k.startswith("slovakcopa."):        return "slovakcopa"
    if k.startswith("slovaksum."):         return "slovaksum"
    if k.startswith("smesum."):            return "smesum"
    if k.startswith("toxic_sk."):          return "toxic_sk"
    if k.startswith("truthfulqa."):        return "truthful_qa"
    if k.startswith("skj1."):              return "skj1"

    # ── e_tests: split by prompt type ──────────────────────────────────────
    # prompt contains '_mqa_'  → e_tests_mqa  (single/multiple choice)
    # prompt contains '_qa_'   → e_tests_qa   (open-ended; '_mqa_' is not '_qa_')
    if k.startswith("e_tests."):
        if "_mqa_" in k:
            return "e_tests_mqa"
        if "_qa_" in k:
            return "e_tests_qa"

    # ── senti_sk: split by number of classes ───────────────────────────────
    if k.startswith("senti_sk."):
        if "_binary" in k:
            return "senti_sk_binary"
        if "_3_class" in k:
            return "senti_sk_3class"

    # ── sklep: split by subtask ─────────────────────────────────────────────
    if k.startswith("sklep."):
        if "sentiment" in k:
            return "sklep_sentiment_analysis"
        if "hate_speech" in k:
            return "sklep_hate_speech"
        if "nli_3_classes" in k:
            return "sklep_nli"
        if "nli_binary" in k:
            return "sklep_rte"
        if "_qa" in k:
            return "sklep_qa"

    # ── slovak_tests: split by prompt type ─────────────────────────────────
    if k.startswith("slovak_tests."):
        if "_mqa_" in k:
            return "slovak_tests_mqa"
        if "_qa_" in k:
            return "slovak_tests_qa"

    # ── unclassified ────────────────────────────────────────────────────────
    return None


# ── core processing ────────────────────────────────────────────────────────────


def collect_raw(files: list[Path]) -> tuple[dict, dict, dict]:
    """
    Read all JSON files for one model.

    For duplicate task|shot keys, uses metrics from the newest file (by mtime).

    Returns
    -------
    raw_results  : {subset_key|shot: {raw_metric: value}}
    config_general, versions  from the most-recently-written file
    """
    # Track which file each key came from (for conflict resolution)
    key_to_source: dict[str, Path] = {}
    raw_results: dict[str, dict] = {}
    config_general: dict = {}
    versions: dict = {}
    latest_mtime = 0.0

    # Sort files by mtime (oldest first) so newer files overwrite older ones
    files_by_mtime = sorted(files, key=lambda p: p.stat().st_mtime)

    for fpath in files_by_mtime:
        with open(fpath) as fh:
            data = json.load(fh)

        # Update config/versions from the latest file
        file_mtime = fpath.stat().st_mtime
        if file_mtime > latest_mtime:
            config_general = data.get("config_general", {})
            versions = data.get("versions", {})
            latest_mtime = file_mtime

        for key, metrics in data.get("results", {}).items():
            if "|" not in key:  # skip lighteval's own "all" entry
                continue
            base_key, shot = key.rsplit("|", 1)
            if base_key.endswith(":_average"):  # skip lighteval's per-prompt averages
                continue
            # Keep shot count as part of the key: subset_key|shot
            full_key = f"{base_key}|{shot}"

            # Only update if this is from a newer file
            if full_key not in key_to_source or fpath.stat().st_mtime > key_to_source[full_key].stat().st_mtime:
                key_to_source[full_key] = fpath
                raw_results[full_key] = metrics

    return raw_results, config_general, versions


def extract_dataset(subset_key: str) -> str:
    """Extract dataset name from subset key (e.g., 'analogies_sk.prompt1|0' → 'analogies_sk')."""
    base = subset_key.rsplit("|", 1)[0] if "|" in subset_key else subset_key
    if "." in base:
        return base.split(".")[0]
    return base


def build_subset_results(raw_results: dict) -> dict[str, dict]:
    """
    Filter metrics and apply normalisation per shot configuration.
    Returns {subset_key|shot: {norm_metric: value}}
    """
    subset_results: dict[str, dict] = {}

    for full_key, metrics in raw_results.items():
        # Collect every raw metric name
        all_raw_metrics: set[str] = set(metrics.keys())

        # Extract dataset name for dataset-specific normalization
        dataset = extract_dataset(full_key)

        # Normalise names, keeping first non-None value for collisions
        normalised: dict[str, float] = {}
        for raw_metric in sorted(all_raw_metrics):   # sorted → deterministic collision handling
            norm_name = normalize_metric(raw_metric, dataset)
            if norm_name is None:
                continue  # filtered out
            if norm_name in normalised:
                continue  # already filled by a higher-priority raw name

            normalised[norm_name] = metrics[raw_metric]

        if normalised:
            subset_results[full_key] = normalised

    return subset_results


def strip_shot(subset_key: str) -> str:
    """Remove shot suffix from key: 'task.prompt|3' → 'task.prompt'."""
    if "|" in subset_key:
        return subset_key.rsplit("|", 1)[0]
    return subset_key


def get_language_from_key(subset_key: str) -> Optional[str]:
    """
    Extract language from the prompt part of a subset key.

    Key format: {dataset}.{prompt_variant}:{subtask}|shot
    The language is at the end of the prompt_variant (before :subtask or |shot).

    Examples:
        'arc.prompt1_mqa_en|0' → 'EN'
        'toxic_sk.prompt1_xyz_sk|0' → 'SK'
        'mmlu.prompt2_mqa_en:abstract_algebra|0' → 'EN'
    """
    # Remove shot suffix first
    base = subset_key.rsplit("|", 1)[0] if "|" in subset_key else subset_key

    # Remove subtask part (after colon)
    if ":" in base:
        base = base.split(":")[0]

    # Now base is like: dataset.promptN_variant_lang
    # Extract dataset and prompt parts
    if "." in base:
        dataset_part, prompt_part = base.split(".", 1)
    else:
        dataset_part = ""
        prompt_part = base

    lp = prompt_part.lower()
    ld = dataset_part.lower()

    # Explicit markers in the prompt variant, handle also oneruler
    if lp.endswith("_sk") or lp.endswith("_slovak") or "_slovak" in lp or lp.endswith("slovak"):
        return "SK"
    if lp.endswith("_en") or lp.endswith("_english") or "_english" in lp or lp.endswith("english"):
        return "EN"

    # Special-case for ifeval
    if ld.startswith("ifeval"):
        return "SK"

    return None


def build_family_averages(subset_results: dict[str, dict]) -> dict[str, dict]:  # noqa: C901
    """
    For each task family and shot count, compute averages:
      1. Per-language averages across prompt variants (e.g., arc_SK_AVERAGE|0, arc_EN_AVERAGE|0)
      2. Combined family average (SK + EN equally weighted, e.g., arc_AVERAGE|0)

    Returns {key|shot: {metric: value}} including both per-language and combined averages.
    """
    # Group subsets by (task_group, language, shot)
    lang_shot_to_subsets: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for full_key in subset_results:
        group = get_task_group(full_key)
        if group is not None:
            lang = get_language_from_key(full_key)
            if lang is None:
                lang = "UNK"
            shot = full_key.rsplit("|", 1)[1] if "|" in full_key else "0"
            lang_shot_to_subsets[(group, lang, shot)].append(full_key)

    # Group by (family, shot) to combine language averages
    family_shot_langs: dict[tuple[str, str], list[tuple[str, list[str]]]] = defaultdict(list)
    for (family, lang, shot), subsets in lang_shot_to_subsets.items():
        family_shot_langs[(family, shot)].append((lang, subsets))

    family_averages: dict[str, dict] = {}

    for (family, shot), lang_subset_pairs in family_shot_langs.items():
        # Collect all primary metric names
        primary_metrics: set[str] = set()
        for _, subsets in lang_subset_pairs:
            for sk in subsets:
                primary_metrics.update(m for m in subset_results[sk] if not m.endswith("_stderr"))

        missing_size_warned: set[str] = set()

        # First pass: compute per-language averages for all metrics
        lang_all_values: dict[str, dict[str, float]] = defaultdict(dict)  # lang → {metric: avg}
        lang_all_stderr: dict[str, dict[str, float]] = defaultdict(dict)  # lang → {metric: stderr}

        for lang, subsets in lang_subset_pairs:
            for metric in primary_metrics:
                stderr_key = f"{metric}_stderr"
                total_w = 0.0
                weighted_sum = 0.0
                weighted_se_sum = 0.0
                has_stderr = False

                for full_key in subsets:
                    res = subset_results[full_key]
                    if metric not in res:
                        continue

                    base_key = strip_shot(full_key)
                    weight = SUBSET_SIZES.get(base_key, None)
                    if weight is None:
                        if base_key not in missing_size_warned:
                            print(f"  [WARN] no size for '{base_key}', using weight=1", file=sys.stderr)
                            missing_size_warned.add(base_key)
                        weight = 1

                    weighted_sum += res[metric] * weight
                    total_w += weight

                    if stderr_key in res:
                        has_stderr = True
                        weighted_se_sum += res[stderr_key] * weight

                if total_w > 0:
                    lang_all_values[lang][metric] = weighted_sum / total_w
                    if has_stderr:
                        lang_all_stderr[lang][stderr_key] = weighted_se_sum / total_w

            # Store per-language averages (e.g., arc_SK_AVERAGE|0, arc_EN_AVERAGE|0)
            lang_key = f"{family}_{lang.upper()}_AVERAGE|{shot}"
            family_averages[lang_key] = dict(lang_all_values[lang])
            if lang_all_stderr[lang]:
                family_averages[lang_key].update(lang_all_stderr[lang])

            # Per-prompt size-weighted averages, exposing the across-prompt
            # spread (std over the prompt variants) without altering the
            # pooled average above. Stored per metric as "<metric>__prompts":
            # {prompt_id: value} for the table generator to render as +/- sigma.
            prompt_subsets: dict[str, list[str]] = defaultdict(list)
            for full_key in subsets:
                pm = re.search(r"prompt(\d+)", full_key)
                prompt_subsets[pm.group(1) if pm else "0"].append(full_key)
            for metric in primary_metrics:
                pvals: dict[str, float] = {}
                for pid, pks in prompt_subsets.items():
                    tw = 0.0
                    ws = 0.0
                    for full_key in pks:
                        res = subset_results[full_key]
                        if metric not in res:
                            continue
                        w = SUBSET_SIZES.get(strip_shot(full_key), None) or 1
                        ws += res[metric] * w
                        tw += w
                    if tw > 0:
                        pvals[pid] = ws / tw
                if len(pvals) >= 2:
                    family_averages[lang_key][f"{metric}__prompts"] = pvals

        # Second pass: compute combined family average (SK + EN equally weighted)
        combined_key = f"{family}_AVERAGE|{shot}"
        family_averages[combined_key] = {}

        for metric in primary_metrics:
            stderr_key = f"{metric}_stderr"
            # Get values from each language that has them
            vals = [lang_all_values[lang][metric] for lang in lang_all_values if metric in lang_all_values[lang]]
            se_vals = [
                lang_all_stderr[lang][stderr_key] for lang in lang_all_stderr if stderr_key in lang_all_stderr[lang]
            ]

            if vals:
                family_averages[combined_key][metric] = sum(vals) / len(vals)
            if se_vals:
                # Pool standard errors using quadratic mean (for independent equal-weight samples)
                family_averages[combined_key][stderr_key] = math.sqrt(sum(s * s for s in se_vals)) / len(se_vals)

    return family_averages


def build_language_averages(family_averages: dict[str, dict]) -> dict[str, dict]:
    """
    Unweighted average of each primary metric across all task families per language,
    computed separately for Slovak and English, and for each shot count.

    Returns {lang_shot: {metric: value}} where lang_shot is "SK|0", "EN|3", etc.
    """
    # Group by (language, shot): {"sk|0": {metric: [vals]}, "en|3": {metric: [vals]}}
    lang_shot_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    lang_shot_stderr: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for fam_key, fam_res in family_averages.items():
        # Only process per-language family entries (e.g., "arc_SK_AVERAGE|0", "arc_EN_AVERAGE|0")
        lang_match = re.search(r"_(SK|EN|UNK)_AVERAGE\|(\d+)$", fam_key)
        if not lang_match:
            continue

        lang = lang_match.group(1)  # "SK", "EN", or "UNK"
        shot = lang_match.group(2)
        lang_shot = f"{lang}_AVERAGE|{shot}"

        for key, val in fam_res.items():
            if key.endswith("__prompts"):
                continue  # per-prompt spread data, not a scalar metric to pool
            if key.endswith("_stderr"):
                lang_shot_stderr[lang_shot][key].append(val)
            else:
                lang_shot_metrics[lang_shot][key].append(val)

    lang_averages: dict[str, dict[str, float]] = {}
    for lang_shot in lang_shot_metrics:
        lang_averages[lang_shot] = {}
        for metric, vals in lang_shot_metrics[lang_shot].items():
            lang_averages[lang_shot][metric] = sum(vals) / len(vals)
        for stderr_key, vals in lang_shot_stderr[lang_shot].items():
            # Pool standard errors using quadratic mean (for independent equal-weight samples)
            lang_averages[lang_shot][stderr_key] = math.sqrt(sum(s * s for s in vals)) / len(vals)

    return lang_averages


def build_overall(family_averages: dict[str, dict]) -> dict[str, dict]:
    """
    Unweighted average across all task-family averages.

    Uses the combined family averages (e.g., arc_AVERAGE|0, mmlu_AVERAGE|0) which already
    combine SK+EN equally for bilingual tasks, and single-language averages for monolingual tasks.

    Returns {key: {metric: value}} where key is "AVERAGE|0", "AVERAGE|3", etc.
    """
    # Group by shot count: {"0": {metric: [vals]}, "3": {metric: [vals]}}
    shot_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    shot_stderr: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for fam_key, fam_res in family_averages.items():
        # Only process combined family averages (e.g., "arc_AVERAGE|0", "mmlu_AVERAGE|0")
        # Skip per-language entries (e.g., "arc_SK_AVERAGE|0", "arc_EN_AVERAGE|0")
        if not re.match(r"^[a-z_]+_AVERAGE\|\d+$", fam_key):
            continue

        # Extract shot from key: "arc_AVERAGE|3" → "3"
        shot = fam_key.rsplit("|", 1)[1]

        for key, val in fam_res.items():
            if key.endswith("_stderr"):
                shot_stderr[shot][key].append(val)
            else:
                shot_metrics[shot][key].append(val)

    overall: dict[str, dict[str, float]] = {}
    for shot in shot_metrics:
        overall_key = f"AVERAGE|{shot}"
        overall[overall_key] = {}
        for metric, vals in shot_metrics[shot].items():
            overall[overall_key][metric] = sum(vals) / len(vals)
        for stderr_key, vals in shot_stderr[shot].items():
            # Pool standard errors using quadratic mean (for independent equal-weight samples)
            overall[overall_key][stderr_key] = math.sqrt(sum(s * s for s in vals)) / len(vals)

    return overall


def process_model(files: list[Path]) -> dict:
    """Full pipeline for one model. Returns the merged output dict."""
    raw_results, config_general, versions = collect_raw(files)
    model_name = config_general.get("model_name", "unknown")
    print(f"  Processing {model_name} from {len(files)} file(s) …", file=sys.stderr)

    subset_results = build_subset_results(raw_results)
    family_averages = build_family_averages(subset_results)
    lang_averages = build_language_averages(family_averages)
    overall_by_shot = build_overall(family_averages)

    # Note: lang_averages is kept for potential future use or debugging

    # Build final results dict: subsets → family averages → language averages → overall averages
    results: dict[str, dict] = {}
    results.update(subset_results)
    results.update(family_averages)

    # Add language-level averages per shot: "SK_AVERAGE|0", "EN_AVERAGE|3", etc.
    results.update(lang_averages)

    # Add overall average entries per shot: "AVERAGE|0", "AVERAGE|3", etc.
    results.update(overall_by_shot)

    return {
        "model_name": model_name,
        "config_general": config_general,
        "versions": versions,
        "results": results,
    }


# ── main ───────────────────────────────────────────────────────────────────────


def safe_model_filename(model_name: str) -> str:
    """Turn 'google/gemma-4-31B-it' into 'google__gemma-4-31B-it'."""
    return model_name.replace("/", "__").replace("\\", "__")


def main(input_dir: str = "./results", output_dir: str = "./aggregated") -> None:
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_files = sorted(in_path.rglob("*.json"))
    if not json_files:
        print(f"No JSON files found in {in_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(json_files)} JSON file(s) in {in_path}", file=sys.stderr)

    # Group files by model name
    model_files: dict[str, list[Path]] = defaultdict(list)
    for fpath in json_files:
        try:
            with open(fpath) as fh:
                data = json.load(fh)
            model_name = data["config_general"]["model_name"]
            model_files[model_name].append(fpath)
        except Exception as exc:
            print(f"  [SKIP] {fpath.name}: {exc}", file=sys.stderr)

    print(f"Found {len(model_files)} unique model(s).", file=sys.stderr)

    for model_name, files in model_files.items():
        output = process_model(files)

        safe_name = safe_model_filename(model_name)
        out_file = out_path / f"{safe_name}.json"
        with open(out_file, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2, ensure_ascii=False)
        print(f"  → Written: {out_file}", file=sys.stderr)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(
        input_dir=args[0] if len(args) > 0 else "./results",
        output_dir=args[1] if len(args) > 1 else "./aggregated",
    )
