#!/usr/bin/env python3
"""
Generate an HTML leaderboard from aggregated lighteval benchmark results.

Reads pre-aggregated JSON files from <output_dir>/aggregated/ created by
generate_final_results.py. Each model's task family averages (_AVERAGE|shot)
and overall average (AVERAGE|shot) are used directly.

Each model appears as a row. Each task appears as a top-level column group,
with language × shot as the second level and individual metrics as the third.

Usage:
    python generate_leaderboard.py --output-dir /path/to/outputs \
                                   --leaderboard-dir leaderboard
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from html import escape
from pathlib import Path
from statistics import mean


# ─── Key parsing ─────────────────────────────────────────────────────────────


def parse_task_key(key: str) -> dict | None:  # noqa
    """
    Parse a lighteval task key into structured components.

    This function is retained for backwards compatibility but is no longer
    used in the main pipeline since we now work with pre-aggregated results.
    It can still be useful for debugging or if raw results need to be processed.

    Returns a dict with keys: task_group, language, shot, prompt_num, subtask
    Returns None if the key should be skipped (e.g., "all" key).

    Key format examples:
      arc.prompt1_mqa_single_correct_en|0
      mmlu.prompt1_mqa_single_correct_en:_average|0
      sklep.prompt1_hate_speech_en:hate-speech|0
      senti_sk.prompt1_sentiment_3_class_en|0
      ifeval_sk.ifeval|0
      oneruler_sk.oneruler_english:_average|0
    """
    if key == "all":
        return None

    # ── Split shot ────────────────────────────────────────────────────────────
    shot_sep = key.rfind("|")
    if shot_sep < 0:
        return None
    task_part = key[:shot_sep]
    try:
        shot = int(key[shot_sep + 1 :])
    except ValueError:
        return None

    # ── Split benchmark ───────────────────────────────────────────────────────
    if "." not in task_part:
        return None
    dot_idx = task_part.index(".")
    benchmark = task_part[:dot_idx]
    rest = task_part[dot_idx + 1 :]

    # ── Split optional subtask after ':' ─────────────────────────────────────
    if ":" in rest:
        colon_idx = rest.index(":")
        prompt_part = rest[:colon_idx]
        subtask = rest[colon_idx + 1 :]
    else:
        prompt_part = rest
        subtask = None

    # ── Special case: ifeval_sk ───────────────────────────────────────────────
    if benchmark == "ifeval_sk" and not prompt_part.startswith("prompt"):
        return {
            "task_group": "ifeval_sk",
            "language": "ALL",
            "shot": shot,
            "prompt_num": None,
            "subtask": subtask,
        }

    # ── Special case: oneruler_sk (no prompt number, lang in variant name) ────
    if benchmark == "oneruler_sk" and not prompt_part.startswith("prompt"):
        # Accept per-subtask entries and _average; we'll average subsets later.
        if "english" in prompt_part.lower():
            lang = "EN"
        elif "slovak" in prompt_part.lower():
            lang = "SK"
        else:
            lang = "UNK"
        return {
            "task_group": "oneruler_sk",
            "language": lang,
            "shot": shot,
            "prompt_num": None,
            "subtask": subtask,
        }

    # ── Standard format: promptN_stuff_lang ──────────────────────────────────
    m = re.match(r"^prompt(\d+)_(.+)_(en|sk)$", prompt_part, re.IGNORECASE)
    if not m:
        return None

    prompt_num = int(m.group(1))
    middle = m.group(2)  # e.g. "mqa_single_correct", "sentiment_3_class"
    language = m.group(3).upper()

    # ── Determine task group ──────────────────────────────────────────────────
    if benchmark == "mmlu":
        # Accept individual subjects (subtask) as well as _average; we'll
        # compute the macro-average across subjects later so that subsets
        # present in different JSON files are handled correctly.
        task_group = "mmlu"

    elif benchmark == "truthfulqa":
        task_group = "truthfulqa"

    elif benchmark == "sklep":
        task_group = f"sklep:{subtask}" if subtask else "sklep"

    elif benchmark == "senti_sk":
        task_group = f"senti_sk:{middle}"

    else:
        # arc, belebele, chiby, financial_exams, multi_wiki_qa,
        # nli_sk, skj1, toxic_sk, …
        task_group = benchmark

    return {
        "task_group": task_group,
        "language": language,
        "shot": shot,
        "prompt_num": prompt_num,
        "subtask": subtask,
    }


# ─── Loading & aggregation ────────────────────────────────────────────────────


def find_model_files(output_dir: str) -> dict[str, list[str]]:
    """
    Walk output_dir/aggregated/ and return a mapping
    {model_name: [list_of_json_file_paths]}.

    Reads from aggregated directory where generate_final_results.py writes
    one JSON file per model.
    """
    base = os.path.join(output_dir, "aggregated")
    mapping: dict[str, list[str]] = {}

    if not os.path.exists(base):
        print(f"[WARN] Aggregated directory not found: {base}", file=sys.stderr)
        return mapping

    for fname in sorted(os.listdir(base)):
        if fname.endswith(".json"):
            fpath = os.path.join(base, fname)
            try:
                with open(fpath) as f:
                    data = json.load(f)
                model_name = data.get("model_name", "unknown")
                if model_name not in mapping:
                    mapping[model_name] = []
                mapping[model_name].append(fpath)
            except Exception as e:
                print(f"[WARN] Cannot read {fpath}: {e}", file=sys.stderr)

    return mapping


def load_model(json_files: list[str]) -> tuple[str, dict]:
    """
    Load results from aggregated JSON file(s) belonging to the same model.

    With pre-aggregated data we expect a single file containing:
      - task family averages: "<family>_AVERAGE|shot"
      - overall average: "AVERAGE|shot"

    Returns (model_name, merged_task_results).
    """
    merged: dict[str, dict] = {}
    model_name = None

    for path in json_files:
        with open(path) as f:
            data = json.load(f)

        candidate = data.get("model_name") or data.get("config_general", {}).get("model_name")
        if candidate:
            model_name = candidate

        for task_key, metrics in data.get("results", {}).items():
            # Skip shot-less entries; we keep keys with shot suffix like "AVERAGE|0"
            if "|" not in task_key:
                continue
            if task_key not in merged:
                merged[task_key] = {}
            merged[task_key].update(metrics)

    return model_name or "unknown", merged


def aggregate_results(raw_results: dict) -> dict:  # noqa
    """
    Process pre-aggregated results from generate_final_results.py.

    The input contains:
      - Per-language averages: "<family>_EN_AVERAGE|shot", "<family>_SK_AVERAGE|shot"
      - Combined family average: "<family>_AVERAGE|shot" (SK + EN)
      - Overall language averages: "SK_AVERAGE|shot", "EN_AVERAGE|shot"
      - Overall average: "AVERAGE|shot"

    Returns {(task_group, lang, shot): {metric: value}}.
    """
    aggregated: dict[tuple, dict[str, float]] = {}

    # Metric name normalization map
    METRIC_NORMALIZATION = {
        "em_norm_relaxed": "acc",
        "em_relaxed": "acc",
        "em_relaxed_multi_label": "acc",
        "oneruler_accuracy": "acc",
        "exact_match_slk_prefix": "qem",
        "f1_slk": "qf1",
    }

    def _norm(m: str) -> str:
        return METRIC_NORMALIZATION.get(m, m)

    # Map family names to task group names
    family_to_task = {
        "arc": "arc",
        "analogies_sk": "analogies_sk",
        "belebele": "belebele",
        "chiby": "chiby",
        "driving_test": "driving_tests",
        "e_tests_mqa": "e_tests_mqa",
        "e_tests_qa": "e_tests_qa",
        "financial_exams": "financial_exams",
        "ifeval_sk": "ifeval_sk",
        "globalmmlu_lite": "globalmmlu_lite",
        "mmlu": "mmlu",
        "movie_review": "movie_reviews",
        "multi_wiki_qa": "multi_wiki_qa",
        "nli_sk": "nli_sk",
        "oneruler": "oneruler_sk",
        "scio": "scio",
        "slohry": "slohry",
        "slovak_trivia": "slovak_trivia",
        "slovakcopa": "slovakcopa",
        "slovaksum": "slovaksum",
        "smesum": "smesum",
        "sklep_sentiment_analysis": "sklep:sentiment-analysis",
        "sklep_hate_speech": "sklep:hate-speech",
        "sklep_nli": "sklep:nli",
        "sklep_rte": "sklep:rte",
        "sklep_qa": "sklep:question-answering",
        "senti_sk_binary": "senti_sk:sentiment_binary",
        "senti_sk_3class": "senti_sk:sentiment_3_class",
        "skj1": "skj1",
        "slovak_tests_mqa": "slovak_tests_mqa",
        "slovak_tests_qa": "slovak_tests_qa",
        "toxic_sk": "toxic_sk",
        "truthful_qa": "truthfulqa",
    }

    for key, metrics in raw_results.items():
        # Skip keys without shot separator
        if "|" not in key:
            continue

        parts = key.split("|")
        shot = int(parts[1])

        # Handle "AVERAGE|shot" → AVG task group (overall average across all families)
        if key.startswith("AVERAGE|"):
            col_key = ("AVG", "ALL", shot)
            aggregated[col_key] = {}
            for mkey, mval in metrics.items():
                if mkey.endswith("_stderr") or not isinstance(mval, (int, float)):
                    continue
                nm = _norm(mkey)
                if nm in {"em", "em_norm"}:
                    continue
                aggregated[col_key][nm] = float(mval)
            continue

        # Handle overall language averages: "SK_AVERAGE|shot" or "EN_AVERAGE|shot"
        # These are unweighted averages across all task families for each language
        if re.match(r"^(SK|EN)_AVERAGE\|\d+$", key):
            lang = parts[0]  # "SK" or "EN"
            col_key = ("AVG_LANG", lang, shot)

            if col_key not in aggregated:
                aggregated[col_key] = {}

            for mkey, mval in metrics.items():
                if mkey.endswith("_stderr") or not isinstance(mval, (int, float)):
                    continue
                nm = _norm(mkey)
                if nm in {"em", "em_norm"}:
                    continue
                aggregated[col_key][nm] = float(mval)
            continue

        # Handle per-language family averages: "<family>_EN_AVERAGE|shot" or "<family>_SK_AVERAGE|shot"
        family_name = None
        lang = None

        if key.endswith("_EN_AVERAGE|" + str(shot)):
            base = parts[0]
            family_name = base.rsplit("_EN_AVERAGE", 1)[0]
            lang = "EN"
        elif key.endswith("_SK_AVERAGE|" + str(shot)):
            base = parts[0]
            family_name = base.rsplit("_SK_AVERAGE", 1)[0]
            lang = "SK"

        if family_name and lang:
            task_group = family_to_task.get(family_name, family_name)
            col_key = (task_group, lang, shot)

            if col_key not in aggregated:
                aggregated[col_key] = {}

            for mkey, mval in metrics.items():
                if mkey.endswith("_stderr") or not isinstance(mval, (int, float)):
                    continue
                nm = _norm(mkey)
                if nm in {"em", "em_norm"}:
                    continue
                aggregated[col_key][nm] = float(mval)
            continue

        # Handle combined family average: "<family>_AVERAGE|shot" (skip - already have per-lang)
        if re.match(r"^[a-z_]+_AVERAGE\|\d+$", key):
            # These are the combined SK+EN averages, we skip them since we show per-language
            continue

    return aggregated


def build_column_schema(all_model_data: list[tuple[str, dict]]) -> list[dict]:
    """
    Derive the full column schema for the HTML table from aggregated model data.

    Scans all models to discover which task groups, language/shot combinations,
    and metrics are present. Builds a schema that defines the table structure:
    - Task groups are ordered according to TASK_ORDER (AVG first, then others)
    - Language/shot combinations are sorted by shot number, then language (EN, SK)
    - Metrics are sorted alphabetically within each lang_shot

    Each task family now shows separate columns for EN and SK prompt averages
    (computed across prompt1+prompt2+prompt3 for each language).

    Returns a list of task dicts:
        [
            {
                'task': 'arc',
                'lang_shots': [
                    {'lang_shot': 'EN|0', 'metrics': ['acc', 'qem']},
                    {'lang_shot': 'EN|3', 'metrics': ['acc', 'qem']},
                    {'lang_shot': 'SK|0', 'metrics': ['acc', 'qem']},
                    {'lang_shot': 'SK|3', 'metrics': ['acc', 'qem']},
                ],
                'total_cols': N,
                'is_avg': False,
            },
            ...
        ]
    """
    seen: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))

    for _model_name, agg in all_model_data:
        for (tg, lang, shot), metrics in agg.items():
            ls = f"{lang}|{shot}"
            # Exclude metrics that shouldn't be shown in the leaderboard.
            for m in metrics.keys():
                if m in {"em", "em_norm"}:
                    continue
                seen[tg][ls].add(m)

    TASK_ORDER = [
        "AVG",  # overall average — always first
        "AVG_LANG",  # per-language averages across all tasks — second
        "arc",
        "belebele",
        "chiby",
        "driving_tests",
        "e_tests",
        "financial_exams",
        "ifeval_sk",
        "globalmmlu_lite",
        "mmlu",
        "movie_reviews",
        "multi_wiki_qa",
        "nli_sk",
        "oneruler_sk",
        "senti_sk:sentiment_3_class",
        "senti_sk:sentiment_binary",
        "sklep:hate-speech",
        "sklep:nli",
        "sklep:question-answering",
        "sklep:rte",
        "sklep:sentiment-analysis",
        "skj1",
        "slovak_tests",
        "slovakcopa",
        "slovaksum",
        "slovak_trivia",
        "smesum",
        "toxic_sk",
        "truthfulqa",
        "analogies_sk",
        "slohry",
        "scio"
    ]

    def task_sort_key(task_name: str) -> tuple:
        try:
            return (0, TASK_ORDER.index(task_name), task_name)
        except ValueError:
            return (1, 0, task_name)

    def ls_sort(ls: str) -> tuple:
        lang, shot_str = ls.split("|")
        lang_order = {"EN": 0, "SK": 1, "ALL": 2}.get(lang, 3)
        return (int(shot_str), lang_order)

    schema = []
    for tg in sorted(seen.keys(), key=task_sort_key):
        lang_shots = []
        for ls in sorted(seen[tg].keys(), key=ls_sort):
            metrics_sorted = sorted(seen[tg][ls])
            lang_shots.append({"lang_shot": ls, "metrics": metrics_sorted})

        total_cols = sum(len(ls["metrics"]) for ls in lang_shots)
        is_avg = tg == "AVG"
        is_lang_avg = tg == "AVG_LANG"
        schema.append(
            {
                "task": tg,
                "lang_shots": lang_shots,
                "total_cols": total_cols,
                "is_avg": is_avg,
                "is_lang_avg": is_lang_avg,
            }
        )

    return schema


# ─── Value helpers ────────────────────────────────────────────────────────────


def get_value(agg: dict, task_group: str, lang_shot: str, metric: str) -> float | None:
    """
    Look up a metric value from the aggregated results dictionary.

    Args:
        agg: Aggregated results {(task_group, lang, shot): {metric: value}}
        task_group: Task group name (e.g., "arc", "AVG", "mmlu")
        lang_shot: Language and shot combined (e.g., "ALL|0", "EN|3")
        metric: Metric name (e.g., "acc", "qem", "qf1")

    Returns:
        The metric value if found, None otherwise.
    """
    lang, shot_str = lang_shot.split("|")
    key = (task_group, lang, int(shot_str))
    return agg.get(key, {}).get(metric)


def compute_column_ranges(
    all_model_data: list[tuple[str, dict]],
    schema: list[dict],
) -> dict[tuple, tuple[float, float]]:
    """Return {(task_group, lang_shot, metric): (min_val, max_val)} for colour scaling."""
    ranges: dict[tuple, list[float]] = defaultdict(list)
    for _name, agg in all_model_data:
        for task_item in schema:
            tg = task_item["task"]
            for ls_item in task_item["lang_shots"]:
                ls = ls_item["lang_shot"]
                for metric in ls_item["metrics"]:
                    v = get_value(agg, tg, ls, metric)
                    if v is not None:
                        ranges[(tg, ls, metric)].append(v)

    return {k: (min(vs), max(vs)) for k, vs in ranges.items() if len(vs) > 1}


# ─── HTML generation ─────────────────────────────────────────────────────────

CSS = """
:root {
  --bg: #0f1117;
  --surface: #1a1d27;
  --surface2: #22263a;
  --border: #2e3350;
  --accent: #5b7fff;
  --accent2: #a78bfa;
  --avg-accent: #f59e0b;
  --avg-accent-dim: rgba(245,158,11,.15);
  --text: #e2e8f0;
  --text-muted: #8892b0;
  --header-bg: #1e2235;
  --sticky-bg: #1a1d27;
  --row-hover: #1e2337;
  --font-mono: "JetBrains Mono", "Fira Code", "Consolas", monospace;
  --font-sans: "Inter", system-ui, -apple-system, sans-serif;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-sans);
  min-height: 100vh;
}

/* ── Topbar ── */
.topbar {
  position: sticky;
  top: 0;
  z-index: 100;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 0 24px;
  display: flex;
  align-items: center;
  gap: 24px;
  height: 56px;
  backdrop-filter: blur(8px);
}
.topbar h1 {
  font-size: 1.1rem;
  font-weight: 700;
  letter-spacing: 0.03em;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  white-space: nowrap;
}
.topbar .meta {
  font-size: 0.75rem;
  color: var(--text-muted);
  white-space: nowrap;
}
.topbar .spacer { flex: 1; }
#filter-input {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text);
  font-size: 0.8rem;
  padding: 5px 10px;
  width: 200px;
  outline: none;
  transition: border-color .2s;
}
#filter-input:focus { border-color: var(--accent); }
#filter-input::placeholder { color: var(--text-muted); }

/* ── Wrap ── */
.table-wrap {
  overflow: auto;
  max-height: calc(100vh - 56px);
  padding: 16px;
}

/* ── Table ── */
table {
  border-collapse: separate;
  border-spacing: 0;
  font-size: 0.78rem;
  white-space: nowrap;
  min-width: max-content;
}

/* Sticky model column — use .th-model instead of th:first-child because
   the Model cell has rowspan=3 and is absent from thead rows 2 and 3,
   so th:first-child there would incorrectly pin the first AVG header. */
.th-model, td:first-child {
  position: sticky;
  left: 0;
  z-index: 3;
  background: var(--sticky-bg) !important;
  min-width: 180px;
  max-width: 280px;
  overflow: hidden;
  text-overflow: ellipsis;
}
.th-model { z-index: 5; }
tbody tr:hover       td:first-child { background: var(--row-hover) !important; }
tbody tr:nth-child(even) td:first-child { background: #1c1f2e !important; }
tbody tr:nth-child(even):hover td:first-child { background: var(--row-hover) !important; }

thead th {
  background: var(--header-bg);
  color: var(--text-muted);
  font-weight: 600;
  text-align: center;
  padding: 6px 8px;
  border-bottom: 1px solid var(--border);
  border-right: 1px solid var(--border);
  position: sticky;
}

/* Sticky header rows */
thead tr:nth-child(1) th { top: 0;    z-index: 4; }
thead tr:nth-child(2) th { top: 30px; z-index: 4; }
thead tr:nth-child(3) th { top: 61px; z-index: 4; }
thead tr:nth-child(1) th:first-child { z-index: 6; }

/* Task-level header */
.th-task {
  font-size: 0.82rem;
  font-weight: 700;
  color: var(--accent);
  letter-spacing: 0.04em;
  background: var(--surface2) !important;
  border-top: 2px solid var(--accent) !important;
  padding: 8px 10px !important;
}
/* AVG column group gets a gold accent */
.th-task.th-avg {
  color: var(--avg-accent) !important;
  border-top-color: var(--avg-accent) !important;
  background: rgba(245,158,11,.08) !important;
}
.th-ls.th-avg-ls {
  color: var(--avg-accent) !important;
  background: rgba(245,158,11,.06) !important;
}
.th-metric.th-avg-metric {
  color: var(--avg-accent) !important;
  background: rgba(245,158,11,.06) !important;
}
.th-metric.th-avg-metric:hover {
  background: rgba(245,158,11,.15) !important;
}
.th-metric.th-avg-metric.sort-asc,
.th-metric.th-avg-metric.sort-desc {
  color: var(--avg-accent) !important;
  background: rgba(245,158,11,.2) !important;
}
td.avg-cell {
  background: rgba(245,158,11,.04) !important;
  font-weight: 500;
}
tbody tr:hover td.avg-cell { background: rgba(245,158,11,.10) !important; }

/* Language AVG column group styling (purple accent) */
.th-task.th-avg-lang {
  color: var(--accent2) !important;
  border-top-color: var(--accent2) !important;
  background: rgba(167,139,250,.08) !important;
}
.th-ls.th-avg-lang-ls {
  color: var(--accent2) !important;
  background: rgba(167,139,250,.06) !important;
}
.th-metric.th-avg-lang-metric {
  color: var(--accent2) !important;
  background: rgba(167,139,250,.06) !important;
}
.th-metric.th-avg-lang-metric:hover {
  background: rgba(167,139,250,.15) !important;
}
.th-metric.th-avg-lang-metric.sort-asc,
.th-metric.th-avg-lang-metric.sort-desc {
  color: var(--accent2) !important;
  background: rgba(167,139,250,.2) !important;
}
td.avg-lang-cell {
  background: rgba(167,139,250,.04) !important;
  font-weight: 500;
}
tbody tr:hover td.avg-lang-cell { background: rgba(167,139,250,.10) !important; }

/* Lang|Shot header */
.th-ls {
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--accent2);
  background: var(--header-bg) !important;
  border-bottom: 1px solid var(--border) !important;
}
/* Metric header */
.th-metric {
  font-size: 0.68rem;
  font-weight: 500;
  color: var(--text-muted);
  font-family: var(--font-mono);
  border-right: 1px solid var(--border) !important;
  cursor: pointer;
  user-select: none;
  transition: color .15s, background .15s;
}
.th-metric:hover { color: var(--text); background: var(--surface2) !important; }
.th-metric.sort-asc,
.th-metric.sort-desc { color: var(--accent) !important; background: rgba(91,127,255,.12) !important; }
.th-metric .sort-arrow { margin-left: 3px; opacity: 0.5; font-size: 0.6rem; }
.th-metric.sort-asc .sort-arrow,
.th-metric.sort-desc .sort-arrow { opacity: 1; }

/* Task group separators */
.task-start th { border-left: 2px solid var(--border) !important; }

/* Model column header */
.th-model {
  color: var(--text) !important;
  font-size: 0.8rem;
  text-align: left !important;
  padding-left: 12px !important;
  background: var(--surface2) !important;
  z-index: 6 !important;
}

/* Body */
tbody tr { transition: background .15s; }
tbody tr:hover td { background: var(--row-hover) !important; }
tbody tr:nth-child(even) td { background: rgba(255,255,255,0.02); }

td {
  padding: 5px 7px;
  text-align: right;
  border-right: 1px solid var(--border);
  border-bottom: 1px solid rgba(46,51,80,.4);
  font-family: var(--font-mono);
  font-size: 0.75rem;
  color: var(--text);
}
td:first-child {
  text-align: left;
  font-family: var(--font-sans);
  font-size: 0.8rem;
  padding-left: 12px;
  border-right: 2px solid var(--border);
}
td.missing { color: var(--text-muted); font-size: 0.65rem; }
.task-start-cell { border-left: 2px solid var(--border) !important; }

/* Heat-map cells */
td.val-cell { position: relative; }
td.val-cell::before {
  content: "";
  position: absolute;
  inset: 0;
  opacity: var(--heat-opacity, 0);
  background: var(--heat-color, transparent);
  pointer-events: none;
  z-index: 0;
}
td.val-cell span { position: relative; z-index: 1; }

/* Legend */
.legend {
  display: flex;
  gap: 16px;
  align-items: center;
  font-size: 0.72rem;
  color: var(--text-muted);
  padding: 0 16px 12px;
}
.legend-swatch {
  display: inline-block;
  width: 40px; height: 10px;
  border-radius: 3px;
}
.legend-label { margin-left: 4px; }
"""

JS = """
(function () {
  // ── Filter ────────────────────────────────────────────────────────────────
  const input = document.getElementById('filter-input');

  // Helper function to apply filter
  function applyFilter(q) {
    document.querySelectorAll('tbody tr').forEach(tr => {
      const name = tr.querySelector('td')?.textContent.toLowerCase() ?? '';
      tr.style.display = name.includes(q) ? '' : 'none';
    });
  }

  // Parse URL parameters on page load
  const urlParams = new URLSearchParams(window.location.search);
  const filterParam = urlParams.get('filter');
  if (filterParam) {
    input.value = filterParam;
    applyFilter(filterParam.toLowerCase());
  }

  // Listen to input changes and update URL
  input.addEventListener('input', function () {
    const q = this.value.toLowerCase();
    applyFilter(q);

    // Update URL without page reload
    const newParams = new URLSearchParams(window.location.search);
    if (q) {
      newParams.set('filter', this.value);
    } else {
      newParams.delete('filter');
    }
    const newUrl = newParams.toString() ? `?${newParams.toString()}` : window.location.pathname;
    window.history.replaceState({}, '', newUrl);
  });

  // ── Sort ──────────────────────────────────────────────────────────────────
  // Metric <th> cells live in header row 3. Because the Model <th> spans 3
  // rows it is absent from rows 2–3, so in row 3 cellIndex 0 maps to tbody
  // column 1.  Therefore: td_column_index = th.cellIndex + 1.

  let activeTh = null;
  let sortDesc  = true;

  function getSortValue(tr, colIdx) {
    const td = tr.cells[colIdx];
    if (!td) return null;
    const span = td.querySelector('span');
    const text = span ? span.textContent : td.textContent;
    const v = parseFloat(text);
    return isNaN(v) ? null : v;
  }

  function applySort(th) {
    const colIdx = th.cellIndex + 1;

    if (th === activeTh) {
      sortDesc = !sortDesc;
    } else {
      if (activeTh) {
        activeTh.classList.remove('sort-asc', 'sort-desc');
        const prev = activeTh.querySelector('.sort-arrow');
        if (prev) prev.textContent = ' \u21c5';
      }
      activeTh = th;
      sortDesc  = true;
    }

    th.classList.toggle('sort-desc',  sortDesc);
    th.classList.toggle('sort-asc',  !sortDesc);
    const arrow = th.querySelector('.sort-arrow');
    if (arrow) arrow.textContent = sortDesc ? ' \u25bc' : ' \u25b2';

    const tbody = document.querySelector('tbody');
    const rows  = Array.from(tbody.querySelectorAll('tr'));
    rows.sort((a, b) => {
      const va = getSortValue(a, colIdx);
      const vb = getSortValue(b, colIdx);
      if (va === null && vb === null) return 0;
      if (va === null) return  1;   // nulls always last
      if (vb === null) return -1;
      return sortDesc ? vb - va : va - vb;
    });
    rows.forEach(r => tbody.appendChild(r));
  }

  document.querySelectorAll('.th-metric').forEach(th => {
    th.insertAdjacentHTML('beforeend', '<span class="sort-arrow"> \u21c5</span>');
    th.addEventListener('click', () => applySort(th));
  });
})();
"""


def heat_style(value: float | None, vmin: float, vmax: float) -> str:
    """
    Generate inline CSS custom properties for heat-map cell coloring.

    Maps a value to a color gradient: red (low) → yellow → green (high).
    The opacity is proportional to how high the value is within the range.

    Args:
        value: The metric value to color (None returns empty string)
        vmin: Minimum value in the column (for normalization)
        vmax: Maximum value in the column (for normalization)

    Returns:
        Inline CSS string like "--heat-color:rgb(...);--heat-opacity:0.XXX;"
        or empty string if value is None.
    """
    if value is None:
        return ""
    rng = vmax - vmin
    t = 0.5 if rng < 1e-9 else max(0.0, min(1.0, (value - vmin) / rng))
    opacity = 0.12 + t * 0.35
    if t < 0.5:
        r, g, b = 248, int(113 + t * 2 * (211 - 113)), 113
    else:
        tt = (t - 0.5) * 2
        r, g, b = int(248 + tt * (52 - 248)), 211, int(113 + tt * (153 - 113))
    return f"--heat-color:rgb({r},{g},{b});--heat-opacity:{opacity:.3f};"


def build_html(  # noqa
    all_model_data: list[tuple[str, dict]],
    schema: list[dict],
    col_ranges: dict,
    generated_at: str,
) -> str:
    """
    Build the complete HTML leaderboard page from model data and schema.

    Generates a fully-featured HTML table with:
    - Sticky header rows (3 levels: task group → lang|shot → metric)
    - Sticky first column (model names)
    - Heat-map colored cells based on relative performance
    - Interactive filtering by model name
    - Clickable column headers for sorting
    - Responsive design with scrollable table area

    Args:
        all_model_data: List of (model_name, aggregated_results) tuples
        schema: Column schema from build_column_schema()
        col_ranges: Min/max ranges per column for heat-map scaling
        generated_at: Timestamp string to display in the header

    Returns:
        Complete HTML document as a string.
    """
    n_models = len(all_model_data)
    total_value_cols = sum(t["total_cols"] for t in schema)

    # ── Header row 1: task groups ──────────────────────────────────────────
    hr1_cells = [f'<th rowspan="3" class="th-model">Model ({n_models})</th>']
    for i, task_item in enumerate(schema):
        sep = " task-start" if i > 0 else ""
        is_avg = task_item["is_avg"]
        is_lang_avg = task_item.get("is_lang_avg", False)

        if is_avg:
            label = "⌀ AVG"
            avg_class = " th-avg"
        elif is_lang_avg:
            label = "lang AVG"
            avg_class = " th-avg-lang"
        else:
            label = escape(task_item["task"])
            avg_class = ""

        hr1_cells.append(f'<th colspan="{task_item["total_cols"]}" class="th-task{sep}{avg_class}">{label}</th>')

    # ── Header row 2: lang|shot groups ────────────────────────────────────
    hr2_cells = []
    for i, task_item in enumerate(schema):
        first = True
        is_lang_avg = task_item.get("is_lang_avg", False)
        for ls_item in task_item["lang_shots"]:
            sep = " task-start" if (i > 0 and first) else ""
            avg_class = " th-avg-ls" if task_item["is_avg"] else (" th-avg-lang-ls" if is_lang_avg else "")
            hr2_cells.append(
                f'<th colspan="{len(ls_item["metrics"])}" class="th-ls{sep}{avg_class}">{escape(ls_item["lang_shot"])}</th>'
            )
            first = False

    # ── Header row 3: metric names ─────────────────────────────────────────
    hr3_cells = []
    for i, task_item in enumerate(schema):
        first = True
        for ls_item in task_item["lang_shots"]:
            for j, metric in enumerate(ls_item["metrics"]):
                sep = " task-start" if (i > 0 and first and j == 0) else ""
                if task_item["is_avg"]:
                    avg_class = " th-avg-metric"
                elif is_lang_avg:
                    avg_class = " th-avg-lang-metric"
                else:
                    avg_class = ""
                short = metric.replace("_", "\u200b_")
                hr3_cells.append(f'<th class="th-metric{sep}{avg_class}">{escape(short)}</th>')
                first = False

    thead = (
        "<thead>\n"
        f"  <tr>{''.join(hr1_cells)}</tr>\n"
        f"  <tr>{''.join(hr2_cells)}</tr>\n"
        f"  <tr>{''.join(hr3_cells)}</tr>\n"
        "</thead>"
    )

    # ── Body rows ──────────────────────────────────────────────────────────
    rows_html = []
    for model_name, agg in sorted(all_model_data, key=lambda x: x[0].lower()):
        cells = [f'<td title="{escape(model_name)}">{escape(model_name)}</td>']
        first_task = True
        for task_item in schema:
            tg = task_item["task"]
            is_avg = task_item["is_avg"]
            is_lang_avg = task_item.get("is_lang_avg", False)
            first_in_task = True
            for ls_item in task_item["lang_shots"]:
                ls = ls_item["lang_shot"]
                for metric in ls_item["metrics"]:
                    v = get_value(agg, tg, ls, metric)
                    col_key = (tg, ls, metric)

                    # left border at every task-group boundary
                    extra_cls = ""
                    if (
                        not first_task
                        and first_in_task
                        and ls_item is task_item["lang_shots"][0]
                        and metric == ls_item["metrics"][0]
                    ):
                        extra_cls = " task-start-cell"
                    if is_avg:
                        extra_cls += " avg-cell"
                    elif is_lang_avg:
                        extra_cls += " avg-lang-cell"

                    if v is None:
                        cells.append(f'<td class="missing{extra_cls}">—</td>')
                    else:
                        vmin, vmax = col_ranges.get(col_key, (v, v))
                        hs = heat_style(v, vmin, vmax)
                        style = f' style="{hs}"' if hs else ""
                        cells.append(f'<td class="val-cell{extra_cls}"{style}><span>{v:.4f}</span></td>')
                    first_in_task = False
            first_task = False

        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    tbody = "<tbody>\n" + "\n".join(rows_html) + "\n</tbody>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SK-Bench Leaderboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>{CSS}</style>
</head>
<body>
  <div class="topbar">
    <h1>⚡ SK-Bench Leaderboard</h1>
    <span class="meta">{n_models} model{"s" if n_models != 1 else ""} &nbsp;·&nbsp; {total_value_cols} metric columns &nbsp;·&nbsp; generated {generated_at}</span>
    <div class="spacer"></div>
    <input id="filter-input" type="search" placeholder="Filter models…">
  </div>
  <div class="legend">
    <span>Heat-map:</span>
    <span class="legend-swatch" style="background:linear-gradient(to right,rgb(248,113,113),rgb(248,211,113),rgb(52,211,153))"></span>
    <span class="legend-label">low → high within each metric column</span>
  </div>
  <div class="table-wrap">
    <table>
      {thead}
      {tbody}
    </table>
  </div>
  <script>{JS}</script>
</body>
</html>"""


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """
    Main entry point: parse arguments, load data, generate leaderboard HTML.

    Workflow:
        1. Find all aggregated JSON files in <output_dir>/aggregated/
        2. Load and merge results for each model
        3. Transform to aggregated format {(task_group, lang, shot): metrics}
        4. Build column schema by scanning all models
        5. Compute min/max ranges for heat-map coloring
        6. Generate HTML and write to <leaderboard-dir>/index.html
    """
    parser = argparse.ArgumentParser(
        description="Generate HTML leaderboard from aggregated lighteval results."
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Root directory containing aggregated/ subfolder with model JSON files (default: ./aggregated)",
    )
    parser.add_argument(
        "--leaderboard-dir",
        default="leaderboard",
        help="Output directory for the HTML file (default: leaderboard)",
    )
    args = parser.parse_args()

    model_files = find_model_files(args.output_dir)
    if not model_files:
        print(f"No results found under {args.output_dir}/aggregated/")
        return

    print(f"Found {len(model_files)} model(s):")
    all_model_data: list[tuple[str, dict]] = []
    for model_path, json_files in sorted(model_files.items()):
        model_name, raw = load_model(json_files)
        print(f"  Loaded {len(raw)} raw keys from {model_name}:")
        
        agg = aggregate_results(raw)
        all_model_data.append((model_name, agg))

    schema = build_column_schema(all_model_data)
    col_ranges = compute_column_ranges(all_model_data, schema)

    print(f"\nColumn schema: {len(schema)} task group(s)")
    for t in schema:
        ls_labels = [ls["lang_shot"] for ls in t["lang_shots"]]
        print(f"  {'[AVG]' if t['is_avg'] else t['task']:40s}  lang×shot: {ls_labels}  cols: {t['total_cols']}")

    from datetime import datetime, timezone

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = build_html(all_model_data, schema, col_ranges, generated_at)

    Path(args.leaderboard_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(args.leaderboard_dir, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nLeaderboard written to: {out_path}  ({os.path.getsize(out_path) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
