#!/usr/bin/env python3
"""Generate paper-ready LaTeX result tables from the aggregated sk-bench JSONs.

Reads ``final_results/<family>/aggregated/*.json`` and emits booktabs LaTeX
tables. The same engine produces the comprehensive table (all task groups), the
thematic subset tables, and the thinking-models table — they differ only by the
selected task groups and model family.

Cell value = Slovak (SK), 0-shot, one canonical primary metric per task group
(see ``CANONICAL_METRIC``), read from ``<group>_SK_AVERAGE|0``. Values are shown
as percentages; the best per column is bold, the second-best underlined.

Open-weight models are ordered by parameter count (ascending); proprietary /
undisclosed-size models form a separate block placed last, ordered by the table
average. Sizes are only used for ordering — they are not displayed — so the
estimates for a few closed / MoE models do not appear in the output.

Usage:
    python scripts/generate_latex_tables.py --output-dir <dir> [--which all]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

# ── task groups: key → (label, theme); order = display order ────────────────────
# Canonical primary metric per group (verified against the aggregates).
CANONICAL_METRIC: dict[str, str] = {
    # MCQ / knowledge / classification → accuracy
    **{g: "acc" for g in (
        "arc", "mmlu", "globalmmlu_lite", "truthful_qa", "scio", "financial_exams",
        "driving_test", "slovakcopa", "skj1", "slovak_trivia", "belebele", "chiby",
        "e_tests_mqa", "slovak_tests_mqa", "nli_sk", "sklep_nli", "sklep_rte",
        "senti_sk_3class", "senti_sk_binary", "sklep_sentiment_analysis",
        "movie_review", "toxic_sk", "sklep_hate_speech", "analogies_sk", "slohry",
        "oneruler",
    )},
    # open-ended QA → quasi-exact-match
    **{g: "qem" for g in ("multi_wiki_qa", "e_tests_qa", "slovak_tests_qa", "sklep_qa")},
    # summarization → ROUGE-L
    "slovaksum": "rougeL",
    "smesum": "rougeL",
    # instruction following → prompt-level strict accuracy
    "ifeval_sk": "prompt_level_strict_acc",
}

# group key → (short column label, theme)
TASK_GROUPS: list[tuple[str, str, str]] = [
    ("arc",                      "ARC",        "Knowledge & Reasoning"),
    ("mmlu",                     "MMLU",       "Knowledge & Reasoning"),
    ("globalmmlu_lite",          "G-MMLU",     "Knowledge & Reasoning"),
    ("truthful_qa",              "TruthQA",    "Knowledge & Reasoning"),
    ("scio",                     "SCIO",       "Knowledge & Reasoning"),
    ("financial_exams",          "FinExam",    "Knowledge & Reasoning"),
    ("driving_test",             "Driving",    "Knowledge & Reasoning"),
    ("slovakcopa",               "COPA",       "Knowledge & Reasoning"),
    ("skj1",                     "SKJ1",       "Knowledge & Reasoning"),
    ("slovak_trivia",            "Trivia",     "Knowledge & Reasoning"),
    ("belebele",                 "Belebele",   "Reading & QA"),
    ("chiby",                    "Chiby",      "Reading & QA"),
    ("multi_wiki_qa",            "WikiQA",     "Reading & QA"),
    ("e_tests_mqa",              "ETest-M",    "Reading & QA"),
    ("e_tests_qa",               "ETest-Q",    "Reading & QA"),
    ("slovak_tests_mqa",         "SKTest-M",   "Reading & QA"),
    ("slovak_tests_qa",          "SKTest-Q",   "Reading & QA"),
    ("oneruler",                 "OneRuler",   "Reading & QA"),
    ("sklep_qa",                 "sKLEP-QA",   "Reading & QA"),
    ("nli_sk",                   "NLI",        "Classification"),
    ("sklep_nli",                "sKLEP-NLI",  "Classification"),
    ("sklep_rte",                "sKLEP-RTE",  "Classification"),
    ("senti_sk_3class",          "Senti-3",    "Classification"),
    ("senti_sk_binary",          "Senti-2",    "Classification"),
    ("sklep_sentiment_analysis", "sKLEP-Sent", "Classification"),
    ("movie_review",             "MovieRev",   "Classification"),
    ("toxic_sk",                 "Toxic",      "Classification"),
    ("sklep_hate_speech",        "sKLEP-Hate", "Classification"),
    ("slovaksum",                "SlovakSum",  "Generation"),
    ("smesum",                   "SMESum",     "Generation"),
    ("ifeval_sk",                "IFEval",     "Instruction"),
    ("analogies_sk",             "Analogies",  "Grammar & Morphology"),
    ("slohry",                   "Slohry",     "Grammar & Morphology"),
]

THEME_ORDER = [
    "Knowledge & Reasoning", "Reading & QA", "Classification",
    "Generation", "Instruction", "Grammar & Morphology",
]

# task key -> \label{task:<slug>} in the paper's Task Descriptions appendix
# (app:tasks-detail). Per-task column headers are wrapped in \hyperref so a
# reader can click a task name and jump to its description. Multiple columns
# of the same dataset (e.g. ETest-M/-Q) share one slug.
TASK_DOC_SLUG = {
    "arc": "arc-sk", "mmlu": "mmlu-sk", "globalmmlu_lite": "gmmlu",
    "truthful_qa": "truthqa-sk", "scio": "scio", "financial_exams": "finexam",
    "driving_test": "autoskola", "slovakcopa": "copa", "skj1": "skj1",
    "slovak_trivia": "trivia", "belebele": "belebele", "chiby": "chiby",
    "multi_wiki_qa": "wikiqa", "e_tests_mqa": "etest", "e_tests_qa": "etest",
    "slovak_tests_mqa": "sktest", "slovak_tests_qa": "sktest",
    "oneruler": "oneruler", "sklep_qa": "sklep-qa", "nli_sk": "nli-sk",
    "sklep_nli": "sklep-nli", "sklep_rte": "sklep-rte",
    "senti_sk_3class": "senti", "senti_sk_binary": "senti",
    "sklep_sentiment_analysis": "sklep-sent", "movie_review": "movierev",
    "toxic_sk": "toxic", "sklep_hate_speech": "sklep-hate",
    "slovaksum": "slovaksum", "smesum": "smesum", "ifeval_sk": "ifeval",
    "analogies_sk": "analogies", "slohry": "slohry",
}


def task_header(key: str, label: str) -> str:
    """Column label, made a \\hyperref jump to its appendix entry if known."""
    slug = TASK_DOC_SLUG.get(key)
    return f"\\hyperref[task:{slug}]{{{label}}}" if slug else label

# short column labels for the compact summary leaderboard
# Full reporting-group names spelled out for the leaderboard column headers.
THEME_HEADER = {
    "Knowledge & Reasoning": "Knowledge",
    "Reading & QA":          "Reading",
    "Classification":        "Classification",
    "Generation":            "Generation",
    "Instruction":           "Instruction",
    "Grammar & Morphology":  "Grammar",
}


def groups_in_theme(theme: str) -> list[str]:
    return [g for g, _, th in TASK_GROUPS if th == theme]

# ── model metadata ──────────────────────────────────────────────────────────────
# display name, parameter count in billions (None = undisclosed), is_open.
# params are used ONLY for row ordering (not displayed); a few closed / MoE
# values are best-effort and easy to edit. `is_open=False` => proprietary block.
M = lambda name, p, openw: {"name": name, "params": p, "open": openw}  # noqa: E731
MODEL_META: dict[str, dict] = {
    # ── proprietary / API-only (placed last) ─────────────────────────────────
    "anthropic.claude-opus-4-6":            M("Claude Opus 4.6",   None, False),
    "anthropic.claude-sonnet-4-6":          M("Claude Sonnet 4.6", None, False),
    "openai__gpt-5.5":                      M("GPT-5.5",           None, False),
    "openai__gpt-5.4-mini":                 M("GPT-5.4 mini",      None, False),
    "google__gemini-3.1-pro-preview":       M("Gemini 3.1 Pro",   None, False),
    "google__gemini-3.1-flash-lite-preview":M("Gemini 3.1 Flash-Lite", None, False),
    # ── open-weight ───────────────────────────────────────────────────────────
    "ibm-granite__granite-4.0-350m":        M("Granite 4.0 350M", 0.35, True),
    "ibm-granite__granite-4.0-h-350m":      M("Granite 4.0-H 350M", 0.35, True),
    "Qwen__Qwen3-0.6B":                     M("Qwen3 0.6B",       0.6, True),
    "Qwen__Qwen3.5-0.8B":                   M("Qwen3.5 0.8B",     0.8, True),
    "google__gemma-3-1b-it":                M("Gemma 3 1B",       1.0, True),
    "ibm-granite__granite-4.0-1b":          M("Granite 4.0 1B",   1.0, True),
    "ibm-granite__granite-4.0-h-1b":        M("Granite 4.0-H 1B", 1.0, True),
    "utter-project__EuroLLM-1.7B-Instruct": M("EuroLLM 1.7B",     1.7, True),
    "Qwen__Qwen3-1.7B":                     M("Qwen3 1.7B",       1.7, True),
    "ibm-granite__granite-3.2-2b-instruct": M("Granite 3.2 2B",   2.0, True),
    "Qwen__Qwen3.5-2B":                     M("Qwen3.5 2B",       2.0, True),
    "meta-llama__Llama-3.2-3B-Instruct":    M("Llama 3.2 3B",     3.0, True),
    "ibm-granite__granite-4.0-h-small":     M("Granite 4.0-H Small", 32.0, True),
    "CohereLabs__tiny-aya-global":          M("Tiny-Aya Global",  3.35, True),
    "CohereLabs__tiny-aya-water":           M("Tiny-Aya Water",   3.35, True),
    "mrshu__mistral-sk-7b-alpaca-slovak-it":M("Mistral-SK 7B (Alpaca)", 7.0, True),
    "slovak-nlp__mistral-sk-7b":            M("Mistral-SK 7B",    7.0, True),
    "openGPT-X__Teuken-7B-instruct-v0.6":   M("Teuken 7B",        7.0, True),
    "meta-llama__Llama-3.1-8B-Instruct":    M("Llama 3.1 8B",     8.0, True),
    "CohereLabs__aya-expanse-8b":           M("Aya Expanse 8B",   8.0, True),
    "mistralai__Ministral-3-8B-Instruct-2512":  M("Ministral 3 8B", 8.0, True),
    "mistralai__Ministral-3-8B-Reasoning-2512": M("Ministral 3 8B (R)", 8.0, True),
    "utter-project__EuroLLM-9B-Instruct-2512":  M("EuroLLM 9B",   9.0, True),
    "Qwen__Qwen3.5-9B":                     M("Qwen3.5 9B",       9.0, True),
    "speakleash__Bielik-11B-v3.0-Instruct": M("Bielik 11B",       11.0, True),
    "Qwen__Qwen3-14B":                      M("Qwen3 14B",        14.0, True),
    "mrshu__qwen3-14b-sk-alpaca-slovak-it": M("Qwen3-SK 14B (Alpaca)", 14.0, True),
    "slovak-nlp__Qwen3-14B-sk":             M("Qwen3-SK 14B",     14.0, True),
    "mistralai__Ministral-3-14B-Instruct-2512":  M("Ministral 3 14B", 14.0, True),
    "mistralai__Ministral-3-14B-Reasoning-2512": M("Ministral 3 14B (R)", 14.0, True),
    "openai__gpt-oss-20b":                  M("gpt-oss 20B",      20.0, True),
    "utter-project__EuroLLM-22B-Instruct-2512": M("EuroLLM 22B",  22.0, True),
    "Qwen__Qwen3.5-27B":                    M("Qwen3.5 27B",      27.0, True),
    "Qwen__Qwen3.6-27B":                    M("Qwen3.6 27B",      27.0, True),
    "TildeAI__TildeOpen-30b":               M("TildeOpen 30B",    30.0, True),
    "TildeAI__TildeOpen-30b-64k":           M("TildeOpen 30B-64k", 30.0, True),
    "mrshu__tildeopen-30b-64k-sk-alpaca-slovak-it": M("TildeOpen-SK 30B (Alpaca)", 30.0, True),
    "google__gemma-4-31B-it":               M("Gemma 4 31B",      31.0, True),
    "CohereLabs__aya-expanse-32b":          M("Aya Expanse 32B",  32.0, True),
    "Qwen__Qwen3.6-35B-A3B":                M("Qwen3.6 35B-A3B",  35.0, True),
    "meta-llama__Llama-3.3-70B-Instruct":   M("Llama 3.3 70B",    70.0, True),
    "swiss-ai__Apertus-70B-Instruct-2509":  M("Apertus 70B",      70.0, True),
    "CohereLabs__c4ai-command-a-03-2025":   M("Command A",        111.0, True),
    "openai__gpt-oss-120b":                 M("gpt-oss 120B",     120.0, True),
    "MiniMaxAI__MiniMax-M2.7":              M("MiniMax M2.7",     230.0, True),
    "zai-org__GLM-5.1-FP8":                 M("GLM-5.1",          754.0, True),
    "Qwen__Qwen3.5-397B-A17B":              M("Qwen3.5 397B-A17B", 397.0, True),
    "mistralai__Mistral-Large-3-675B-Instruct-2512": M("Mistral Large 3 675B", 675.0, True),
    "moonshotai__Kimi-K2.6":                M("Kimi K2.6",        1100.0, True),
    # ── thinking-family extras (Qwen + gemma already above) ──────────────────
}


def metric_for(group: str) -> str:
    return CANONICAL_METRIC[group]


def lang_value(results: dict, group: str, lang: str) -> float | None:
    """0-shot primary-metric value for a task group under prompt language
    ``lang`` (``"SK"`` or ``"EN"``), or None if absent. EN reads only the
    ``EN_AVERAGE`` aggregate (no fallback); SK falls back through the
    historical averaged suffixes."""
    metric = metric_for(group)
    if lang == "EN":
        entry = results.get(f"{group}_EN_AVERAGE|0")
        return entry[metric] if entry and metric in entry else None
    for suffix in ("SK_AVERAGE", "AVERAGE", "UNK_AVERAGE"):
        entry = results.get(f"{group}_{suffix}|0")
        if entry and metric in entry:
            return entry[metric]
    return None


def sk_value(results: dict, group: str) -> float | None:
    """SK 0-shot primary-metric value for a task group, or None if absent."""
    return lang_value(results, group, "SK")


# Tasks with no English prompt (Slovak-specific by construction): IFEval-SK's
# checkers and Slohry's morphology. Excluded from the EN table and comparison.
SLOVAK_ONLY = {"ifeval_sk", "slohry"}


def load_family(family: str, lang: str = "SK") -> dict[str, dict[str, float | None]]:
    """Return {model_key: {group: value or None}} for one results family,
    under prompt language ``lang`` (``"SK"`` default, or ``"EN"``)."""
    base = Path("final_results") / family / "aggregated"
    out: dict[str, dict[str, float | None]] = {}
    for fpath in sorted(base.glob("*.json")):
        key = fpath.stem
        results = json.loads(fpath.read_text())["results"]
        out[key] = {g: lang_value(results, g, lang) for g, _, _ in TASK_GROUPS}
    return out


def lang_prompt_values(results: dict, group: str, lang: str) -> dict[str, float] | None:
    """Per-prompt primary-metric values for a task group under prompt language
    ``lang`` (``{prompt_id: value}``), or None if not recorded. Used to derive
    the across-prompt spread (std over the prompt templates)."""
    metric = metric_for(group)
    suffix = "EN_AVERAGE" if lang == "EN" else "SK_AVERAGE"
    entry = results.get(f"{group}_{suffix}|0")
    if entry:
        pv = entry.get(f"{metric}__prompts")
        if isinstance(pv, dict) and len(pv) >= 2:
            return {k: float(v) for k, v in pv.items()}
    return None


def load_family_prompts(family: str, lang: str = "SK") -> dict[str, dict[str, dict[str, float] | None]]:
    """{model: {group: {prompt_id: value} or None}} for one results family."""
    base = Path("final_results") / family / "aggregated"
    out: dict[str, dict] = {}
    for fpath in sorted(base.glob("*.json")):
        results = json.loads(fpath.read_text())["results"]
        out[fpath.stem] = {g: lang_prompt_values(results, g, lang) for g, _, _ in TASK_GROUPS}
    return out


def table_average(row: dict[str, float | None], groups: list[str]) -> float | None:
    vals = [row[g] for g in groups if row.get(g) is not None]
    return sum(vals) / len(vals) if vals else None


# Open-weight size tiers (upper bound exclusive, in billions of params) and their
# table labels. Grounded in the Open LLM Leaderboard's parameter binning and the
# size-tiered analysis of BenCzechMark (Fajcik et al., TACL 2025).
SIZE_TIERS: list[tuple[float, str]] = [
    (2.0,            r"$<$2B"),
    (9.0,            r"2--9B"),
    (33.0,           r"9--32B"),   # inclusive of 32B (e.g. Granite-4.0-H-Small, Aya-Expanse-32B)
    (200.0,          r"32--200B"),
    (float("inf"),   r"$>$200B"),
]
PROPRIETARY_LABEL = "Proprietary (size undisclosed)"


def size_tier(params: float) -> str:
    for upper, lab in SIZE_TIERS:
        if params < upper:
            return lab
    return SIZE_TIERS[-1][1]


def order_models(data: dict[str, dict], groups: list[str]) -> list[tuple[str, list[str]]]:
    """Group rows into (tier_label, [model_keys]).

    Open-weight models are split into size tiers (ascending); the proprietary /
    undisclosed-size models form a single block placed last, sorted by table avg.
    Only non-empty tiers are returned.
    """
    def avg(k):
        a = table_average(data[k], groups)
        return a if a is not None else -1.0

    unknown = [k for k in data if k not in MODEL_META]
    if unknown:
        raise SystemExit(f"Models missing from MODEL_META: {unknown}")

    open_models = sorted(
        [k for k in data if MODEL_META[k]["open"]],
        key=lambda k: (MODEL_META[k]["params"], MODEL_META[k]["name"].lower()),
    )
    closed_models = sorted([k for k in data if not MODEL_META[k]["open"]],
                           key=avg, reverse=True)

    tiers: list[tuple[str, list[str]]] = []
    for _, lab in SIZE_TIERS:
        ks = [k for k in open_models if size_tier(MODEL_META[k]["params"]) == lab]
        if ks:
            tiers.append((lab, ks))
    if closed_models:
        tiers.append((PROPRIETARY_LABEL, closed_models))
    return tiers


def fmt(v: float | None) -> str:
    return "--" if v is None else f"{v * 100:.1f}"


def rank_styles(values: list[float | None]) -> dict[int, str]:
    """Map column-row index → 'bold'|'under' for best / second-best (higher better)."""
    indexed = [(i, v) for i, v in enumerate(values) if v is not None]
    indexed.sort(key=lambda t: t[1], reverse=True)
    styles: dict[int, str] = {}
    if indexed:
        styles[indexed[0][0]] = "bold"
    if len(indexed) > 1 and indexed[1][1] != indexed[0][1]:
        styles[indexed[1][0]] = "under"
    return styles


def cell(v: float | None, style: str | None) -> str:
    s = fmt(v)
    if style == "bold":
        return f"\\textbf{{{s}}}"
    if style == "under":
        return f"\\underline{{{s}}}"
    return s


def render_table(data, tiers, groups, labels, caption, label) -> str:
    """Render one booktabs LaTeX table with size-tiered model blocks.

    `tiers` is a list of (tier_label, [model_keys]); each becomes an italic
    header row followed by its models, separated by \\midrule. Bold/underline
    ranking is GLOBAL (best/second per task across all models in the table).

    Always a two-column-spanning ``table*`` with rotated column headers and
    ``\\resizebox`` to the text width. The many-column per-task tables scale
    *down* to fit, so an upright float works; we never use ``sidewaystable``.
    """
    rows = [m for _, ks in tiers for m in ks]
    col_keys = list(groups) + ["__avg__"]
    col_values = {g: [data[m][g] for m in rows] for g in groups}
    col_values["__avg__"] = [table_average(data[m], groups) for m in rows]
    col_styles = {c: rank_styles(col_values[c]) for c in col_keys}

    ncol = len(groups)
    colspec = "l" + "r" * ncol + "|r"
    head_cells = [f"\\rotatebox{{90}}{{{task_header(key, lab)}}}"
                  for key, lab in zip(groups, labels)] + ["\\rotatebox{90}{\\textbf{Avg}}"]

    lines: list[str] = []
    lines.append("% Requires: \\usepackage{booktabs}, graphicx (\\resizebox)")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Model} & " + " & ".join(head_cells) + r" \\")
    lines.append("\\midrule")

    i = 0
    for t, (tlabel, ks) in enumerate(tiers):
        if t > 0:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{ncol + 2}}}{{l}}{{\\textit{{{tlabel}}}}} \\\\")
        for m in ks:
            name = MODEL_META[m]["name"].replace("&", "\\&")
            cells = [cell(data[m][g], col_styles[g].get(i)) for g in groups]
            cells.append(cell(table_average(data[m], groups), col_styles["__avg__"].get(i)))
            lines.append(f"{name} & " + " & ".join(cells) + r" \\")
            i += 1

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


CAPTION_NOTE = ("Slovak (SK), 0-shot. Each cell is the task's primary metric "
                "($\\times100$): accuracy for multiple-choice and classification, "
                "quasi-exact-match (qem) for open QA, ROUGE-L for summarisation, "
                "and prompt-level strict accuracy for IFEval. \\textbf{Best} and "
                "\\underline{second-best} per column. Avg is the unweighted mean "
                "of the columns shown.")

BODY_CAPTION = (
    "Headline sk-bench leaderboard (Slovak, 0-shot): the top three "
    "open-weights models per size tier (by Overall) and the top three "
    "proprietary APIs. Each cell is the unweighted mean of that group's "
    "per-task primary metrics ($\\times100$); \\textbf{best} and "
    "\\underline{second-best} per column are over \\emph{all 55} models, so "
    "a mark appears only where a shown model holds the record. \\textbf{Overall} "
    "is the unweighted mean of the six group averages. The small grey "
    "$\\pm$ value is the standard deviation across the three prompt templates. "
    "The full 55-model leaderboard is in Appendix~\\ref{app:detailed} "
    "(Table~\\ref{tab:results-full}); per-task scores in "
    "Table~\\ref{tab:skbench-comprehensive}.")

FULL_CAPTION = (
    "Full sk-bench leaderboard: all 55 models by reporting group, grouped by "
    "size tier (Slovak, 0-shot). Each cell is the unweighted mean of that "
    "group's per-task primary metrics ($\\times100$); \\textbf{best} and "
    "\\underline{second-best} per column. \\textbf{Overall} is the unweighted "
    "mean of the six group averages. The small grey $\\pm$ value is the "
    "standard deviation across the three prompt templates. The body "
    "leaderboard (Table~\\ref{tab:results-main}) shows the top three per "
    "tier; per-task scores are in Table~\\ref{tab:skbench-comprehensive}.")


def _extremes(vals: list[float | None]) -> tuple[float | None, float | None]:
    """Distinct best and second-best values (higher better)."""
    s = sorted({v for v in vals if v is not None}, reverse=True)
    return (s[0] if s else None, s[1] if len(s) > 1 else None)


def _prompt_overall_std(prompts_m) -> float | None:
    """Std across prompt templates of the Overall score: per prompt id, the
    Overall is the macro-mean of the six theme scores (each a mean over its
    tasks' per-prompt values); std is taken over those per-prompt Overalls."""
    pids: set[str] = set()
    for th in THEME_ORDER:
        for g in groups_in_theme(th):
            if prompts_m.get(g):
                pids |= set(prompts_m[g].keys())
    per_pid = []
    for pid in pids:
        theme_vals = []
        for th in THEME_ORDER:
            tv = [prompts_m[g][pid] for g in groups_in_theme(th)
                  if prompts_m.get(g) and pid in prompts_m[g]]
            if tv:
                theme_vals.append(sum(tv) / len(tv))
        if theme_vals:
            per_pid.append(sum(theme_vals) / len(theme_vals))
    if len(per_pid) >= 2:
        return st.stdev(per_pid)
    return 0.0 if per_pid else None


def _prompt_group_std(prompts_m, group_keys) -> float | None:
    """Std across prompt templates of a reporting group's score: for each
    prompt id, the group score = mean over the group's tasks of that task's
    per-prompt value; std is taken over those per-prompt group scores."""
    pids: set[str] = set()
    for g in group_keys:
        if prompts_m.get(g):
            pids |= set(prompts_m[g].keys())
    per_pid = []
    for pid in pids:
        tv = [prompts_m[g][pid] for g in group_keys
              if prompts_m.get(g) and pid in prompts_m[g]]
        if tv:
            per_pid.append(sum(tv) / len(tv))
    if len(per_pid) >= 2:
        return st.stdev(per_pid)
    return 0.0 if per_pid else None


def build_theme_data(data, prompts=None):
    """Return (means, stds). ``means[model]`` = {theme: group-average,
    '__overall__': macro-average}; ``stds[model]`` = the matching across-prompt
    std for each, or None if no per-prompt data was supplied."""
    means: dict[str, dict[str, float | None]] = {}
    stds: dict[str, dict[str, float | None]] = {}
    for m in data:
        pm = (prompts or {}).get(m, {})
        tm = {th: table_average(data[m], groups_in_theme(th)) for th in THEME_ORDER}
        vals = [tm[th] for th in THEME_ORDER if tm[th] is not None]
        tm["__overall__"] = sum(vals) / len(vals) if vals else None
        ts = {th: _prompt_group_std(pm, groups_in_theme(th)) for th in THEME_ORDER}
        ts["__overall__"] = _prompt_overall_std(pm)
        means[m] = tm
        stds[m] = ts
    return means, stds


def top_n_per_tier(theme_data, tiers, n):
    """Pick each tier's top-n models by Overall, then display them sorted by
    parameter size (ascending) where size is known; proprietary models
    (undisclosed size) keep Overall order. Mirrors the size ordering of the
    full leaderboard."""
    out = []
    for lab, ks in tiers:
        chosen = sorted(ks, key=lambda k: (theme_data[k]["__overall__"] or -1.0),
                        reverse=True)[:n]
        if all(MODEL_META[k]["params"] is not None for k in chosen):
            chosen = sorted(chosen, key=lambda k: MODEL_META[k]["params"])
        out.append((lab, chosen))
    return out


def render_leaderboard(theme_data, theme_std, render_tiers, label, caption,
                       resize=False, longtable=False) -> str:
    """Leaderboard: model x six reporting-group averages + Overall, tier-grouped.
    ``theme_data`` supplies the FULL model set so best/second-best are ranked
    globally; ``theme_std`` gives the matching across-prompt std, rendered as a
    small grey $\\pm\\sigma$; ``render_tiers`` controls which rows are printed
    (top-n for the body, all for the appendix).

    The body table is a two-column-spanning ``table*`` (``resize=True``). The
    full 55-model table is tall and narrow, so ``\\resizebox`` would scale it
    *up* and overflow the page; instead ``longtable=True`` emits a multi-page
    ``longtable`` wrapped in ``\\onecolumn``/``\\twocolumn`` (matching the
    dataset catalogue). We never use ``sidewaystable``."""
    cols = THEME_ORDER + ["__overall__"]
    ext = {c: _extremes([theme_data[m][c] for m in theme_data]) for c in cols}

    def fcell(v, c, sd):
        if v is None:
            return "--"
        s = f"{v * 100:.1f}"
        best, second = ext[c]
        if best is not None and abs(v - best) < 1e-9:
            s = f"\\textbf{{{s}}}"
        elif second is not None and abs(v - second) < 1e-9:
            s = f"\\underline{{{s}}}"
        if sd is not None:
            s += f"\\,{{\\tiny\\color{{gray}}$\\pm${sd * 100:.1f}}}"
        return s

    ncol = len(THEME_ORDER)
    colspec = "l" + "r" * ncol + "|r"
    head = "\\textbf{Model} & " + " & ".join(
        [f"\\textbf{{{THEME_HEADER[th]}}}" for th in THEME_ORDER]
        + ["\\textbf{Overall}"]) + r" \\"

    def body_rows() -> list[str]:
        out: list[str] = []
        for ti, (tlabel, ks) in enumerate(render_tiers):
            if ti > 0:
                out.append("\\midrule")
            out.append(f"\\multicolumn{{{ncol + 2}}}{{l}}{{\\textit{{{tlabel}}}}} \\\\")
            for m in ks:
                name = MODEL_META[m]["name"].replace("&", "\\&")
                cells = [fcell(theme_data[m][th], th, theme_std[m][th])
                         for th in THEME_ORDER]
                cells.append(fcell(theme_data[m]["__overall__"], "__overall__",
                                   theme_std[m]["__overall__"]))
                out.append(f"{name} & " + " & ".join(cells) + r" \\")
        return out

    if longtable:
        # Full-page, multi-page, upright. caption/label go inside the longtable.
        L = ["% Requires: \\usepackage{booktabs}, longtable",
             "\\onecolumn", "\\begingroup", "\\small",
             "\\setlength{\\tabcolsep}{3pt}",
             f"\\begin{{longtable}}{{{colspec}}}",
             f"\\caption{{{caption}}}\\label{{{label}}}\\\\",
             "\\toprule", head, "\\midrule", "\\endfirsthead",
             f"\\multicolumn{{{ncol + 2}}}{{l}}"
             "{\\small\\itshape continued from previous page}\\\\[2pt]",
             "\\toprule", head, "\\midrule", "\\endhead",
             "\\midrule",
             f"\\multicolumn{{{ncol + 2}}}{{r}}"
             "{\\small\\itshape continued on next page}\\\\",
             "\\endfoot", "\\bottomrule", "\\endlastfoot"]
        L += body_rows()
        L += ["\\end{longtable}", "\\endgroup", "\\twocolumn"]
        return "\n".join(L) + "\n"

    L = ["% Requires: \\usepackage{booktabs}"
         + (", graphicx (\\resizebox)" if resize else ""),
         "\\begin{table*}[t]", "\\centering\\small",
         "\\setlength{\\tabcolsep}{5pt}"]
    if resize:
        L.append("\\resizebox{\\linewidth}{!}{%")
    L += [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", head, "\\midrule"]
    L += body_rows()
    L += ["\\bottomrule", "\\end{tabular}"]
    if resize:
        L.append("}")
    L += [f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table*}"]
    return "\n".join(L) + "\n"


EN_CAPTION_NOTE = (
    "English (EN) prompts, 0-shot --- the English-template counterpart of "
    "Table~\\ref{tab:skbench-comprehensive}. IFEval-SK and Slohry are "
    "Slovak-only by construction and omitted. Each cell is the task's primary "
    "metric ($\\times100$); \\textbf{best} and \\underline{second-best} per "
    "column. Avg is the unweighted mean of the columns shown.")

LANG_CAPTION = (
    "Prompt-language effect: mean Slovak- vs.\\ English-prompt score per "
    "reporting group, macro-averaged across all evaluated models over the "
    "tasks available in both languages (IFEval-SK and Slohry are Slovak-only "
    "and excluded). $\\Delta = \\text{EN} - \\text{SK}$; a positive value "
    "means the English prompt scores higher. Full per-task English scores are "
    "in Table~\\ref{tab:skbench-comprehensive-en}.")


def render_lang_comparison(data_sk, data_en, label) -> str:
    """Per reporting-group SK vs EN means (macro-averaged across models) and
    their delta, over tasks present in both languages."""
    en_groups = [g for g, _, _ in TASK_GROUPS if g not in SLOVAK_ONLY]
    models = list(data_sk)

    def means(group_set):
        sk_acc, en_acc = [], []
        for m in models:
            pairs = [(data_sk[m][g], data_en[m][g]) for g in group_set
                     if data_sk[m].get(g) is not None and data_en[m].get(g) is not None]
            if pairs:
                sk_acc.append(sum(p[0] for p in pairs) / len(pairs))
                en_acc.append(sum(p[1] for p in pairs) / len(pairs))
        if not sk_acc:
            return None
        return sum(sk_acc) / len(sk_acc), sum(en_acc) / len(en_acc)

    rows = []
    for theme in THEME_ORDER:
        tg = [g for g in groups_in_theme(theme) if g in en_groups]
        r = means(tg)
        if r:
            rows.append((theme, r[0], r[1]))
    overall = means(en_groups)

    def line(name, sk, en, bold=False):
        nm = name.replace("&", r"\&")
        if bold:
            nm = f"\\textbf{{{nm}}}"
        return (f"{nm} & {sk * 100:.1f} & {en * 100:.1f} & "
                f"{(en - sk) * 100:+.1f} \\\\")

    out = ["% Requires: \\usepackage{booktabs}", "\\begin{table}[t]",
           "\\centering\\small", "\\setlength{\\tabcolsep}{6pt}",
           "\\begin{tabular}{lrrr}", "\\toprule",
           "\\textbf{Reporting group} & \\textbf{SK} & \\textbf{EN} & "
           "\\textbf{$\\Delta$} \\\\", "\\midrule"]
    out += [line(t, sk, en) for t, sk, en in rows]
    out.append("\\midrule")
    out.append(line("Overall", overall[0], overall[1], bold=True))
    out += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{LANG_CAPTION}}}",
            f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(out) + "\n"


THINK_CAPTION = (
    "Effect of test-time reasoning (``thinking'' mode), Slovak, 0-shot, for the "
    "models evaluated in both modes (sorted by size). \\textbf{Std} and "
    "\\textbf{Think} are the overall macro-average; $\\Delta$ columns are "
    "thinking $-$ standard for the overall score and for the three most "
    "affected reporting groups (Instruction following, Grammar \\& morphology, "
    "Generation). Reasoning runs under an 8{,}192-token generation cap; the two "
    "smallest models (\\dag) exhaust it inside the reasoning trace without "
    "emitting a parseable answer, so their answer-extraction groups collapse "
    "while instruction and grammar still improve (negative overall $\\Delta$ in "
    "bold). Full per-task thinking scores are in "
    "Table~\\ref{tab:skbench-thinking}.")


def render_thinking_delta(std_theme, thk_theme, label) -> str:
    """Paired standard-vs-thinking overall scores and per-group deltas for the
    models evaluated in both modes, sorted by parameter count."""
    G_INSTR, G_GRAM, G_GEN = "Instruction", "Grammar & Morphology", "Generation"
    models = sorted(thk_theme, key=lambda m: MODEL_META[m].get("params") or 0)

    def d(m, key):
        a, b = thk_theme[m].get(key), std_theme[m].get(key)
        return None if a is None or b is None else (a - b) * 100

    def sign(v):
        return "--" if v is None else f"{v:+.1f}"

    out = ["% Requires: \\usepackage{booktabs}, graphicx (\\resizebox)",
           "\\begin{table}[t]",
           "\\centering\\small", "\\setlength{\\tabcolsep}{4pt}",
           "\\resizebox{\\columnwidth}{!}{%",
           "\\begin{tabular}{lrrr@{\\hskip 7pt}rrr}", "\\toprule",
           "& \\multicolumn{3}{c}{\\textbf{Overall}} & "
           "\\multicolumn{3}{c}{\\textbf{$\\Delta$ by group}} \\\\",
           "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}",
           "\\textbf{Model} & Std & Think & $\\Delta$ & Instr. & Gram. & Gen. \\\\",
           "\\midrule"]
    for m in models:
        name = MODEL_META[m]["name"].replace("&", "\\&")
        std_o = std_theme[m].get("__overall__")
        thk_o = thk_theme[m].get("__overall__")
        do = d(m, "__overall__")
        flag = "\\dag" if (do is not None and do < 0) else ""
        do_s = f"\\textbf{{{sign(do)}}}" if (do is not None and do < 0) else sign(do)
        std_s = "--" if std_o is None else f"{std_o * 100:.1f}"
        thk_s = "--" if thk_o is None else f"{thk_o * 100:.1f}"
        out.append(f"{name}{flag} & {std_s} & {thk_s} & {do_s} & "
                   f"{sign(d(m, G_INSTR))} & {sign(d(m, G_GRAM))} & "
                   f"{sign(d(m, G_GEN))} \\\\")
    out += ["\\bottomrule", "\\end{tabular}}", f"\\caption{{{THINK_CAPTION}}}",
            f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(out) + "\n"


# (base Slovak LM, Alpaca-instruction-tuned counterpart); see experimental setup
ALPACA_PAIRS = [
    ("slovak-nlp__mistral-sk-7b", "mrshu__mistral-sk-7b-alpaca-slovak-it"),
    ("slovak-nlp__Qwen3-14B-sk", "mrshu__qwen3-14b-sk-alpaca-slovak-it"),
    ("TildeAI__TildeOpen-30b-64k", "mrshu__tildeopen-30b-64k-sk-alpaca-slovak-it"),
]

ALPACA_CAPTION = (
    "Effect of Slovak Alpaca-style instruction-tuning on three base models "
    "(Slovak, 0-shot), sorted by size. \\textbf{Base} and \\textbf{+Alp.} are "
    "the overall macro-average of the base Slovak LM and its instruction-tuned "
    "counterpart; $\\Delta$ columns are tuned $-$ base for the overall score "
    "and the four most affected reporting groups (Knowledge \\& reasoning, "
    "Classification, Instruction following, Grammar \\& morphology). The net "
    "effect rises with base capability---negative for the weakest base "
    "(in bold), flat for the mid one, clearly positive for the strongest---"
    "while instruction following (the tuned skill) improves or holds for all "
    "three and the other groups swing in both directions.")


def render_alpaca_delta(theme_data, label) -> str:
    """Base-vs-Alpaca-instruction-tuned overall scores and per-group deltas
    for the three Slovak base/tuned pairs, sorted by parameter count."""
    G_KNOW, G_CLS = "Knowledge & Reasoning", "Classification"
    G_INSTR, G_GRAM = "Instruction", "Grammar & Morphology"
    pairs = sorted((p for p in ALPACA_PAIRS if p[0] in theme_data
                    and p[1] in theme_data),
                   key=lambda p: MODEL_META[p[0]].get("params") or 0)

    def d(base, alp, key):
        a, b = theme_data[alp].get(key), theme_data[base].get(key)
        return None if a is None or b is None else (a - b) * 100

    def sign(v):
        return "--" if v is None else f"{v:+.1f}"

    out = ["% Requires: \\usepackage{booktabs}, graphicx (\\resizebox)",
           "\\begin{table}[t]",
           "\\centering\\small", "\\setlength{\\tabcolsep}{4pt}",
           "\\resizebox{\\columnwidth}{!}{%",
           "\\begin{tabular}{lrrr@{\\hskip 7pt}rrrr}", "\\toprule",
           "& \\multicolumn{3}{c}{\\textbf{Overall}} & "
           "\\multicolumn{4}{c}{\\textbf{$\\Delta$ by group}} \\\\",
           "\\cmidrule(lr){2-4}\\cmidrule(l){5-8}",
           "\\textbf{Lineage} & Base & +Alp. & $\\Delta$ & "
           "Know. & Class. & Instr. & Gram. \\\\", "\\midrule"]
    for base, alp in pairs:
        name = MODEL_META[alp]["name"].replace(" (Alpaca)", "").replace("&", "\\&")
        b_o, a_o = theme_data[base].get("__overall__"), theme_data[alp].get("__overall__")
        do = d(base, alp, "__overall__")
        do_s = f"\\textbf{{{sign(do)}}}" if (do is not None and do < 0) else sign(do)
        b_s = "--" if b_o is None else f"{b_o * 100:.1f}"
        a_s = "--" if a_o is None else f"{a_o * 100:.1f}"
        out.append(f"{name} & {b_s} & {a_s} & {do_s} & "
                   f"{sign(d(base, alp, G_KNOW))} & {sign(d(base, alp, G_CLS))} & "
                   f"{sign(d(base, alp, G_INSTR))} & {sign(d(base, alp, G_GRAM))} \\\\")
    out += ["\\bottomrule", "\\end{tabular}}", f"\\caption{{{ALPACA_CAPTION}}}",
            f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(out) + "\n"


ONERULER_LENS = [4096, 8192, 16384]

ONERULER_CAPTION = (
    "OneRuler-SK by context length: mean retrieval accuracy ($\\times100$, "
    "0-shot) over the RULER-style subtasks at 4{,}096 / 8{,}192 / 16{,}384 "
    "tokens, macro-averaged across all evaluated models, for the matched "
    "Slovak and English haystacks. Because SK and EN share the same model and "
    "the same length, $\\Delta=\\text{SK}-\\text{EN}$ isolates the language "
    "effect from the context-window limit. The gap does not widen with "
    "length---it closes and slightly reverses by 16k---so up to 16k there is "
    "no Slovak-specific long-context penalty. 16k is the longest length "
    "evaluated.")


def oneruler_length_means(family: str = "outputs"):
    """Macro-mean (over models) of OneRuler accuracy per context length, for
    the Slovak and English haystacks. Returns {lang: {length: mean}}."""
    base = Path("final_results") / family / "aggregated"
    per_lang = {"slovak": {L: [] for L in ONERULER_LENS},
                "english": {L: [] for L in ONERULER_LENS}}
    for fpath in sorted(base.glob("*.json")):
        if fpath.stem not in MODEL_META:
            continue
        res = json.loads(fpath.read_text())["results"]
        for lang in ("slovak", "english"):
            for L in ONERULER_LENS:
                vals = [v["acc"] for k, v in res.items()
                        if k.startswith(f"oneruler_sk.oneruler_{lang}:")
                        and k.endswith(f"_{L}|0") and "acc" in v]
                if vals:
                    per_lang[lang][L].append(sum(vals) / len(vals))
    return {lang: {L: (sum(xs) / len(xs) if xs else None)
                   for L, xs in lens.items()}
            for lang, lens in per_lang.items()}


def render_oneruler_length(label) -> str:
    """Compact OneRuler SK-vs-EN accuracy table indexed by context length."""
    m = oneruler_length_means()
    out = ["% Requires: \\usepackage{booktabs}", "\\begin{table}[t]",
           "\\centering\\small", "\\setlength{\\tabcolsep}{6pt}",
           "\\begin{tabular}{lrrr}", "\\toprule",
           "\\textbf{Context} & \\textbf{SK} & \\textbf{EN} & "
           "\\textbf{$\\Delta$} \\\\", "\\midrule"]
    for L in ONERULER_LENS:
        sk, en = m["slovak"][L], m["english"][L]
        g = (sk - en) * 100 if (sk is not None and en is not None) else None
        sk_s = "--" if sk is None else f"{sk * 100:.1f}"
        en_s = "--" if en is None else f"{en * 100:.1f}"
        g_s = "--" if g is None else f"{g:+.1f}"
        out.append(f"{L:,} & {sk_s} & {en_s} & {g_s} \\\\".replace(",", "{,}"))
    out += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{ONERULER_CAPTION}}}",
            f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate LaTeX result tables.")
    ap.add_argument("--output-dir", required=True, help="Directory to write .tex files into.")
    ap.add_argument("--which", default="all",
                    choices=["all", "comprehensive", "thinking",
                             "summary", "english", "alpaca", "oneruler"])
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    label_of = {g: lab for g, lab, _ in TASK_GROUPS}
    all_groups = [g for g, _, _ in TASK_GROUPS]

    outputs = load_family("outputs")
    written: list[str] = []

    if args.which in ("all", "comprehensive"):
        tiers = order_models(outputs, all_groups)
        tex = render_table(outputs, tiers, all_groups,
                           [label_of[g] for g in all_groups],
                           "sk-bench results across all tasks. " + CAPTION_NOTE,
                           "tab:skbench-comprehensive")
        (out / "table_comprehensive.tex").write_text(tex)
        written.append("table_comprehensive.tex")

    if args.which in ("all", "summary"):
        tiers = order_models(outputs, all_groups)
        theme_data, theme_std = build_theme_data(outputs, load_family_prompts("outputs"))
        # body: top-3 per tier, upright full-width, ranked over all 55
        body_tiers = top_n_per_tier(theme_data, tiers, 3)
        (out / "table_summary.tex").write_text(
            render_leaderboard(theme_data, theme_std, body_tiers, "tab:results-main",
                               BODY_CAPTION, resize=True))
        written.append("table_summary.tex")
        # appendix: the full leaderboard (all 55); tall+narrow, so a full-page
        # upright longtable rather than a rotated float
        (out / "table_summary_full.tex").write_text(
            render_leaderboard(theme_data, theme_std, tiers, "tab:results-full",
                               FULL_CAPTION, longtable=True))
        written.append("table_summary_full.tex")

    if args.which in ("all", "alpaca"):
        theme_data, _ = build_theme_data(outputs, load_family_prompts("outputs"))
        (out / "table_alpaca_delta.tex").write_text(
            render_alpaca_delta(theme_data, "tab:alpaca-delta"))
        written.append("table_alpaca_delta.tex")

    if args.which in ("all", "oneruler"):
        (out / "table_oneruler_length.tex").write_text(
            render_oneruler_length("tab:oneruler-length"))
        written.append("table_oneruler_length.tex")

    if args.which in ("all", "english"):
        outputs_en = load_family("outputs", lang="EN")
        en_groups = [g for g, _, _ in TASK_GROUPS if g not in SLOVAK_ONLY]
        # SK vs EN per-group comparison (uses the SK aggregates already loaded)
        (out / "table_lang_comparison.tex").write_text(
            render_lang_comparison(outputs, outputs_en, "tab:lang-comparison"))
        written.append("table_lang_comparison.tex")
        # full English per-task table (parallel to the SK comprehensive one)
        tiers = order_models(outputs_en, en_groups)
        tex = render_table(outputs_en, tiers, en_groups,
                           [label_of[g] for g in en_groups],
                           "sk-bench results across all tasks. " + EN_CAPTION_NOTE,
                           "tab:skbench-comprehensive-en")
        (out / "table_comprehensive_en.tex").write_text(tex)
        written.append("table_comprehensive_en.tex")

    if args.which in ("all", "thinking"):
        thinking = load_family("thinking")
        tiers = order_models(thinking, all_groups)
        tex = render_table(thinking, tiers, all_groups, [label_of[g] for g in all_groups],
                           "sk-bench results for thinking-mode models. " + CAPTION_NOTE,
                           "tab:skbench-thinking")
        (out / "table_thinking.tex").write_text(tex)
        written.append("table_thinking.tex")
        # paired standard-vs-thinking comparison for the body ablation
        std_theme, _ = build_theme_data(outputs, load_family_prompts("outputs"))
        thk_theme, _ = build_theme_data(thinking, load_family_prompts("thinking"))
        thk_theme = {m: thk_theme[m] for m in thk_theme if m in std_theme}
        (out / "table_thinking_delta.tex").write_text(
            render_thinking_delta(std_theme, thk_theme, "tab:thinking-delta"))
        written.append("table_thinking_delta.tex")

    print(f"Wrote {len(written)} table(s) to {out}:")
    for w in written:
        print(f"  {w}")


if __name__ == "__main__":
    main()
