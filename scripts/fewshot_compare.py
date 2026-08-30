#!/usr/bin/env python3
"""
Zero-shot vs few-shot (3-shot) comparison, per reporting group, for Slovak
and English prompts, computed with the SAME aggregation as the headline
leaderboard: per-group score = mean over the group's tasks of the task's
canonical primary metric; overall = mean over the six reporting groups.

For a fair 0-vs-3 comparison we restrict, per model and group, to the tasks
that have BOTH a 0-shot and a 3-shot value in that language, then
macro-average across the models that expose few-shot results.

Reuses the group/metric definitions from generate_latex_tables.py.
"""
from __future__ import annotations
import glob, json, os, sys
from statistics import mean

CANONICAL_METRIC = {
    **{g: "acc" for g in (
        "arc", "mmlu", "globalmmlu_lite", "truthful_qa", "scio", "financial_exams",
        "driving_test", "slovakcopa", "skj1", "slovak_trivia", "belebele", "chiby",
        "e_tests_mqa", "slovak_tests_mqa", "nli_sk", "sklep_nli", "sklep_rte",
        "senti_sk_3class", "senti_sk_binary", "sklep_sentiment_analysis",
        "movie_review", "toxic_sk", "sklep_hate_speech", "analogies_sk", "slohry",
        "oneruler")},
    **{g: "qem" for g in ("multi_wiki_qa", "e_tests_qa", "slovak_tests_qa", "sklep_qa")},
    "slovaksum": "rougeL", "smesum": "rougeL",
    "ifeval_sk": "prompt_level_strict_acc",
}
TASK_GROUPS = [
    ("arc","Knowledge & Reasoning"),("mmlu","Knowledge & Reasoning"),
    ("globalmmlu_lite","Knowledge & Reasoning"),("truthful_qa","Knowledge & Reasoning"),
    ("scio","Knowledge & Reasoning"),("financial_exams","Knowledge & Reasoning"),
    ("driving_test","Knowledge & Reasoning"),("slovakcopa","Knowledge & Reasoning"),
    ("skj1","Knowledge & Reasoning"),("slovak_trivia","Knowledge & Reasoning"),
    ("belebele","Reading & QA"),("chiby","Reading & QA"),("multi_wiki_qa","Reading & QA"),
    ("e_tests_mqa","Reading & QA"),("e_tests_qa","Reading & QA"),
    ("slovak_tests_mqa","Reading & QA"),("slovak_tests_qa","Reading & QA"),
    ("oneruler","Reading & QA"),("sklep_qa","Reading & QA"),
    ("nli_sk","Classification"),("sklep_nli","Classification"),("sklep_rte","Classification"),
    ("senti_sk_3class","Classification"),("senti_sk_binary","Classification"),
    ("sklep_sentiment_analysis","Classification"),("movie_review","Classification"),
    ("toxic_sk","Classification"),("sklep_hate_speech","Classification"),
    ("slovaksum","Generation"),("smesum","Generation"),
    ("ifeval_sk","Instruction"),
    ("analogies_sk","Grammar & Morphology"),("slohry","Grammar & Morphology"),
]
THEMES = ["Knowledge & Reasoning","Reading & QA","Classification",
          "Grammar & Morphology"]
AGG = "final_results/outputs/aggregated"


def task_val(results, key, lang, shot):
    suffixes = ["EN_AVERAGE"] if lang == "EN" else ["SK_AVERAGE", "AVERAGE", "UNK_AVERAGE"]
    for suf in suffixes:
        e = results.get(f"{key}_{suf}|{shot}")
        if e is not None:
            return e.get(CANONICAL_METRIC[key])
    return None


# Row labels for the LaTeX table, matching the headline leaderboard groups.
DISPLAY = {
    "Knowledge & Reasoning": "Knowledge \\& reasoning",
    "Reading & QA": "Reading \\& QA",
    "Classification": "Classification",
    "Generation": "Generation",
    "Instruction": "Instruction following",
    "Grammar & Morphology": "Grammar \\& morphology",
}


def model_themes(results, lang):
    """Per-theme {theme: (v0, v3_or_None)} for one model.

    For a theme with at least one task carrying BOTH a 0- and a 3-shot value,
    v3 is the mean over those paired tasks and v0 the paired 0-shot over the
    same tasks (fair Delta). For a theme with no 3-shot task (Generation,
    Instruction here), v3 is None and v0 is the mean over its 0-shot tasks."""
    out = {}
    for th in THEMES:
        keys = [k for k, t in TASK_GROUPS if t == th]
        p0, p3, z0 = [], [], []
        for k in keys:
            a = task_val(results, k, lang, 0)
            b = task_val(results, k, lang, 3)
            if a is not None:
                z0.append(a)
            if a is not None and b is not None:
                p0.append(a); p3.append(b)
        if p3:
            out[th] = (mean(p0), mean(p3))
        elif z0:
            out[th] = (mean(z0), None)
    return out


def collect(files, lang):
    """Macro-average over the models that expose few-shot results. Returns
    (acc0, acc3, ov0, ov3, n): per-theme 0/3-shot value lists, per-model
    paired-group overalls, and the few-shot model count."""
    acc0 = {th: [] for th in THEMES}
    acc3 = {th: [] for th in THEMES}
    ov0, ov3, n = [], [], 0
    for f in files:
        r = json.load(open(f))["results"]
        mt = model_themes(r, lang)
        paired = {th: v for th, v in mt.items() if v[1] is not None}
        if not paired:
            continue  # model has no few-shot results
        n += 1
        for th, (v0, v3) in mt.items():
            if v0 is not None:
                acc0[th].append(v0)
            if v3 is not None:
                acc3[th].append(v3)
        ov0.append(mean(v0 for v0, _ in paired.values()))
        ov3.append(mean(v3 for _, v3 in paired.values()))
    return acc0, acc3, ov0, ov3, n


def fmt(xs):
    return f"{100 * mean(xs):.1f}" if xs else "--"


CAPTION = (
    "Zero-shot vs.\\ 3-shot, macro-averaged over the @N@ models with few-shot "
    "results, for the four reporting groups of the headline leaderboard "
    "(Table~\\ref{tab:results-main}) that have a few-shot counterpart. "
    "Generation (summarisation) and instruction following were run zero-shot "
    "only and are therefore omitted. The \\textbf{Overall} row is the "
    "macro-average over these four groups. Few-shot helps most on "
    "classification and reading \\& QA, slightly hurts knowledge \\& "
    "reasoning, and moves the overall little; the pattern is the same under "
    "Slovak and English prompts.")


def emit_tex(data, path):
    sk, en = data["SK"], data["EN"]
    caption = CAPTION.replace("@N@", str(sk[4]))
    L = ["% Generated by sk-bench/scripts/fewshot_compare.py",
         "\\begin{table}[t]", "\\centering\\small",
         "\\setlength{\\tabcolsep}{5pt}",
         "\\begin{tabular}{lrrrr}", "\\toprule",
         " & \\multicolumn{2}{c}{\\textbf{Slovak}} & "
         "\\multicolumn{2}{c}{\\textbf{English}}\\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}",
         "\\textbf{Reporting group} & 0-shot & 3-shot & 0-shot & 3-shot\\\\",
         "\\midrule"]
    for th in THEMES:
        L.append(f"{DISPLAY[th]:<22} & {fmt(sk[0][th])} & {fmt(sk[1][th])} "
                 f"& {fmt(en[0][th])} & {fmt(en[1][th])}\\\\")
    L += ["\\midrule",
          f"Overall & {fmt(sk[2])} & {fmt(sk[3])} & {fmt(en[2])} & {fmt(en[3])}\\\\",
          "\\bottomrule", "\\end{tabular}",
          f"\\caption{{{caption}}}", "\\label{tab:fewshot}", "\\end{table}"]
    open(path, "w").write("\n".join(L) + "\n")
    print(f"\nWrote {path}")


def main():
    files = sorted(glob.glob(f"{AGG}/*.json"))
    data = {lang: collect(files, lang) for lang in ("SK", "EN")}
    for lang in ("SK", "EN"):
        acc0, acc3, ov0, ov3, n = data[lang]
        print(f"\n### {lang} prompts  (models with few-shot: n={n})")
        print(f"    {'group':24s} {'0-shot':>7} {'3-shot':>7}")
        for th in THEMES:
            print(f"    {th:24s} {fmt(acc0[th]):>7} {fmt(acc3[th]):>7}")
        print(f"    {'OVERALL (paired)':24s} {fmt(ov0):>7} {fmt(ov3):>7}")
    if len(sys.argv) > 1:
        emit_tex(data, sys.argv[1])


if __name__ == "__main__":
    main()
