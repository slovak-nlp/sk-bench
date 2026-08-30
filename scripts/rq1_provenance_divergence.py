#!/usr/bin/env python3
"""
RQ1 provenance-divergence analysis for sk-bench.

Quantifies how much native, machine-translated, and LLM-generated evaluation
data for the *same skill* disagree about Slovak LLM rankings. For each matched
contrast it reports the Spearman rank correlation between the two leaderboards
(a) overall and (b) *within* each parameter-size band, plus the top-10 rank
overlap. The within-band view is the decision-relevant one: a single rho over a
350M-to-1T model pool is dominated by between-tier ordering and can mask large
within-tier reordering.

Reads the per-model aggregated JSONs written by generate_final_results.py
(<output_dir>/aggregated/*.json). Uses zero-shot (|0) Slovak (SK) task-family
averages, matching the released leaderboard.

Usage:
    python scripts/rq1_provenance_divergence.py \
        --agg final_results/outputs/aggregated [--latex]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import defaultdict
from math import sqrt

# Matched contrasts: (native/human key, metric, translated/generated key, metric)
CONTRASTS = {
    "NLI": ("nli_sk_SK_AVERAGE|0", "acc", "sklep_nli_SK_AVERAGE|0", "acc"),
    "MCQ": ("e_tests_mqa_SK_AVERAGE|0", "acc", "mmlu_SK_AVERAGE|0", "acc"),
    "QA": ("sklep_qa_SK_AVERAGE|0", "qem", "multi_wiki_qa_SK_AVERAGE|0", "qem"),
}
CONTRAST_DESC = {
    "NLI": "native NLI-SK-annotated vs. translated skLEP-NLI",
    "MCQ": "native e-testy-sk vs. translated Okapi-MMLU-SK",
    "QA": "human SK-QuAD vs. LLM-generated MultiWikiQA-SK",
}

# Parameter sizes (B) by filename substring; 9999 = proprietary (undisclosed).
# Order matters: more specific substrings first.
SIZES = [
    ("granite-4.0-h-350m", .35), ("granite-4.0-350m", .35), ("Qwen3-0.6B", .6),
    ("Qwen3.5-0.8B", .8), ("gemma-3-1b", 1), ("granite-4.0-h-1b", 1),
    ("granite-4.0-1b", 1), ("Qwen3-1.7B", 1.7), ("EuroLLM-1.7B", 1.7),
    ("granite-3.2-2b", 2), ("Qwen3.5-2B", 2), ("Llama-3.2-3B", 3),
    ("tiny-aya-global", 3.35), ("tiny-aya-water", 3.35), ("Teuken-7B", 7),
    ("mistral-sk-7b-alpaca", 7), ("mistral-sk-7b", 7), ("Llama-3.1-8B", 8),
    ("Ministral-3-8B-Instruct", 8), ("Ministral-3-8B-Reasoning", 8),
    ("aya-expanse-8b", 8), ("EuroLLM-9B", 9), ("Qwen3.5-9B", 9), ("Bielik-11B", 11),
    ("Ministral-3-14B-Instruct", 14), ("Ministral-3-14B-Reasoning", 14),
    ("qwen3-14b-sk-alpaca", 14), ("Qwen3-14B-sk", 14), ("Qwen3-14B", 14),
    ("gpt-oss-20b", 20), ("EuroLLM-22B", 22), ("Qwen3.5-27B", 27),
    ("Qwen3.6-27B", 27), ("tildeopen-sk-30b-alpaca", 30), ("TildeOpen-30b-64k", 30),
    ("TildeOpen-30b", 30), ("gemma-4-31B", 31), ("aya-expanse-32b", 32),
    ("granite-4.0-h-small", 32), ("Qwen3.6-35B-A3B", 35), ("Llama-3.3-70B", 70),
    ("Apertus-70B", 70), ("command-a", 111), ("gpt-oss-120b", 120),
    ("Qwen3.5-397B", 397), ("Mistral-Large-3-675B", 675), ("MiniMax-M2.7", 230),
    ("GLM-5.1", 754), ("Kimi-K2.6", 1100),
    ("gemini", 9999), ("gpt-5", 9999), ("claude", 9999),
]
BANDS = ["<2B", "2-9B", "9-32B", "32-200B", ">200B", "Proprietary"]


def size_of(fn: str):
    base = os.path.basename(fn).lower()
    for sub, s in SIZES:
        if sub.lower() in base:
            return s
    return None


def band(s):
    if s is None:
        return "??"
    if s == 9999:
        return "Proprietary"
    if s < 2:
        return "<2B"
    if s < 9:
        return "2-9B"
    if s < 32:
        return "9-32B"
    if s <= 200:
        return "32-200B"
    return ">200B"


def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(a, b):
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = sqrt(sum((x - ma) ** 2 for x in a))
    db = sqrt(sum((x - mb) ** 2 for x in b))
    return num / (da * db) if da and db else float("nan")


def spearman(a, b):
    return pearson(rankdata(a), rankdata(b))


def boot_ci(pairs, B=2000, seed=1234):
    """Percentile bootstrap 95% CI for Spearman rho over (x, y) pairs."""
    if len(pairs) < 3:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(pairs)
    rs = []
    for _ in range(B):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        xs = [p[0] for p in sample]
        ys = [p[1] for p in sample]
        r = spearman(xs, ys)
        if r == r:  # not nan
            rs.append(r)
    rs.sort()
    lo = rs[int(0.025 * len(rs))]
    hi = rs[int(0.975 * len(rs)) - 1]
    return (lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default="final_results/outputs/aggregated")
    ap.add_argument("--latex", action="store_true", help="emit a LaTeX table")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.agg}/*.json"))
    models = []
    for f in files:
        d = json.load(open(f))
        s = size_of(f)
        models.append((os.path.basename(f)[:-5], s, band(s), d["results"]))
    print(f"Loaded {len(models)} models")
    unmapped = [m[0] for m in models if m[1] is None]
    if unmapped:
        print("WARNING unmapped size:", unmapped)
    print()

    rows = {}
    for name, (kn, mn, kt, mt) in CONTRASTS.items():
        pairs = []
        for nm, sz, bd, r in models:
            vn, vt = r.get(kn), r.get(kt)
            if vn is None or vt is None:
                continue
            x, y = vn.get(mn), vt.get(mt)
            if x is None or y is None:
                continue
            pairs.append((nm, bd, x, y))
        nat = [(x, y) for _, _, x, y in pairs]
        rho = spearman([x for x, _ in nat], [y for _, y in nat])
        lo, hi = boot_ci(nat)
        print(f"### {name}: {CONTRAST_DESC[name]}  | n={len(pairs)}")
        print(f"    OVERALL  rho={rho:.3f}  95% CI [{lo:.2f},{hi:.2f}]")
        per_band = {}
        bands = defaultdict(list)
        for _, bd, x, y in pairs:
            bands[bd].append((x, y))
        for bd in BANDS:
            v = bands.get(bd, [])
            if len(v) >= 4:
                r_ = spearman([a for a, _ in v], [b for _, b in v])
                blo, bhi = boot_ci(v)
                per_band[bd] = (r_, len(v), blo, bhi)
                print(f"    {bd:12s} n={len(v):2d}  rho={r_:.3f}  CI [{blo:.2f},{bhi:.2f}]")
            elif v:
                per_band[bd] = (None, len(v), None, None)
                print(f"    {bd:12s} n={len(v):2d}  (too few)")
        k = args.topk
        top_n = set(nm for nm, _, x, _ in sorted(pairs, key=lambda p: -p[2])[:k])
        top_t = set(nm for nm, _, _, y in sorted(pairs, key=lambda p: -p[3])[:k])
        overlap = len(top_n & top_t)
        rows[name] = (rho, lo, hi, per_band, overlap, k, len(pairs))
        print(f"    top-{k} overlap: {overlap}/{k}\n")

    if args.latex:
        print("\n% ---- LaTeX (paste into latex/tables/) ----")
        print(r"\begin{tabular}{l r" + "r" * len(BANDS) + r" c}")
        print(r"\toprule")
        hdr = " & ".join([r"\textbf{Contrast}", r"\textbf{Overall}"]
                         + [f"\\textbf{{{b}}}" for b in BANDS] + [f"\\textbf{{top-{args.topk}}}"])
        print(hdr + r" \\")
        print(r"\midrule")
        for name in CONTRASTS:
            rho, lo, hi, per_band, overlap, k, n = rows[name]
            cells = [name, f"{rho:.2f}"]
            for b in BANDS:
                if b in per_band and per_band[b][0] is not None:
                    cells.append(f"{per_band[b][0]:.2f}")
                else:
                    cells.append("--")
            cells.append(f"{overlap}/{k}")
            print(" & ".join(cells) + r" \\")
        print(r"\bottomrule")
        print(r"\end{tabular}")


if __name__ == "__main__":
    main()
