"""Write the leaderboard snapshot that site/build.py renders.

Usage: python3 site/snapshot_leaderboard.py <dir with aggregated *.json>

The aggregated results live in the private slovak-nlp/sk-bench-results repo
(outputs/aggregated). The scores are computed with the same functions as the
paper tables in scripts/generate_latex_tables.py: Slovak prompts, 0-shot, the
mean of each reporting group, and Overall as the mean of the six groups.
Each model also carries its per-task scores with Slovak and English prompts
(None where a task has no English prompt) and its parameter count.
Writes site/data/leaderboard.json, which is committed so the page builds
without access to the private repo.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "scripts"))
import generate_latex_tables as g  # noqa: E402


def lang_comparison(data_sk: dict, data_en: dict) -> list[dict]:
    """SK vs EN prompt means per reporting group and overall, macro-averaged
    across models, exactly as render_lang_comparison in the paper script."""
    both = [t for t, _, _ in g.TASK_GROUPS if t not in g.SLOVAK_ONLY]

    def means(tasks: list[str]) -> tuple[float, float] | None:
        sk, en = [], []
        for m in data_sk:
            pairs = [(data_sk[m][t], data_en[m][t]) for t in tasks
                     if data_sk[m].get(t) is not None and data_en[m].get(t) is not None]
            if pairs:
                sk.append(sum(a for a, _ in pairs) / len(pairs))
                en.append(sum(b for _, b in pairs) / len(pairs))
        return (sum(sk) / len(sk), sum(en) / len(en)) if sk else None

    rows = []
    for th in g.THEME_ORDER + ["__overall__"]:
        tasks = both if th == "__overall__" else [t for t in g.groups_in_theme(th) if t in both]
        r = means(tasks)
        if r:
            rows.append({"group": th, "sk": round(r[0] * 100, 1), "en": round(r[1] * 100, 1),
                         "delta": round((r[1] - r[0]) * 100, 1)})
    return rows


def main() -> None:
    src = Path(sys.argv[1])
    data, data_en = {}, {}
    for f in sorted(src.glob("*.json")):
        results = json.loads(f.read_text())["results"]
        data[f.stem] = {grp: g.lang_value(results, grp, "SK") for grp, _, _ in g.TASK_GROUPS}
        data_en[f.stem] = {grp: g.lang_value(results, grp, "EN") for grp, _, _ in g.TASK_GROUPS}
    unknown = [k for k in data if k not in g.MODEL_META]
    if unknown:
        sys.exit(f"models missing from MODEL_META: {unknown}")
    means, _ = g.build_theme_data(data)
    pct = lambda v: None if v is None else round(v * 100, 1)  # noqa: E731
    tasks = [grp for grp, _, _ in g.TASK_GROUPS]
    models = [
        {
            "key": k,
            "name": g.MODEL_META[k]["name"],
            "open": g.MODEL_META[k]["open"],
            "params": g.MODEL_META[k]["params"],
            "overall": pct(means[k]["__overall__"]),
            "groups": [pct(means[k][th]) for th in g.THEME_ORDER],
            "tasks": [pct(data[k][t]) for t in tasks],
            "tasks_en": [None if t in g.SLOVAK_ONLY else pct(data_en[k][t]) for t in tasks],
        }
        for k in data
    ]
    models.sort(key=lambda m: -m["overall"])
    lang = lang_comparison(data, data_en)
    out = ROOT / "data" / "leaderboard.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "groups": g.THEME_ORDER,
        "tasks": [{"key": t, "label": lbl, "group": th, "metric": g.metric_for(t)}
                  for t, lbl, th in g.TASK_GROUPS],
        "lang": lang,
        "models": models},
                              ensure_ascii=False, indent=1) + "\n")
    print(f"{len(models)} models ->", out)


if __name__ == "__main__":
    main()
