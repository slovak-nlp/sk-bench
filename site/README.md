# sk-bench project page

Source for https://slovak-nlp.github.io/sk-bench/ (Slovak) and
https://slovak-nlp.github.io/sk-bench/en/ (English). The page is the
landing page for the paper and its press release. The top part is written
for the general public, the leaderboard, dataset tables and method further
down are for researchers.

## Files

- `build.py` generates the whole page. All text (Slovak and English), the
  authors, the quiz questions, the dataset descriptions and the CSS and
  JavaScript live here. It uses the Python standard library only
- `data/leaderboard.json` is the score snapshot the page renders: 55 models
  with overall, group and per-task scores (Slovak and English prompts)
- `snapshot_leaderboard.py` regenerates `data/leaderboard.json` from the
  aggregated results in the private `slovak-nlp/sk-bench-results` repo,
  using the same functions as the paper tables
- `og_card.py` renders the social share cards `og/og-sk.png` and
  `og/og-en.png` (1200x630, used for `og:image` and `twitter:image`) and
  `og/apple-touch-icon.png` from `og/favicon.svg`. The PNGs are committed,
  re-run it on macOS with Chrome and `rsvg-convert` when the headline
  numbers change
- `dist/` is the build output. It is not committed on `main`, it is what
  gets copied to the `gh-pages` branch

## Build and preview

```bash
python3 site/build.py
python3 -m http.server 8765 -d site/dist
```

The page uses system fonts and makes no third-party requests (GDPR). Keep
it that way: no Google Fonts, no CDNs, no analytics.

## Update the scores

```bash
python3 site/snapshot_leaderboard.py <sk-bench-results>/outputs/aggregated
python3 site/build.py
```

Leave out the `thinking_*` files from the aggregated folder, the page shows
the standard runs. The page says the data is as of the paper's release, so
change that text in `build.py` if the snapshot moves on.

## Deploy

GitHub Pages serves the `gh-pages` branch as is.

```bash
git fetch origin gh-pages
git worktree add -B gh-pages ../sk-bench-ghp origin/gh-pages
cp -R site/dist/. ../sk-bench-ghp/
cd ../sk-bench-ghp && git add -A && git commit -m "feat: ..." && git push origin gh-pages
cd - && git worktree remove ../sk-bench-ghp
```

## Still to do

- **Per-question quiz results.** After a reader answers a quiz question, the
  page shows a dashed "Doplniť" / "To fill in" note. Fill in how many of the
  55 models answered that item correctly in all three Slovak phrasings by
  setting `models=(right, total)` on each entry of `QUESTIONS` in
  `build.py`. This needs the lighteval `--save-details` outputs for all 55
  models, which are not in this repo
- **Dataset count.** The paper says 11 new datasets, the press release says
  12. The page follows the paper
- **Data link.** The "Kód a dáta" button points to the sk-bench Hugging Face
  collection, but not every dataset is public yet
- **Maturita figures** in finding 1 are derived from the per-test scores,
  not copied from a paper table. See `MATURITA` in `build.py`
