"""Render the 1200x630 social share cards (og:image) for the project page.

The cards are committed as PNGs in site/og/ so build.py stays stdlib-only.
Re-run this only when the headline numbers change:

    python3 site/og_card.py

Needs Google Chrome (headless) and rsvg-convert (brew install librsvg). Fonts are the same system serif stack the
page uses, so render on macOS to get Charter.
"""

import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent / "og"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CARDS = {
    "sk": dict(
        h1="Ako dobre rozumie umelá inteligencia slovenčine?",
        sub="55 modelov na 30 testoch v slovenčine",
        bars=[("Najlepší komerčný model", "Gemini 3.1 Pro", 72.6, "72,6"),
              ("Najlepší model s otvorenými váhami", "gpt-oss 120B", 60.0, "60,0")],
        unit="bodu zo 100",
    ),
    "en": dict(
        h1="How well does AI understand Slovak?",
        sub="55 models on 30 Slovak test sets",
        bars=[("Best commercial model", "Gemini 3.1 Pro", 72.6, "72.6"),
              ("Best open model", "gpt-oss 120B", 60.0, "60.0")],
        unit="points out of 100",
    ),
}

CSS = """
* { box-sizing: border-box; margin: 0; }
html, body { width: 1200px; height: 630px; }
body { background: #FAF9F6; color: #1B1E24; padding: 64px 80px 56px;
  font-family: Charter, "Bitstream Charter", Georgia, serif;
  display: flex; flex-direction: column; -webkit-font-smoothing: antialiased; }
.sans { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; }
.top { display: flex; justify-content: space-between; align-items: center; font-size: 26px; }
.brand { font-weight: 700; letter-spacing: -.01em; }
.pill { font-size: 22px; font-weight: 600; color: #1F3F68; border: 2px solid #1F3F68;
  border-radius: 999px; padding: 4px 18px; }
h1 { font-size: 70px; line-height: 1.08; letter-spacing: -.015em; font-weight: 700;
  margin-top: 44px; max-width: 980px; }
.sub { font-size: 28px; color: #5A606B; margin-top: 18px; }
.bars { margin-top: auto; display: grid; gap: 18px; }
.row { display: grid; grid-template-columns: 1fr 96px; align-items: end; column-gap: 20px; }
.lab { font-size: 22px; color: #5A606B; margin-bottom: 8px; }
.lab b { color: #1B1E24; font-weight: 600; }
.track { grid-column: 1; height: 22px; background: #ECEAE3; border-radius: 6px; overflow: hidden; }
.fill { height: 100%; border-radius: 6px; }
.val { grid-column: 2; grid-row: 1 / span 2; font-size: 40px; font-weight: 700; text-align: right;
  align-self: end; line-height: 1; }
.foot { display: flex; justify-content: space-between; margin-top: 26px; font-size: 20px; color: #5A606B; }
"""


def html(c: dict, lang: str) -> str:
    rows = ""
    for (label, model, v, vs), color in zip(c["bars"], ("#3B4049", "#2D6CB5")):
        rows += (f'<div class="row"><div class="lab sans">{label}: <b>{model}</b></div>'
                 f'<div class="val sans" style="color:{color}">{vs}</div>'
                 f'<div class="track"><div class="fill" style="width:{v}%;background:{color}"></div></div></div>')
    return f"""<!doctype html><html lang="{lang}"><meta charset="utf-8"><style>{CSS}</style>
<body><div class="top sans"><span class="brand">sk-bench</span><span class="pill">EMNLP 2026</span></div>
<h1>{c["h1"]}</h1><p class="sub sans">{c["sub"]}</p>
<div class="bars">{rows}</div>
<div class="foot sans"><span>{c["unit"]}</span><span>slovak-nlp.github.io/sk-bench</span></div></body></html>"""


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for lang, c in CARDS.items():
        src, png = OUT / f"_card-{lang}.html", OUT / f"og-{lang}.png"
        src.write_text(html(c, lang))
        png.unlink(missing_ok=True)
        r = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                            "--force-device-scale-factor=1", "--window-size=1200,630",
                            f"--screenshot={png}", src.as_uri()], capture_output=True, text=True)
        src.unlink()
        if not png.exists():
            raise SystemExit(f"Chrome did not write {png}:\n{r.stderr[-2000:]}")
        print("wrote", png)
    # 180x180 touch icon for iOS home screens and some link previews
    subprocess.run(["rsvg-convert", "-w", "180", "-h", "180", "-o", str(OUT / "apple-touch-icon.png"),
                    str(OUT / "favicon.svg")], check=True)
    print("wrote", OUT / "apple-touch-icon.png")


if __name__ == "__main__":
    main()
