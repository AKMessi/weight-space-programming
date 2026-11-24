"""Assemble the self-contained deployable static site into _site/
for GitHub Pages, Vercel, Netlify, or Cloudflare Pages without duplicating PDFs/PNGs in git."""
from __future__ import annotations

import pathlib
import shutil


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "_site"


def build() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Copy site source files
    for p in (ROOT / "site").iterdir():
        dst = OUT / p.name
        if p.is_dir():
            shutil.copytree(p, dst)
        else:
            shutil.copy2(p, dst)

    # 2. Copy both verified PDFs and all 8 300-DPI paper figures
    papers_out = OUT / "papers"
    figs_out = papers_out / "figures"
    figs_out.mkdir(parents=True, exist_ok=True)

    for pdf in [
        "Paper1_Greedy_Perturbation_Illusion_and_Zero_Gradient_WSP_Kakad2026.pdf",
        "Paper2_Quantization_Cliff_and_BA_WSC_Kakad2026.pdf",
    ]:
        shutil.copy2(ROOT / "papers" / pdf, papers_out / pdf)

    for fig in sorted((ROOT / "papers" / "figures").glob("*.png")):
        shutil.copy2(fig, figs_out / fig.name)

    # 3. Copy plain-English infographic for OG card & figure gallery
    assets_out = OUT / "assets"
    assets_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        ROOT / "research" / "wsp" / "figures" / "wsp_simple_infographic.png",
        assets_out / "wsp_simple_infographic.png",
    )

    # 4. Prevent GitHub Pages Jekyll interference
    (OUT / ".nojekyll").write_text("", encoding="utf-8")

    files = list(OUT.rglob("*"))
    total_mb = sum(f.stat().st_size for f in files if f.is_file()) / 1e6
    print(f"[OK] Built deployable static site in {OUT} ({sum(1 for f in files if f.is_file())} files, {total_mb:.2f} MB)")


if __name__ == "__main__":
    build()
