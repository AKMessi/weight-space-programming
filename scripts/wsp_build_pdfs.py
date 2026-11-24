"""Compile Paper 1 & Paper 2 Typst manuscripts to publication-grade PDFs,
programmatically verify every cited number inside the rendered PDF text,
and export page-by-page PNG previews for visual inspection."""
from __future__ import annotations

import json
import pathlib
import sys
import fitz  # PyMuPDF
import typst


ROOT = pathlib.Path(__file__).resolve().parents[1]
PREVIEW_DIR = ROOT / "artifacts" / "pdf_preview"

PAPERS = [
    (
        "paper1",
        ROOT / "papers" / "typst" / "paper1_conditional_wsp_and_greedy_illusion.typ",
        ROOT / "papers" / "Paper1_Greedy_Perturbation_Illusion_and_Zero_Gradient_WSP_Kakad2026.pdf",
        [
            "Aaryan Kakad",
            "Independent Machine Learning Researcher",
            "105 (35.0%)",
            "104 (34.7%)",
            "122 (40.7%)",
            "0.0148",
            "0.0146",
            "114 (38.0%)",
            "71/100",
            "80/100",
            "56/100",
            "71/100 (+15pp, p=0.0003)",
            "67/100 (+11pp, p=0.0074)",
            "10 (6.7%, 10",
            "0.0039",
            "43 (28.7%)",
            "53 (35.3%)",
            "49 (32.7%)",
            "+61 / -25",
            "+65 / -26",
            "0.000130",
            "f9c9f1d3c1e21755",
            "cedf89af31c9041b",
            "6ec1aa822868a25c",
            "18,513",
        ],
    ),
    (
        "paper2",
        ROOT / "papers" / "typst" / "paper2_quantization_cliff_and_ba_wsc.typ",
        ROOT / "papers" / "Paper2_Quantization_Cliff_and_BA_WSC_Kakad2026.pdf",
        [
            "Aaryan Kakad",
            "Independent Machine Learning Researcher",
            "0.99732 / 0.00050",
            "4,514",
            "0.9962 / 0.00005",
            "16,685",
            "10,119",
            "0.99741 / 0.00034",
            "12,133",
            "35 (23.3%)",
            "46 (30.7%, +7.3pp)",
            "50 (33.3%, +10.0pp)",
            "19 (12.7%)",
            "32 (21.3%, +8.7pp)",
            "0.0044",
            "15 (#1!)",
            "0.00336",
            "66 (+6)",
            "45 (-3)",
            "54 (+6)",
            "0.0663",
            "0.00210",
            "f9c9f1d3c1e21755",
            "cedf89af31c9041b",
            "6ec1aa822868a25c",
            "18,513",
        ],
    ),
]


def main() -> None:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    # Clean old previews
    for old in PREVIEW_DIR.glob("*.png"):
        old.unlink()

    all_ok = True
    for tag, src, dst, required_tokens in PAPERS:
        print(f"Compiling {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)} ...")
        pdf_bytes = typst.compile(str(src), root=str(ROOT))
        dst.write_bytes(pdf_bytes)

        doc = fitz.open(dst)
        full_text = "\n".join(page.get_text("text") for page in doc)
        # Normalize hyphenated line breaks and whitespace slightly for substring checks
        norm_text = " ".join(full_text.split())

        missing = []
        for tok in required_tokens:
            if tok not in full_text and tok not in norm_text:
                missing.append(tok)

        print(f"  Pages: {len(doc)} | Size: {dst.stat().st_size / 1e6:.2f} MB | Images embedded: {sum(len(p.get_images()) for p in doc)}")
        if missing:
            print(f"  [FAIL] Missing tokens in rendered PDF {tag}: {missing}")
            all_ok = False
        else:
            print(f"  [OK] All {len(required_tokens)}/{len(required_tokens)} PDF text audit tokens verified!")

        for idx, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=145)
            out_png = PREVIEW_DIR / f"{tag}_p{idx:02d}.png"
            pix.save(str(out_png))
        doc.close()

    if not all_ok:
        sys.exit(1)
    print("\nBOTH PDFS COMPILED, VERIFIED, AND RENDERED TO PAGE PREVIEWS SUCCESSFULLY!")


if __name__ == "__main__":
    main()
