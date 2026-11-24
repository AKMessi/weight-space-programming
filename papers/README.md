# Companion Research Papers (Aaryan Kakad, November 2026)

Both papers are programmatically audited against immutable raw JSON logs in `../research/wsp/` (`python scripts/wsp_verify_papers.py` and `python scripts/wsp_build_pdfs.py`).

## Camera-Ready PDFs
1. [**Paper 1 — The Greedy-Perturbation Illusion and Zero-Gradient Control-Flow Compilation in Weight Space (11 pages)**](Paper1_Greedy_Perturbation_Illusion_and_Zero_Gradient_WSP_Kakad2026.pdf)
   - Sources: [Markdown](paper1_conditional_wsp_and_greedy_illusion.md) · [Typst](typst/paper1_conditional_wsp_and_greedy_illusion.typ) · [arXiv LaTeX](latex/paper1_arxiv.tex)
   - Figures (300 DPI): [Fig 1](figures/p1_fig1_greedy_illusion_and_stochastic.png) · [Fig 2](figures/p1_fig2_universal_layer_anatomy.png) · [Fig 3](figures/p1_fig3_ood_and_cross_arch.png) · [Fig 4](figures/p1_fig4_pareto_ppl_vs_acc.png)

2. [**Paper 2 — The Quantization Cliff of Weight-Space Programming: Hardware-Aligned Constant-Block Embedding and Post-Quantization Healing (10 pages)**](Paper2_Quantization_Cliff_and_BA_WSC_Kakad2026.pdf)
   - Sources: [Markdown](paper2_quantization_cliff_and_ba_wsc.md) · [Typst](typst/paper2_quantization_cliff_and_ba_wsc.typ) · [arXiv LaTeX](latex/paper2_arxiv.tex)
   - Figures (300 DPI): [Fig 1](figures/p2_fig1_quant_loop_cliff_3models.png) · [Fig 2](figures/p2_fig2_theorem1_corollary1_3models.png) · [Fig 3](figures/p2_fig3_pq_healing_matrix_qwen3.png) · [Fig 4](figures/p2_fig4_double_crossover_2models.png)

## Shared Assets & Build Commands
- Bibliography: [`references.bib`](references.bib)
- Typst Academic Preprint Template: [`typst/template.typ`](typst/template.typ)
- Rebuild figures: `python scripts/wsp_generate_all_paper_figures.py`
- Rebuild & verify PDFs: `python scripts/wsp_build_pdfs.py`
- Audit all cited numbers & statistics: `python scripts/wsp_verify_papers.py`
