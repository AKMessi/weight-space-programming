"""Correct the E4 engineering output-cap mismatch; cases and interventions unchanged."""
import json
from pathlib import Path
from portfolio_batch import ROOTS,run_one
config=json.loads(Path('configs/portfolio.json').read_text())
for root in ROOTS:
    run_one(config,root+'_AB','datasets/portfolio/engineering.json','e4engineering-v2',max_tokens=192)
    run_one(config,root+'_BA','datasets/portfolio/engineering.json','e4ordercheck-v2',max_tokens=192,limit=6)
