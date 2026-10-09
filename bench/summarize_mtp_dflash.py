#!/usr/bin/env python3
"""Markdown tables from results/llama-mtp-dflash.jsonl (one table row per method/config, in run order).

  .venv/bin/python bench/summarize_mtp_dflash.py [results/llama-mtp-dflash.jsonl]
"""
import json, sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "llama-mtp-dflash.jsonl"
rows = [json.loads(l) for l in open(path) if l.strip()]
groups = OrderedDict()
for r in rows:
    groups.setdefault((r["method"], r["config"], r["model"]), []).append(r)
prompts = list(OrderedDict.fromkeys(r["prompt_id"] for r in rows))

print("| method | config | " + " | ".join(prompts) + " | mean tok/s | accepted/drafted | match |")
print("|---|---|" + "---:|" * len(prompts) + "---:|---:|---|")
for (method, cfg, model), rs in groups.items():
    by = {r["prompt_id"]: r for r in rs}
    cells = [f"{by[p]['gen_tok_s']:.2f}" if p in by else "" for p in prompts]
    mean = sum(r["gen_tok_s"] for r in rs) / len(rs)
    dn = sum(r["draft_n"] or 0 for r in rs)
    da = sum(r["draft_accepted"] or 0 for r in rs)
    acc = f"{da}/{dn} ({da / dn:.0%})" if dn else "-"
    nm = sum(1 for r in rs if r["matches_baseline"])
    bad = [r["prompt_id"] for r in rs if r["matches_baseline"] is False]
    match = f"{nm}/{len(rs)}" + (f" (diff: {', '.join(bad)})" if bad else "")
    print(f"| {method} | {cfg} | " + " | ".join(cells) + f" | **{mean:.2f}** | {acc} | {match} |")
