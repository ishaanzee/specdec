#!/usr/bin/env python3
"""Print markdown tables from a results/<experiment>.jsonl written by bench/llama_bench.py.

  .venv/bin/python bench/llama_results_table.py results/llama-ngram-draft.jsonl [--paired-only]

One row per (method, config) run, one column per prompt: gen tok/s, and for
paired runs the speedup vs the adjacent no-spec reference. Also mean acceptance
and whether every prompt matched the baseline output.
"""
import json, sys, statistics
from collections import OrderedDict

path = sys.argv[1]
paired_only = "--paired-only" in sys.argv
rows = [json.loads(l) for l in open(path) if l.strip()]
groups = OrderedDict()
for r in rows:
    key = (r["method"], r["config"], r.get("time", "")[:16], "speedup" in r)
    # rows written by one harness invocation share method/config and (nearly) the same time
    k2 = next((k for k in groups if k[:2] == key[:2] and k[3] == key[3] and r["prompt_id"] not in groups[k]), None)
    groups.setdefault(k2 or key, OrderedDict())[r["prompt_id"]] = r

pids = []
for g in groups.values():
    for p in g:
        if p not in pids:
            pids.append(p)

hdr = "| method | config | n | " + " | ".join(pids) + " | mean | accept | match |"
print(hdr)
print("|" + "---|" * (hdr.count("|") - 1))
for (m, c, t, paired), g in groups.items():
    if paired_only and not paired:
        continue
    cells = []
    for p in pids:
        r = g.get(p)
        if not r:
            cells.append("")
        elif paired:
            cells.append(f"{r['gen_tok_s']:.2f} ({r['speedup']:.2f}x)")
        else:
            cells.append(f"{r['gen_tok_s']:.2f}")
    vals = list(g.values())
    mean = statistics.mean(r["gen_tok_s"] for r in vals)
    mean_s = f"{mean:.2f}" + (f" ({statistics.mean(r['speedup'] for r in vals):.2f}x)" if paired else "")
    dn = sum(r["draft_n"] or 0 for r in vals)
    da = sum(r["draft_accepted"] or 0 for r in vals)
    acc = f"{da}/{dn} ({da/dn:.0%})" if dn else "-"
    match = "all" if all(r["matches_baseline"] for r in vals) else ",".join(r["prompt_id"] for r in vals if not r["matches_baseline"]) + " differ"
    print(f"| {m} | {c} | {vals[0]['repeat']} | " + " | ".join(cells) + f" | {mean_s} | {acc} | {match} |")
