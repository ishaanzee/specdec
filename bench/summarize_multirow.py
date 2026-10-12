#!/usr/bin/env python3
"""Table for notes/llama-metal-kernel.md from results/llama-multirow.jsonl (paired runs)."""
import json, collections
rows = collections.OrderedDict()
for l in open("results/llama-multirow.jsonl"):
    r = json.loads(l)
    rows.setdefault(r["config"], {})[r["prompt_id"]] = r
P = ["code_edit", "code_gen", "summarize", "prose", "explain", "math"]
print("| config | " + " | ".join(P) + " | mean tok/s | ref tok/s | speedup | accepted/drafted | match |")
print("|---|" + "---:|" * (len(P) + 4) + "---|")
for c, d in rows.items():
    if not all(p in d for p in P): continue
    g = [d[p]["gen_tok_s"] for p in P]
    ref = [d[p].get("ref_gen_tok_s") or d[p].get("reference_gen_tok_s") or 0 for p in P]
    sp = [d[p].get("speedup") or 0 for p in P]
    acc = sum(d[p]["draft_accepted"] or 0 for p in P); dr = sum(d[p]["draft_n"] or 0 for p in P)
    bad = [p for p in P if not d[p]["matches_baseline"]]
    m = "6/6" if not bad else f"{6-len(bad)}/6 (diff: {', '.join(bad)})"
    print(f"| {c} | " + " | ".join(f"{x:.2f}" for x in g) + f" | **{sum(g)/6:.2f}** | {sum(ref)/6:.2f} | {sum(sp)/6:.3f}x | {acc}/{dr} ({acc/max(dr,1):.0%}) | {m} |")
