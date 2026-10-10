#!/usr/bin/env python3
"""Turn bench/metal_mv_perf.sh logs into a table: us/run per (shape, n) for every config."""
import re, sys, collections
txt = open(sys.argv[1]).read()
cfgs, cur = [], None
data = collections.defaultdict(dict)
for part in re.split(r"(=== config: .*)", txt):
    if part.startswith("=== config:"):
        cur = part[len("=== config: "):].replace("GGML_METAL_MV_MULTIROW", "MR").strip(); cfgs.append(cur); continue
    if cur is None: continue
    # a case header followed (possibly after log noise) by its timing
    for m in re.finditer(r"MUL_MAT\(type_a=(\w+),type_b=f32,m=(\d+),n=(\d+),k=(\d+)[^:]*\):(.*?)(?=MUL_MAT\(|\Z)", part, re.S):
        t = re.search(r"([\d.]+) us/run", m.group(5))
        if t: data[(m.group(1), int(m.group(2)), int(m.group(4)), int(m.group(3)))][cur] = float(t.group(1))
shapes = sorted({k[:3] for k in data}, key=lambda s: (s[0], -s[1]))
for s in shapes:
    w = {"q4_K": 144/256, "q6_K": 210/256, "q8_0": 34/32}[s[0]] * s[1] * s[2]
    print(f"\n{s[0]} m={s[1]} k={s[2]} ({w/1e6:.1f} MB)  us/run (GB/s of weights)")
    print("| n | " + " | ".join(cfgs) + " |"); print("|---|" + "---:|" * len(cfgs))
    for n in sorted({k[3] for k in data if k[:3] == s}):
        row = data[s + (n,)]
        print(f"| {n} | " + " | ".join(f"{row[c]:.0f} ({w/row[c]/1e3:.0f})" if c in row else "-" for c in cfgs) + " |")
