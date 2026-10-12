"""Headline numbers: the Q4_K_M transcode and mlx-community 4-bit in ONE GPU-lock
session (models loaded one after the other; they don't fit together), same
configs on both, then a logits diagnosis of every output that differs from
greedy.

    scripts/gpu.py .venv/bin/python bench/mlx_final_session.py [--repeat 2]

Rows: results/mlx_{greedy,mtp,draft}.jsonl (field "model" tells the target),
power source per stage in results/mlx_final_session.jsonl, diagnoses in
results/mlx_qmm_diagnose.jsonl.
"""
import argparse
import gc
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx  # noqa: E402

from mlx_spec import bench as B  # noqa: E402
from mlx_spec import qmm_kernel as Q  # noqa: E402

TARGETS = ["models/qwen38-27b-q4km-mlx", "mlx-community/Qwen3.8-27B-4bit"]
CONFIGS = ["greedy", "mtp:k=3:vocab=65536"] + [f"mtp:k={k}:vocab=65536:qmm=5-16" for k in (4, 5, 6, 7)] \
    + ["draft:k=6:qmm=5-16"]
META = os.path.join(B.ROOT, "results", "mlx_final_session.jsonl")


def mark(stage, **kw):
    row = {"experiment": "final_session", "stage": stage, "power": Q.power_source(),
           "loadavg": round(os.getloadavg()[0], 2), "time": time.strftime("%F %T"), **kw}
    print(f"[final] {json.dumps(row)}", flush=True)
    with open(META, "a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--no-diagnose", action="store_true")
    a = ap.parse_args()
    prompts = B.load_prompts()
    by_id = {p["id"]: p for p in prompts}
    for target in a.targets:
        mark("load", target=target)
        r = B.Runner(target)
        rows = r.bench(a.configs, prompts, repeat=a.repeat, max_tokens=256, profile=True, write=True)
        mark("bench done", target=target)
        if not a.no_diagnose:
            bdir = B.baseline_dir(r.target_name)
            seen = {}
            for row in rows:
                if row["matches_baseline"] is not False:
                    continue
                p = by_id[row["prompt_id"]]
                ids = B.encode_chat(r.tokenizer, p["prompt"])
                ref = json.loads((bdir / f"{p['id']}.json").read_text())
                spec = next(c for c in a.configs if B.parse_config(c)["method"] == row["method"]
                            and B.config_name(B.parse_config(c)) == row["config"])
                cfg = B.parse_config(spec)
                got = r.run(cfg, ids, 256).tokens  # kernel installed iff cfg has qmm
                key = (p["id"], row["diverge_at_token"], cfg.get("qmm"))
                if key not in seen:
                    seen[key] = B.diagnose(r, ids, ref, got)
                d = seen[key]
                print(f"DIAG {target} {spec:<34} {p['id']:<9} {json.dumps(d)}", flush=True)
                with open(B.ROOT / "results" / "mlx_qmm_diagnose.jsonl", "a") as f:
                    f.write(json.dumps({"model": target, "config": spec, "prompt_id": p["id"],
                                        "qmm": cfg.get("qmm"), "time": time.strftime("%F %T"), **d}) + "\n")
            mark("diagnose done", target=target)
        Q.uninstall(r.model)
        del r
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    main()
