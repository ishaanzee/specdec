"""One GPU-lock session for the small-batch qmm kernel (mlx_spec/qmm_kernel.py):
verify-cost curve without / with the kernel, then MTP (vocab 65536) and
0.8B-draft benchmarks at k = 3..8 without / with it, interleaved, then a
logits diagnosis of every output that differs from greedy.

    scripts/gpu.py .venv/bin/python bench/mlx_qmm_session.py [--range 2-16] [--ks 3 4 5 6 7 8]

Rows: results/mlx_verify_cost.jsonl (field "qmm"), results/mlx_{greedy,mtp,draft}.jsonl
(config names end in "_qmm2-16"), diagnoses in results/mlx_qmm_diagnose.jsonl.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx  # noqa: E402

from mlx_spec import bench as B  # noqa: E402
from mlx_spec import qmm_kernel as Q  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--range", default="2-16")
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8])
    ap.add_argument("--methods", nargs="+", default=["mtp:vocab=65536", "draft"])
    ap.add_argument("--no-verify-cost", action="store_true")
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--no-diagnose", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated prompt ids")
    a = ap.parse_args()

    r = B.Runner()
    meta = os.path.join(B.ROOT, "results", "mlx_qmm_session.jsonl")

    def mark(stage):  # power source per stage: timings on battery are suspect
        row = {"experiment": "qmm_session", "stage": stage, "power": Q.power_source(),
               "loadavg": round(os.getloadavg()[0], 2), "time": time.strftime("%F %T")}
        print(f"[qmm-session] {json.dumps(row)}", flush=True)
        with open(meta, "a") as f:
            f.write(json.dumps(row) + "\n")

    mark("start")
    if not a.no_verify_cost:
        for rng in (None, a.range):
            Q.install(r.model, rng)
            print(f"== verify-cost, qmm={rng}", flush=True)
            B.verify_cost(r, Ts=range(1, 18), reps=7, modes=("replay",))
        Q.uninstall(r.model)
        mark("verify-cost done")
    if a.no_bench:
        return

    configs = ["greedy"]
    for k in a.ks:
        for m in a.methods:
            method, *rest = m.split(":")
            base = ":".join([method, f"k={k}"] + rest)
            configs += [base, f"{base}:qmm={a.range}"]
    prompts = B.load_prompts(only=set(a.only.split(",")) if a.only else None)
    rows = r.bench(configs, prompts, repeat=1, max_tokens=256, profile=True, write=True)
    mark("bench done")
    if a.no_diagnose:
        return

    bdir = B.baseline_dir(r.target_name)
    by_id = {p["id"]: p for p in prompts}
    seen = {}
    for row in rows:
        if row["matches_baseline"] is not False:
            continue
        p = by_id[row["prompt_id"]]
        ids = B.encode_chat(r.tokenizer, p["prompt"])
        ref = json.loads((bdir / f"{p['id']}.json").read_text())
        spec = next(c for c in configs if B.parse_config(c)["method"] == row["method"]
                    and B.config_name(B.parse_config(c)) == row["config"])
        cfg = B.parse_config(spec)
        got = r.run(cfg, ids, 256).tokens  # leaves the kernel installed iff cfg has qmm
        key = (p["id"], row["diverge_at_token"], cfg.get("qmm"))
        if key not in seen:
            seen[key] = B.diagnose(r, ids, ref, got)  # batched logits with the same kernel state
        d = seen[key]
        print(f"DIAG {spec:<36} {p['id']:<9} {json.dumps(d)}", flush=True)
        with open(B.ROOT / "results" / "mlx_qmm_diagnose.jsonl", "a") as f:
            f.write(json.dumps({"config": spec, "prompt_id": p["id"], "qmm": cfg.get("qmm"),
                                "time": time.strftime("%F %T"), **d}) + "\n")
    Q.uninstall(r.model)
    mx.clear_cache()


if __name__ == "__main__":
    main()
