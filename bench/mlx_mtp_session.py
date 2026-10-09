"""Track B MTP benchmark session: one GPU-lock session, results in
results/mlx_mtp.jsonl (+ greedy / draft reference rows in their usual files),
then a logits diagnosis of every output that differs from greedy.

    scripts/gpu.py .venv/bin/python bench/mlx_mtp_session.py [--configs ...]
"""
import argparse
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mlx_spec import bench as B

DEFAULT = ["greedy", "mtp:k=1", "mtp:k=2", "mtp:k=3", "mtp:k=4",
           "mtp:k=2:vocab=65536", "mtp:k=3:vocab=65536", "mtp:k=3:vocab=32768",
           "mtp-hybrid:k=3", "mtp-hybrid:k=3:vocab=65536", "draft:k=3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=DEFAULT)
    ap.add_argument("--no-diagnose", action="store_true")
    a = ap.parse_args()
    r = B.Runner()
    prompts = B.load_prompts()
    rows = r.bench(a.configs, prompts, repeat=1, max_tokens=256, profile=True, write=True)
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
        cfg = next(c for c in a.configs if B.parse_config(c)["method"] == row["method"]
                   and B.config_name(B.parse_config(c)) == row["config"])
        got = r.run(B.parse_config(cfg), ids, 256).tokens
        key = (p["id"], row["diverge_at_token"])
        if key not in seen:
            seen[key] = B.diagnose(r, ids, ref, got)
        d = seen[key]
        print(f"DIAG {cfg:<28} {p['id']:<9} {json.dumps(d)}", flush=True)
        with open(B.ROOT / "results" / "mlx_mtp_diagnose.jsonl", "a") as f:
            f.write(json.dumps({"config": cfg, "prompt_id": p["id"], **d}) + "\n")


if __name__ == "__main__":
    main()
