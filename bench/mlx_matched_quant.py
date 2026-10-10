#!/usr/bin/env python3
"""Speed and output check of one MLX target for notes/mlx-matched-quant.md.

Runs ``Runner.bench`` (same code path as ``python -m mlx_spec bench``) without
touching the shared results files, writes rows to
``results/mlx_matched_quant.jsonl`` (``target`` field), then compares each
greedy output with llama.cpp's Q4_K_M baseline text
(``results/baseline_outputs/Qwen3.8-27B-Q4_K_M``): first differing token, and
``bench.diagnose`` at that token (our model's single-step logit margin between
llama.cpp's token and ours; |margin| < 0.25 = numerical near-tie). Those rows go
to ``results/mlx_matched_quant_vs_llama.jsonl``.

    scripts/gpu.py .venv/bin/python bench/mlx_matched_quant.py --target models/qwen38-27b-q4km-mlx \
        --configs greedy mtp:k=3:vocab=65536 --repeat 2
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mlx_spec import bench as B  # noqa: E402

LLAMA = ROOT / "results" / "baseline_outputs" / "Qwen3.8-27B-Q4_K_M"


def power():
    """Power source from pmset (timings on battery are suspect)."""
    import subprocess
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    return "battery" if "Battery Power" in out else "ac" if "AC Power" in out else out.strip()[:40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--configs", nargs="+", default=["greedy", "mtp:k=3:vocab=65536"])
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--no-diagnose", action="store_true")
    a = ap.parse_args()

    runner = B.Runner(a.target)
    prompts = B.load_prompts()
    rows = runner.bench(a.configs, prompts, repeat=a.repeat, max_tokens=a.max_tokens,
                        profile=True, write=False)
    with open(ROOT / "results" / "mlx_matched_quant.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps({"target": a.target, "power": power(), **r}) + "\n")

    if a.no_diagnose:
        return
    tok = runner.tokenizer
    bdir = B.baseline_dir(a.target)
    out = []
    for p in prompts:
        ours = json.loads((bdir / f"{p['id']}.json").read_text())
        ours = [t for t in ours if t not in runner.eos]
        ref_text = (LLAMA / f"{p['id']}.txt").read_text()
        ref = tok.encode(ref_text, add_special_tokens=False)
        ours_text = tok.decode(ours)
        n = min(len(ref), len(ours))
        d = next((i for i in range(n) if ref[i] != ours[i]), None)
        rec = {"target": a.target, "prompt_id": p["id"], "n_ours": len(ours), "n_llama": len(ref),
               "text_identical": ours_text == ref_text,
               "diverge_at_token": d, "time": time.strftime("%F %T")}
        if d is not None:
            rec["context"] = tok.decode(ours[max(0, d - 6):d])
            rec["llama_token"], rec["ours_token"] = tok.decode([ref[d]]), tok.decode([ours[d]])
            ids = B.encode_chat(tok, p["prompt"])
            rec["diagnose"] = B.diagnose(runner, ids, ref, ours)
        out.append(rec)
        print(json.dumps(rec), flush=True)
    with open(ROOT / "results" / "mlx_matched_quant_vs_llama.jsonl", "a") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
