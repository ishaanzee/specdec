"""MTP drafter exploration (Track B): acceptance per variant and draft cost.

    scripts/gpu.py .venv/bin/python bench/mlx_mtp_explore.py [--tokens 128] [--prompts code_gen,prose,math]

Prints, per variant (bits x hidden), accept histogram and per-position
acceptance at k=3, and the time of the MTP drafting alone for k = 1..4.
Writes nothing except logs (stdout).
"""
import argparse
import os
import statistics
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx

from mlx_spec import bench as B
from mlx_spec.decode import speculative_generate
from mlx_spec.mtp import MTPDrafter, load_mtp_head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--prompts", default="code_gen,prose,math")
    ap.add_argument("--bits", default="4,8,16")
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    r = B.Runner()
    prompts = B.load_prompts(only=set(a.prompts.split(",")))
    for bits in [int(b) for b in a.bits.split(",")]:
        head = load_mtp_head(r.spec.lm.args, bits=bits)
        for hidden in (["normed", "hidden"] if bits == 4 else ["normed"]):
            tot = [0] * (a.k + 1)
            tps = []
            for p in prompts:
                ids = B.encode_chat(r.tokenizer, p["prompt"])
                d = MTPDrafter(r.spec, head, hidden=hidden)
                res = speculative_generate(r.spec, d, ids, k=a.k, max_tokens=a.tokens, eos=r.eos)
                base = B.baseline_dir(r.target_name) / f"{p['id']}.json"
                import json
                ref = json.loads(base.read_text())[: len(res.tokens)] if base.exists() else None
                for i, c in enumerate(res.accept_hist):
                    tot[i] += c
                tps.append(res.gen_tok_s)
                print(f"bits={bits:<2} hidden={hidden:<6} {p['id']:<9} acc {res.draft_accepted}/{res.draft_n} "
                      f"hist {res.accept_hist} {res.gen_tok_s:.2f} tok/s match={res.tokens == ref}", flush=True)
            print(f"== bits={bits} hidden={hidden} k={a.k}: hist {tot} by-pos {B.accept_by_pos(tot)} "
                  f"accept {sum(i * c for i, c in enumerate(tot)) / (a.k * sum(tot)):.3f} "
                  f"mean {statistics.mean(tps):.2f} tok/s", flush=True)
        # draft cost alone
        ids = B.encode_chat(r.tokenizer, prompts[0]["prompt"])
        cache = r.spec.make_cache()
        pre = r.spec.prefill(mx.array(ids, mx.uint32), cache)
        for k in (1, 2, 3, 4):
            ts = []
            for i in range(8):
                d = MTPDrafter(r.spec, head)
                d.start(ids, pre)
                mx.synchronize()
                t = time.perf_counter()
                mx.eval(d.draft(int(ids[-1]), k))
                ts.append(time.perf_counter() - t)
            print(f"bits={bits} draft k={k}: {1e3 * statistics.median(ts[2:]):.2f} ms", flush=True)
        del head, cache, pre
        mx.clear_cache()


if __name__ == "__main__":
    main()
