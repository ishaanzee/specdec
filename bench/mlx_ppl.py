#!/usr/bin/env python3
"""Perplexity and KL divergence of an MLX model on exactly llama-perplexity's chunks.

Reads the token ids and the base log-probs from a llama-perplexity
``--kl-divergence-base`` file (written by e.g.
``llama-perplexity -m Q4_K_M.gguf -f wiki.test.raw -c 2048 --chunks 16
--kl-divergence-base base.kld``), so both engines score the same tokens:
non-overlapping chunks of n_ctx tokens, no BOS (Qwen's tokenizer doesn't add
one), each chunk evaluated from an empty cache, only the second half scored
(positions n_ctx/2 .. n_ctx-2 predict the next token), PPL = exp(mean NLL).
KL(base || model), top-1 agreement and the RMS of the probability of the
correct token follow llama.cpp's ``kl_divergence`` (base log-probs below -16
are ignored, as there).

    scripts/gpu.py .venv/bin/python bench/mlx_ppl.py --model models/qwen38-27b-q4km-mlx \
        --base models/wikitext/q4km-base.kld [--text models/wikitext/wikitext-2-raw/wiki.test.raw]
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def read_base(path):
    f = open(path, "rb")
    assert f.read(8) == b"_logits_"
    n_ctx, n_vocab, n_chunk = np.frombuffer(f.read(12), dtype=np.int32)
    tokens = np.frombuffer(f.read(4 * n_chunk * n_ctx), dtype=np.int32).reshape(n_chunk, n_ctx)
    off = f.tell()
    f.close()
    nv = 2 * ((n_vocab + 1) // 2) + 4
    n_tok = n_ctx - 1 - n_ctx // 2
    mm = np.memmap(path, dtype=np.uint16, mode="r", offset=off, shape=(n_chunk, n_tok, nv))
    return int(n_ctx), int(n_vocab), tokens, mm


def power():
    """Power source from pmset (timings on battery are suspect)."""
    import subprocess
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    return "battery" if "Battery Power" in out else "ac" if "AC Power" in out else out.strip()[:40]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True, help="llama-perplexity --kl-divergence-base file")
    ap.add_argument("--text", default=None, help="check that HF tokenization of this file matches the base tokens")
    ap.add_argument("--chunks", type=int, default=None)
    ap.add_argument("--fp32-head", action="store_true", help="lm_head in float32 instead of the model dtype")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "quality_ppl.jsonl"))
    a = ap.parse_args()

    from mlx_lm import load

    n_ctx, n_vocab, tokens, base = read_base(a.base)
    n_chunk = min(a.chunks or tokens.shape[0], tokens.shape[0])
    first = n_ctx // 2
    model, tok = load(a.model)
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    tm = model.language_model if hasattr(model, "language_model") else model
    inner, head = tm.model, tm.lm_head

    tok_match = None
    if a.text:
        ids = tok.encode(open(a.text, encoding="utf-8").read())
        flat = tokens[:n_chunk].reshape(-1)
        tok_match = bool(len(ids) >= flat.size and np.array_equal(np.array(ids[: flat.size]), flat))
        print(f"HF tokenization matches llama.cpp's for the {flat.size} scored-chunk tokens: {tok_match}"
              f" ({len(ids)} tokens in the file)", flush=True)

    acc = dict(nll=0.0, nll2=0.0, nll_base=0.0, kld=0.0, kld2=0.0, same_top=0, count=0,
               dp2=0.0, maxdp=0.0)
    kl_all = []
    t0 = time.time()
    for c in range(n_chunk):
        x = mx.array(tokens[c].astype(np.uint32))[None]
        h = inner(x)  # final-normed hidden states [1, n_ctx, 5120]
        h = h[0, first : n_ctx - 1]
        tgt = mx.array(tokens[c, first + 1 :].astype(np.int32))
        for r0 in range(0, h.shape[0], 256):
            hr = h[r0 : r0 + 256]
            logits = head(hr.astype(mx.float32) if a.fp32_head else hr).astype(mx.float32)
            lse = mx.logsumexp(logits, axis=-1)
            t = tgt[r0 : r0 + 256]
            logp = logits - lse[:, None]
            nll = -mx.take_along_axis(logp, t[:, None], axis=-1)[:, 0]
            b = mx.array(np.ascontiguousarray(base[c, r0 : r0 + hr.shape[0]]))
            sc = b[:, :2].view(mx.float32)[:, 0]
            mn = b[:, 2:4].view(mx.float32)[:, 0]
            blogp = sc[:, None] * b[:, 4 : 4 + n_vocab].astype(mx.float32) + mn[:, None]
            pb = mx.where(blogp > -16.0, mx.exp(blogp), 0.0)
            kld = (pb * (blogp - logp)).sum(-1)
            nll_b = -mx.take_along_axis(blogp, t[:, None], axis=-1)[:, 0]
            same = (mx.argmax(logits, -1) == mx.argmax(blogp, -1)).sum()
            dp = mx.exp(-nll) - mx.exp(-nll_b)
            mx.eval(nll, kld, nll_b, same, dp)
            nll, kld, nll_b, dp = (np.array(v, dtype=np.float64) for v in (nll, kld, nll_b, dp))
            acc["nll"] += nll.sum(); acc["nll2"] += (nll ** 2).sum(); acc["nll_base"] += nll_b.sum()
            acc["kld"] += kld.sum(); acc["kld2"] += (kld ** 2).sum(); acc["same_top"] += int(same.item())
            acc["dp2"] += (dp ** 2).sum(); acc["maxdp"] = max(acc["maxdp"], float(np.abs(dp).max()))
            acc["count"] += nll.size
            kl_all.append(kld)
        n = acc["count"]
        print(f"[{c + 1}/{n_chunk}] {time.time() - t0:6.0f}s  PPL {np.exp(acc['nll'] / n):.4f}"
              f"  base PPL {np.exp(acc['nll_base'] / n):.4f}  KLD {acc['kld'] / n:.5f}"
              f"  same top {acc['same_top'] / n:.4f}", flush=True)

    n = acc["count"]
    kl_all = np.concatenate(kl_all)
    mean_nll = acc["nll"] / n
    se_nll = np.sqrt(max(acc["nll2"] / n - mean_nll ** 2, 0) / (n - 1))
    row = {
        "label": a.label or a.model, "model": a.model, "base": os.path.basename(a.base),
        "n_ctx": n_ctx, "chunks": n_chunk, "scored_tokens": n, "fp32_head": a.fp32_head,
        "tokenization_matches": tok_match,
        "ppl": float(np.exp(mean_nll)), "ppl_se": float(np.exp(mean_nll) * se_nll),
        "base_ppl": float(np.exp(acc["nll_base"] / n)),
        "kld_mean": acc["kld"] / n, "kld_median": float(np.median(kl_all)),
        "kld_p99": float(np.percentile(kl_all, 99)), "kld_max": float(kl_all.max()),
        "same_top": acc["same_top"] / n, "rms_dp": float(np.sqrt(acc["dp2"] / n)),
        "max_dp": acc["maxdp"], "seconds": time.time() - t0, "power": power(),
    }
    print(json.dumps(row), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
