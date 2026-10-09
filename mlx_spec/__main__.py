"""CLI. Run everything that loads a model under the GPU lock:

  scripts/gpu.py .venv/bin/python -m mlx_spec generate --config ngram:k=8 --prompt "..."
  scripts/gpu.py .venv/bin/python -m mlx_spec bench --configs greedy ngram:k=8:n=2-4 --repeat 2
  scripts/gpu.py .venv/bin/python -m mlx_spec verify-cost
  scripts/gpu.py .venv/bin/python -m mlx_spec mlxlm        # compare with mlx_lm.generate
  .venv/bin/python -m mlx_spec summary                       # results table (no GPU)

Config strings: ``method[:key=value]...``
  greedy                          plain greedy decode (stock mlx-lm caches)
  spec-none[:rollback=..]         speculative loop with an empty drafter (overhead check)
  ngram[:k=8][:n=2-4][:rollback=replay|capture|reforward]
  draft[:k=4][:model=mlx-community/Qwen3.5-0.8B-MLX-4bit][:rollback=..]
"""

import argparse
import os
import sys
import time

# Models must already be downloaded. Without this, mlx_lm.load() asks the Hub
# for updates and a stalled TLS handshake once hung a run while it held the
# GPU lock. Set HF_HUB_OFFLINE=0 to allow downloads.
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mlx_spec", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate for one prompt and print stats")
    g.add_argument("--config", default="ngram:k=8")
    g.add_argument("--prompt", default=None)
    g.add_argument("--prompt-id", default=None)
    g.add_argument("--max-tokens", type=int, default=256)
    g.add_argument("--profile", action="store_true")

    b = sub.add_parser("bench", help="benchmark configs over bench/prompts.jsonl")
    b.add_argument("--configs", nargs="+", required=True)
    b.add_argument("--repeat", type=int, default=1)
    b.add_argument("--max-tokens", type=int, default=256)
    b.add_argument("--only", default=None, help="comma-separated prompt ids")
    b.add_argument("--no-profile", action="store_true")
    b.add_argument("--no-write", action="store_true")

    v = sub.add_parser("verify-cost", help="target pass time vs number of tokens T")
    v.add_argument("--max-T", type=int, default=17)
    v.add_argument("--reps", type=int, default=7)
    v.add_argument("--modes", default="replay,capture,reforward")

    sub.add_parser("mlxlm", help="compare the greedy baseline with mlx_lm.generate")
    sub.add_parser("summary", help="markdown table of results/mlx_*.jsonl (no model)")

    a = ap.parse_args(argv)
    from . import bench as B

    if a.cmd == "summary":
        print(B.summary_table())
        return
    runner = B.Runner(a.target or B.TARGET)

    if a.cmd == "generate":
        prompts = {p["id"]: p for p in B.load_prompts()}
        text = a.prompt if a.prompt else prompts[a.prompt_id or "code_edit"]["prompt"]
        ids = B.encode_chat(runner.tokenizer, text)
        cfg = B.parse_config(a.config)
        runner.warmup(cfg)
        r = runner.run(cfg, ids, a.max_tokens, profile=a.profile)
        print(B.decode_text(runner.tokenizer, r.tokens, runner.eos))
        print(f"\n--- {a.config}: prompt {r.n_prompt} tok @ {r.prompt_tok_s:.1f} tok/s, "
              f"gen {r.n_gen} tok @ {r.gen_tok_s:.2f} tok/s, rounds {r.rounds}, "
              f"accepted {r.draft_accepted}/{r.draft_n}, hist {r.accept_hist}", file=sys.stderr)
        if a.profile:
            print(f"    draft {r.draft_s:.2f}s verify {r.verify_s:.2f}s rollback {r.rollback_s:.2f}s "
                  f"of {r.gen_s:.2f}s", file=sys.stderr)
    elif a.cmd == "bench":
        only = set(a.only.split(",")) if a.only else None
        runner.bench(a.configs, B.load_prompts(only=only), repeat=a.repeat,
                     max_tokens=a.max_tokens, profile=not a.no_profile, write=not a.no_write)
    elif a.cmd == "verify-cost":
        B.verify_cost(runner, Ts=range(1, a.max_T + 1), reps=a.reps, modes=a.modes.split(","))
    elif a.cmd == "mlxlm":
        compare_mlx_lm(runner, B)


def compare_mlx_lm(runner, B):
    """Our greedy loop vs mlx_lm.stream_generate on the same prompts; then try
    mlx-lm's own speculative path (expected to refuse the hybrid cache)."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)
    for p in B.load_prompts():
        ids = B.encode_chat(runner.tokenizer, p["prompt"])
        ours = B.greedy_generate(runner.model, ids, 256, runner.eos)
        toks, last = [], None
        for resp in stream_generate(runner.model, runner.tokenizer, ids, max_tokens=256, sampler=sampler):
            toks.append(resp.token)
            last = resp
        same = toks[: len(ours.tokens)] == ours.tokens
        print(f"{p['id']:<10} ours {ours.gen_tok_s:6.2f} tok/s ({ours.n_gen} tok, prompt {ours.prompt_tok_s:.0f} t/s) | "
              f"mlx_lm {last.generation_tps:6.2f} tok/s ({last.generation_tokens} tok, prompt {last.prompt_tps:.0f} t/s) "
              f"| same tokens: {same}", flush=True)
    text = B.decode_text(runner.tokenizer, ours.tokens, runner.eos)
    print("thinking disabled check, last output starts with:", repr(text[:80]))
    try:
        draft = runner.draft_model("mlx-community/Qwen3.5-0.8B-MLX-4bit")
        t = time.perf_counter()
        n = 0
        for resp in stream_generate(runner.model, runner.tokenizer, ids, max_tokens=64,
                                    sampler=sampler, draft_model=draft, num_draft_tokens=4):
            n += 1
        print(f"mlx_lm speculative: {resp.generation_tps:.2f} tok/s")
    except Exception as e:
        print(f"mlx_lm speculative path failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
