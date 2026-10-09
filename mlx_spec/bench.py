"""Benchmarks: per-prompt generation runs, verify-cost curve, mlx-lm comparison.

Every function here loads or runs a model, so call them through
``scripts/gpu.py`` (the CLI in ``__main__`` is the usual entry point).
"""

import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List

import mlx.core as mx

from .decode import GenResult, greedy_generate, speculative_generate
from .drafters import DraftModelDrafter, NgramDrafter, NoDrafter, restore, snapshot
from .model import SpecModel

ROOT = Path(__file__).resolve().parent.parent
TARGET = "mlx-community/Qwen3.8-27B-4bit"


# ------------------------------------------------------------------ helpers
def load_prompts(path=ROOT / "bench" / "prompts.jsonl", only=None) -> List[dict]:
    ps = [json.loads(l) for l in open(path) if l.strip()]
    return [p for p in ps if not only or p["id"] in only]


def encode_chat(tokenizer, text: str) -> List[int]:
    """Chat template with thinking disabled (``<think>\\n\\n</think>`` prefilled)."""
    s = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )
    return tokenizer.encode(s, add_special_tokens=False)


def eos_ids(tokenizer) -> set:
    return set(tokenizer.eos_token_ids)


def decode_text(tokenizer, tokens, eos) -> str:
    return tokenizer.decode([t for t in tokens if t not in eos])


def baseline_dir(model_tag: str) -> Path:
    return ROOT / "results" / "mlx_baseline_tokens" / model_tag.replace("/", "--")


def parse_config(spec: str) -> Dict[str, str]:
    """'ngram:k=8:n=2-4:rollback=replay' -> {'method': 'ngram', 'k': '8', ...}."""
    method, *kvs = spec.split(":")
    cfg = {"method": method}
    for kv in kvs:
        key, val = kv.split("=", 1)
        cfg[key] = val
    return cfg


def config_name(cfg: Dict[str, str]) -> str:
    return "_".join(f"{k}{v}" for k, v in cfg.items() if k != "method") or "default"


# ------------------------------------------------------------------- runner
class Runner:
    """Holds the target (and draft models) for one locked benchmark session."""

    def __init__(self, target: str = TARGET):
        from mlx_lm import load

        self.target_name = target
        self.model, self.tokenizer = load(target)
        self.spec = SpecModel(self.model)
        self.eos = eos_ids(self.tokenizer)
        self.drafts = {}

    def draft_model(self, name):
        if name not in self.drafts:
            from mlx_lm import load

            m, tok = load(name)
            self.drafts[name] = m
        return self.drafts[name]

    def make_drafter(self, cfg):
        m = cfg["method"]
        if m == "spec-none":
            return NoDrafter()
        if m == "ngram":
            lo, hi = (int(x) for x in cfg.get("n", "2-4").split("-"))
            return NgramDrafter(n_max=hi, n_min=lo)
        if m == "draft":
            name = cfg.get("model", "mlx-community/Qwen3.5-0.8B-MLX-4bit")
            return DraftModelDrafter(self.draft_model(name), name=name)
        raise ValueError(f"unknown method {m}")

    def run(self, cfg, prompt_ids, max_tokens, profile=False) -> GenResult:
        if cfg["method"] == "greedy":
            return greedy_generate(self.model, prompt_ids, max_tokens, self.eos)
        return speculative_generate(
            self.spec,
            self.make_drafter(cfg),
            prompt_ids,
            k=int(cfg.get("k", 4)),
            max_tokens=max_tokens,
            eos=self.eos,
            profile=profile,
            rollback=cfg.get("rollback", "replay"),
        )

    def warmup(self, cfg):
        ids = encode_chat(self.tokenizer, "Say hi.")
        self.run(cfg, ids, 8)

    def bench(self, configs, prompts, repeat=1, max_tokens=256, profile=True, write=True):
        tag = self.target_name
        bdir = baseline_dir(tag)
        all_rows = []
        for spec in configs:
            cfg = parse_config(spec)
            self.warmup(cfg)
            rows = []
            for p in prompts:
                ids = encode_chat(self.tokenizer, p["prompt"])
                runs = [self.run(cfg, ids, max_tokens) for _ in range(repeat)]
                prof = self.run(cfg, ids, max_tokens, profile=True) if profile and cfg["method"] != "greedy" else None
                r0 = runs[0]
                text = decode_text(self.tokenizer, r0.tokens, self.eos)
                bfile = bdir / f"{p['id']}.json"
                if cfg["method"] == "greedy":
                    bdir.mkdir(parents=True, exist_ok=True)
                    bfile.write_text(json.dumps(r0.tokens))
                matches, diverge = None, None
                if bfile.exists():
                    base = json.loads(bfile.read_text())
                    matches = r0.tokens == base
                    if not matches:
                        diverge = next((i for i, (a, b) in enumerate(zip(r0.tokens, base)) if a != b),
                                       min(len(base), len(r0.tokens)))
                med = lambda xs: statistics.median(xs)
                row = {
                    "track": "B",
                    "method": cfg["method"],
                    "config": config_name(cfg),
                    "prompt_id": p["id"],
                    "kind": p.get("kind"),
                    "n_gen": r0.n_gen,
                    "gen_tok_s": round(med([r.gen_tok_s for r in runs]), 3),
                    "prompt_tok_s": round(med([r.prompt_tok_s for r in runs]), 2),
                    "draft_n": r0.draft_n if cfg["method"] != "greedy" else None,
                    "draft_accepted": r0.draft_accepted if cfg["method"] != "greedy" else None,
                    "accept_rate": round(r0.accept_rate, 4) if r0.accept_rate is not None else None,
                    "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "matches_baseline": matches,
                    "diverge_at_token": diverge,
                    "repeats_consistent": all(r.tokens == r0.tokens for r in runs),
                    "n_prompt": r0.n_prompt,
                    "rounds": r0.rounds,
                    "tokens_per_round": round(r0.n_gen / r0.rounds, 3) if r0.rounds else None,
                    "accept_hist": r0.accept_hist or None,
                    "repeat": repeat,
                    "runs": [{"gen_tok_s": round(r.gen_tok_s, 3), "gen_s": round(r.gen_s, 3)} for r in runs],
                    "model": tag,
                    "time": time.strftime("%F %T"),
                }
                if prof is not None:
                    row["profile"] = {
                        "gen_tok_s": round(prof.gen_tok_s, 3),
                        "draft_s": round(prof.draft_s, 3),
                        "verify_s": round(prof.verify_s, 3),
                        "rollback_s": round(prof.rollback_s, 3),
                        "other_s": round(prof.gen_s - prof.draft_s - prof.verify_s - prof.rollback_s, 3),
                        "verify_ms_per_round": round(1e3 * prof.verify_s / max(prof.rounds, 1), 2),
                    }
                rows.append(row)
                acc = f" acc {r0.draft_accepted}/{r0.draft_n}" if r0.draft_n else ""
                print(f"[{cfg['method']}/{row['config']}] {p['id']:<10} gen {row['gen_tok_s']:.2f} tok/s "
                      f"n={r0.n_gen}{acc} match={matches}", flush=True)
            if write:
                out = ROOT / "results" / f"mlx_{cfg['method']}.jsonl"
                with open(out, "a") as f:
                    for row in rows:
                        f.write(json.dumps(row) + "\n")
            mean = statistics.mean(r["gen_tok_s"] for r in rows)
            print(f"== {spec}: mean gen {mean:.2f} tok/s, all match: "
                  f"{all(r['matches_baseline'] for r in rows)}", flush=True)
            all_rows += rows
        return all_rows


# ------------------------------------------------------- verify cost vs k
def verify_cost(runner: Runner, Ts=range(1, 18), reps=7, modes=("replay", "capture", "reforward"),
                write=True):
    """Time one target pass over T tokens (+ argmax) and the rollback after it.

    The cache is primed with a real prompt (~0.5k tokens); each measurement
    rolls the cache back to that point so every pass starts at the same
    position. Rollback is timed keeping half of the T tokens.
    """
    spec, tok = runner.spec, runner.tokenizer
    prompts = load_prompts()
    ids = encode_chat(tok, prompts[0]["prompt"])
    pool = mx.array((ids * 4)[:64], mx.uint32)
    rows = []

    # stock mlx-lm single-token decode step, for reference
    cache = runner.model.make_cache()
    mx.eval(runner.model(mx.array(ids, mx.uint32)[None], cache=cache))
    snap = snapshot(cache)
    times = []
    for _ in range(reps + 2):
        mx.synchronize()
        t = time.perf_counter()
        mx.eval(mx.argmax(runner.model(pool[None, :1], cache=cache)[0, -1]))
        times.append(time.perf_counter() - t)
        restore(cache, snap)
    stock_ms = 1e3 * statistics.median(times[2:])
    print(f"stock mlx-lm decode step (T=1): {stock_ms:.1f} ms", flush=True)
    del cache, snap

    for mode in modes:
        cache = spec.make_cache(mode)
        spec.prefill(mx.array(ids, mx.uint32), cache)
        snap = snapshot(cache)  # primed position; restored after every sample
        for T in Ts:
            v_times, r_times = [], []
            n_keep = (T + 1) // 2
            for i in range(reps + 2):
                mx.synchronize()
                t = time.perf_counter()
                out = spec.forward(pool[:T], cache)
                mx.eval(mx.argmax(out.logits, axis=-1))
                t_v = time.perf_counter()
                spec.commit(cache, n_keep, out)
                mx.eval([c.state for c in cache])
                t_r = time.perf_counter()
                restore(cache, snap)
                if i >= 2:
                    v_times.append(t_v - t)
                    r_times.append(t_r - t_v)
            row = {
                "track": "B", "experiment": "verify_cost", "rollback": mode, "T": T, "k": T - 1,
                "verify_ms": round(1e3 * statistics.median(v_times), 2),
                "verify_ms_min": round(1e3 * min(v_times), 2),
                "rollback_ms": round(1e3 * statistics.median(r_times), 2),
                "rollback_keep": n_keep,
                "stock_decode_ms": round(stock_ms, 2),
                "ctx": len(ids), "reps": reps, "model": runner.target_name,
                "time": time.strftime("%F %T"),
            }
            rows.append(row)
            print(f"{mode:<9} T={T:<3} verify {row['verify_ms']:7.1f} ms  "
                  f"({row['verify_ms'] / stock_ms:4.2f}x decode)  rollback {row['rollback_ms']:6.2f} ms",
                  flush=True)
        del cache
    if write:
        with open(ROOT / "results" / "mlx_verify_cost.jsonl", "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return rows


# --------------------------------------------------------------- diagnosis
def diagnose(runner: Runner, prompt_ids: List[int], ref: List[int], got: List[int], window: int = 8) -> dict:
    """Explain a spec-vs-greedy mismatch at the first differing token ``d``.

    Recomputes the logits that pick token ``d`` twice: (1) exactly like the
    baseline, one token per step, and (2) with the last ``window`` context
    tokens in one batched pass, as a verify pass would. Reports the logit
    margin between the baseline's token and the speculative one in each. A
    margin of a few hundredths that changes sign is a numerical near-tie; a
    large margin in both means a real bug.
    """
    d = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), None)
    if d is None:
        return {"diverge_at": None, "note": "one output is a prefix of the other"}
    model = runner.model
    ctx = ref[:d]
    split = max(0, d - window)

    def logits_after(batched: bool):
        cache = model.make_cache()
        out = model(mx.array(prompt_ids, mx.uint32)[None], cache=cache)
        mx.eval(out)
        head, tail = ctx[:split], ctx[split:]
        for t in head + ([] if batched else tail):
            out = model(mx.array([[t]], mx.uint32), cache=cache)
            mx.eval(out)
        if batched and tail:
            out = model(mx.array([tail], mx.uint32), cache=cache)
        return out[0, -1].astype(mx.float32)

    single, batched = logits_after(False), logits_after(True)
    a, b = ref[d], got[d]
    top2 = mx.sort(single)[-2:].tolist()
    res = {
        "diverge_at": d,
        "ref_token": a, "spec_token": b,
        "single_margin_ref_minus_spec": round((single[a] - single[b]).item(), 4),
        "batched_margin_ref_minus_spec": round((batched[a] - batched[b]).item(), 4),
        "single_top2_gap": round(top2[1] - top2[0], 4),
        "max_abs_logit_diff": round(mx.abs(single - batched).max().item(), 4),
    }
    res["verdict"] = ("numerical near-tie" if abs(res["single_margin_ref_minus_spec"]) < 0.25
                      else "suspicious: clear margin, look for a bug")
    return res
