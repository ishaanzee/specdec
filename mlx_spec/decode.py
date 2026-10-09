"""Greedy baseline and the speculative decoding loop.

Timing convention (same as llama-server's ``timings``): the prompt phase is
the prefill forward pass plus the first token picked from its logits; the
generation phase starts right after and ends when the last token is known.
``gen_tok_s = n_gen / gen_s`` counts all ``n_gen`` generated tokens, including
the first one, exactly like llama-server's ``predicted_per_second``.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn

from .drafters import Drafter
from .model import SpecModel


@dataclass
class GenResult:
    tokens: List[int]
    n_prompt: int
    prompt_s: float
    gen_s: float
    rounds: int = 0
    draft_n: int = 0  # draft tokens proposed and verified
    draft_accepted: int = 0
    accept_hist: List[int] = field(default_factory=list)  # rounds by n accepted
    # time split (only meaningful with profile=True, which adds syncs)
    draft_s: float = 0.0
    verify_s: float = 0.0
    rollback_s: float = 0.0

    @property
    def n_gen(self):
        return len(self.tokens)

    @property
    def gen_tok_s(self):
        return self.n_gen / self.gen_s if self.gen_s > 0 else float("nan")

    @property
    def prompt_tok_s(self):
        return self.n_prompt / self.prompt_s

    @property
    def accept_rate(self):
        return self.draft_accepted / self.draft_n if self.draft_n else None


def _finish(tokens: List[int], eos: Sequence[int], max_tokens: int) -> tuple[List[int], bool]:
    """Truncate at the first EOS (kept) or at max_tokens; report if done."""
    for i, t in enumerate(tokens):
        if t in eos:
            return tokens[: i + 1], True
    return tokens[:max_tokens], len(tokens) >= max_tokens


def greedy_generate(
    model: nn.Module,
    prompt: List[int],
    max_tokens: int = 256,
    eos: Sequence[int] = (),
    prefill_step: int = 2048,
) -> GenResult:
    """Plain greedy decoding with stock mlx-lm caches, one token per step.

    Same structure as mlx-lm's ``generate_step``: the graph for step i + 1 is
    built and queued (``mx.async_eval``) before we block on token i, so
    Python overhead overlaps GPU work.
    """
    cache = model.make_cache()
    p = mx.array(prompt, mx.uint32)
    mx.synchronize()
    t0 = time.perf_counter()
    # Whole prompt in one pass per chunk (as llama.cpp and SpecModel.prefill
    # do), so baseline and speculative runs start from bitwise-equal caches.
    for s in range(0, p.size, prefill_step):
        logits = model(p[None, s : s + prefill_step], cache=cache)
        mx.eval(logits, [c.state for c in cache])
    y = mx.argmax(logits[0, -1], axis=-1)
    mx.eval(y)
    t1 = time.perf_counter()

    def step(y):
        logits = model(y.reshape(1, 1), cache=cache)
        nxt = mx.argmax(logits[0, -1], axis=-1)
        mx.async_eval(nxt)
        return nxt

    tokens = []
    while True:
        nxt = step(y) if len(tokens) + 1 < max_tokens else None
        tok = y.item()
        tokens.append(tok)
        if tok in eos or nxt is None:
            break
        y = nxt
    t2 = time.perf_counter()
    mx.synchronize()  # drain a possibly queued extra step outside the timing
    return GenResult(tokens, len(prompt), t1 - t0, t2 - t1)


def speculative_generate(
    spec: SpecModel,
    drafter: Drafter,
    prompt: List[int],
    k: int,
    max_tokens: int = 256,
    eos: Sequence[int] = (),
    profile: bool = False,
    rollback: Optional[str] = None,
) -> GenResult:
    """Greedy speculative decoding; output equals greedy decoding of ``spec``.

    Each round: the drafter proposes up to ``k`` tokens following ``last``;
    the target runs ONE pass over ``[last] + drafts`` (T = 1 + len(drafts));
    ``target[i] = argmax(logits[i])`` is the target's choice after
    ``[.., last, d_1 .. d_i]``. Accept the longest prefix with
    ``d_{i+1} == target[i]``, say ``n`` drafts, then emit ``d_1 .. d_n`` plus
    ``target[n]`` (the correction, or the bonus token when all ``k`` were
    accepted). The cache keeps ``n + 1`` of the ``T`` tokens (``last`` and
    the accepted drafts); ``target[n]`` becomes the next ``last``.

    ``profile=True`` synchronises after each phase so ``draft_s``,
    ``verify_s`` and ``rollback_s`` are attributable; leave it off for tok/s.
    """
    cache = spec.make_cache(rollback)
    p = mx.array(prompt, mx.uint32)
    mx.synchronize()
    t0 = time.perf_counter()
    pre = spec.prefill(p, cache)
    last = mx.argmax(pre.logits[-1], axis=-1).item()
    drafter.start(prompt, pre)
    t1 = time.perf_counter()

    res = GenResult([last], len(prompt), t1 - t0, 0.0)
    done = last in eos or max_tokens <= 1
    while not done:
        tic = time.perf_counter()
        kk = min(k, max_tokens - len(res.tokens) - 1)
        drafts = drafter.draft(last, kk)
        if profile:
            mx.eval(drafts)
            t_draft = time.perf_counter()
            res.draft_s += t_draft - tic

        inp = mx.concatenate([mx.array([last], mx.uint32), drafts.astype(mx.uint32)])
        out = spec.forward(inp, cache)
        target = mx.argmax(out.logits, axis=-1)
        mx.eval(target, drafts)
        target, drafts = target.tolist(), drafts.tolist()
        if profile:
            t_verify = time.perf_counter()
            res.verify_s += t_verify - t_draft

        n = 0
        while n < len(drafts) and drafts[n] == target[n]:
            n += 1
        rerun = spec.commit(cache, n + 1, out)
        drafter.commit(rerun or out, n)
        if profile:
            mx.eval([c.state for c in cache])
            res.rollback_s += time.perf_counter() - t_verify

        res.rounds += 1
        res.draft_n += len(drafts)
        res.draft_accepted += n
        if len(drafts):
            res.accept_hist += [0] * (n + 1 - len(res.accept_hist))
            res.accept_hist[n] += 1
        new = drafts[:n] + [target[n]]
        res.tokens, done = _finish(res.tokens + new, eos, max_tokens)
        last = target[n]
    res.gen_s = time.perf_counter() - t1
    return res
