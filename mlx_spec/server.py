"""OpenAI-compatible HTTP server on top of mlx_spec's speculative decoding.

    scripts/gpu.py .venv/bin/python -m mlx_spec serve            # MTP k=3, port 8081

Endpoints (response shapes follow llama-server's, incl. ``timings``):

* ``POST /v1/chat/completions``  streaming (SSE) and non-streaming; the chat
  template's ``<think>`` block goes to ``reasoning_content`` (like
  llama-server ``--jinja``); Qwen3-Coder-style XML tool calls -> ``tool_calls``
* ``POST /v1/completions``       raw prompt, no template
* ``GET /v1/models``, ``GET /health``, ``GET /props``

Layers, bottom up:

* ``Engine``: one model, one request at a time. Runs the speculative loop
  (``SpecModel.forward`` / ``commit`` + the drafter interface, same rounds as
  ``decode.speculative_generate``) on a cache that persists across requests.
  **Prompt caching**: the attention layers' KV caches can be trimmed to any
  length, the gated-delta layers' recurrent + conv state can't, so the engine
  snapshots the GDN state (an O(1) list of array references; ~154 MB of
  arrays kept alive per snapshot) at three points of every request: before the
  prompt's final ``<|im_start|>`` (the generation prompt), at the end of the
  prompt, and at the end of the generation. A new request restores the
  longest snapshot that is a prefix of its prompt, trims the KV caches to it,
  and prefills only the rest. Snapshots also keep the logits of their last
  token, so an identical prompt needs no prefill at all. The MTP head's own KV
  cache (one entry per committed (hidden, next token) pair) is trimmed to the
  reused prefix and the missing pairs are fed from stored target hidden
  states, so drafting continues exactly as after a fresh prefill.
* ``TextStream``: streaming detokenizer (mlx-lm's), stop strings (held back
  until they can no longer match), ``<think>`` / ``<tool_call>`` splitting.
* HTTP: stdlib ``ThreadingHTTPServer``; one worker thread owns the model and
  serves a FIFO job queue, so concurrent requests wait their turn.

Sampling: ``temperature == 0`` is greedy. ``temperature > 0`` uses exact
speculative sampling (accept a draft with ``min(1, p/q)``, resample from
``max(p - q, 0)``) where ``p`` is the target distribution after llama.cpp's
sampler chain ``top_k -> top_p -> min_p -> temperature``: the filters are
applied to the target's logits inside the verify step, so the output is
distributed exactly as sampling the target alone with those settings.
Penalties, logit_bias, logprobs are ignored (warned once); grammars /
``response_format`` json are rejected.
"""

import argparse
import json
import math
import os
import queue
import random
import string
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, List, Optional

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from .drafters import HybridDrafter, NoDrafter
from .gdn import LinearCache
from .model import ForwardOutput, SpecModel
from .mtp import MTPDrafter

DEFAULT_MODEL = "mlx-community/Qwen3.8-27B-4bit"
DEFAULT_DRAFTER = "mtp:k=3:vocab=65536"
FINGERPRINT = "mlx_spec"


class RequestError(Exception):
    """Client error -> HTTP 400 with an OpenAI/llama-server style error body."""

    def __init__(self, message, type_="invalid_request_error", code=400, **extra):
        super().__init__(message)
        self.code, self.type, self.extra = code, type_, extra

    def body(self):
        return {"error": {"code": self.code, "message": str(self), "type": self.type, **self.extra}}


_warned = set()


def warn_once(key, msg):
    if key not in _warned:
        _warned.add(key)
        print(f"[serve] warning: {msg}", file=sys.stderr, flush=True)


# =========================================================================
# Sampling


@dataclass
class SamplingParams:
    temperature: float = 0.8  # llama-server defaults
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    seed: Optional[int] = None


def filtered_logits(logits: mx.array, sp: SamplingParams) -> mx.array:
    """[T, V] logits -> [T, V] scaled logits of the sampling distribution.

    llama.cpp's default chain: top_k, then top_p and min_p on the survivors
    (probabilities at temperature 1), then temperature. Filtered tokens get
    -inf. ``softmax`` of the result is the target distribution ``p``.
    """
    x = logits.astype(mx.float32)
    V = x.shape[-1]
    use_k = 0 < sp.top_k < V
    if use_k or sp.top_p < 1.0 or sp.min_p > 0.0:
        cand = mx.topk(x, sp.top_k, axis=-1) if use_k else x
        cand = -mx.sort(-cand, axis=-1)  # descending
        thresh = cand[:, -1:]  # top_k survivor floor (or the global min)
        if sp.top_p < 1.0:
            pc = mx.softmax(cand, axis=-1)
            before = mx.cumsum(pc, axis=-1) - pc  # mass strictly before each candidate
            m = mx.maximum(mx.sum(before < sp.top_p, axis=-1, keepdims=True), 1)
            thresh = mx.maximum(thresh, mx.take_along_axis(cand, m - 1, axis=-1))
        if sp.min_p > 0.0:
            thresh = mx.maximum(thresh, cand[:, :1] + math.log(sp.min_p))
        x = mx.where(x >= thresh, x, -mx.inf)
    return x / sp.temperature


def sample_verify(logits, drafts, q, sp: SamplingParams):
    """``decode._sample_verify`` with the target distribution of ``filtered_logits``.

    Returns lazy ``(accept [k] bool, fix [k], bonus [1])``; draft i is
    accepted w.p. ``min(1, p_i(d_i) / q_i(d_i))`` (``q`` None = one-hot
    proposal), the first rejected position is resampled from
    ``norm(max(p_i - q_i, 0))``, ``bonus`` is sampled from ``p_k``.
    """
    lp = filtered_logits(logits, sp)
    k = drafts.size
    bonus = mx.random.categorical(lp[k:])
    if k == 0:
        return None, None, bonus
    p = mx.softmax(lp[:k], axis=-1)
    rows = mx.arange(k)
    px = p[rows, drafts]
    if q is None:
        ratio = px
        resid = mx.where(mx.arange(p.shape[-1])[None] == drafts[:, None], 0.0, p)
    else:
        ratio = px / q[rows, drafts]
        resid = mx.maximum(p - q, 0.0)
    accept = mx.random.uniform(shape=(k,)) < mx.minimum(ratio, 1.0)
    fix = mx.random.categorical(mx.log(resid))
    return accept, fix, bonus


# =========================================================================
# Engine: generation with a persistent, prefix-reusable cache


@dataclass
class GenParams:
    max_tokens: Optional[int] = None  # None: until the context is full
    sampling: SamplingParams = field(default_factory=SamplingParams)
    cache_prompt: bool = True


@dataclass
class GenResult:
    tokens: List[int]
    n_prompt: int
    n_cached: int
    prompt_s: float
    gen_s: float
    draft_n: int = 0
    draft_accepted: int = 0
    rounds: int = 0
    hit_eos: bool = False
    stopped: bool = False  # on_tokens asked to stop (stop string)
    cancelled: bool = False

    def timings(self):
        n, pn = len(self.tokens), self.n_prompt - self.n_cached
        t = {
            "cache_n": self.n_cached,
            "prompt_n": pn,
            "prompt_ms": 1e3 * self.prompt_s,
            "prompt_per_token_ms": 1e3 * self.prompt_s / pn if pn else 0.0,
            "prompt_per_second": pn / self.prompt_s if pn and self.prompt_s > 0 else 0.0,
            "predicted_n": n,
            "predicted_ms": 1e3 * self.gen_s,
            "predicted_per_token_ms": 1e3 * self.gen_s / n if n else 0.0,
            "predicted_per_second": n / self.gen_s if n and self.gen_s > 0 else 0.0,
        }
        if self.draft_n:
            t["draft_n"] = self.draft_n
            t["draft_n_accepted"] = self.draft_accepted
        return t

    def usage(self):
        n = len(self.tokens)
        return {"completion_tokens": n, "prompt_tokens": self.n_prompt,
                "total_tokens": n + self.n_prompt,
                "prompt_tokens_details": {"cached_tokens": self.n_cached}}


@dataclass
class Snapshot:
    pos: int  # tokens consumed by the target cache
    gdn: list  # (conv window, recurrent state) per LinearCache, in layer order
    logits: mx.array  # [V] logits after tokens[:pos] (predict token pos)


class ContextError(RequestError):
    def __init__(self, n_prompt, n_ctx):
        super().__init__("the request exceeds the available context size, try increasing it",
                         "exceed_context_size_error", n_prompt_tokens=n_prompt, n_ctx=n_ctx)


def _mtp_of(drafter):
    """(MTPDrafter, is_plain) if the drafter uses an MTP head, else (None, False)."""
    if isinstance(drafter, MTPDrafter):
        return drafter, True
    if isinstance(drafter, HybridDrafter) and isinstance(drafter.model, MTPDrafter):
        return drafter.model, False
    return None, False


class Engine:
    """Speculative generation with a cache kept across requests (single session).

    ``generate`` restores the longest snapshot that is a prefix of the new
    prompt (see the module docstring), prefills the rest, then runs the
    speculative rounds, calling ``on_tokens(new_tokens) -> stop?`` after each.
    Not thread-safe: call from one thread.
    """

    def __init__(self, spec: SpecModel, tokenizer, drafter=None, k: int = 3, eos=None,
                 max_context: int = 32768, rollback: str = "replay", prefill_step: int = 2048,
                 max_snapshots: int = 6):
        self.spec, self.tokenizer = spec, tokenizer
        self.drafter = drafter or NoDrafter()
        self.k = k if not isinstance(self.drafter, NoDrafter) else 0
        self.eos = set(eos if eos is not None else tokenizer.eos_token_ids)
        self.max_context, self.rollback = max_context, rollback
        self.prefill_step, self.max_snapshots = prefill_step, max_snapshots
        self.mtp, self.mtp_plain = _mtp_of(self.drafter)
        self.hkind = getattr(self.mtp, "hidden", "normed")
        try:
            self.im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
        except Exception:
            self.im_start = None
        self.reset()

    # ------------------------------------------------------------ session
    def reset(self):
        self.cache = self.spec.make_cache(self.rollback)
        self.tokens: List[int] = []  # consumed by self.cache
        self.H = None  # [len(tokens), hidden] target hidden rows (MTP needs them)
        self.H_parts: list = []
        self.snaps: List[Snapshot] = []
        if self.mtp is not None:
            self.mtp.cache = KVCache()

    def _hidden(self):
        if self.H_parts:
            parts = ([self.H] if self.H is not None else []) + self.H_parts
            self.H = mx.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
            self.H_parts = []
            mx.eval(self.H)
        return self.H

    def _add_hidden(self, out: ForwardOutput, n: int):
        if self.mtp is not None:
            self.H_parts.append((out.normed if self.hkind == "normed" else out.hidden)[:n])

    def _snapshot(self, logits):
        pos = len(self.tokens)
        if any(s.pos == pos for s in self.snaps):
            return
        gdn = [(c[0], c[1]) for c in self.cache if isinstance(c, LinearCache)]
        logits = logits.astype(mx.float32)  # a copy: a row view would pin the pass's [T, V] logits
        mx.eval(logits, gdn)
        self.snaps.append(Snapshot(pos, gdn, logits))
        self._trim_snaps()

    def _trim_snaps(self):
        while len(self.snaps) > self.max_snapshots:  # drop the shortest
            self.snaps.remove(min(self.snaps, key=lambda s: s.pos))

    def reuse_point(self, prompt: List[int]) -> Optional[Snapshot]:
        """The longest snapshot whose tokens are a prefix of ``prompt``."""
        lcp = 0
        for a, b in zip(self.tokens, prompt):
            if a != b:
                break
            lcp += 1
        ok = [s for s in self.snaps if s.pos <= lcp]
        return max(ok, key=lambda s: s.pos) if ok else None

    def _restore(self, snap: Optional[Snapshot]):
        if snap is None:
            self.reset()
            return
        P = snap.pos
        it = iter(snap.gdn)
        for c in self.cache:
            if isinstance(c, LinearCache):
                c[0], c[1] = next(it)
                c.pending = None
            else:
                assert c.offset == len(self.tokens), (c.offset, len(self.tokens))
                c.trim(c.offset - P)
        self.tokens = self.tokens[:P]
        if self.mtp is not None:
            self.H = self._hidden()[:P]
        self.snaps = [s for s in self.snaps if s.pos <= P]

    def _prefill(self, prompt, P, L, B=None):
        """Prefill ``prompt[P:L]`` in chunks of ``prefill_step`` (the same passes
        as ``SpecModel.prefill``, so numerics match a CLI run) and snapshot at
        ``B`` and ``L``. The state at ``B`` (mid-chunk) is recovered like a
        rollback: replay the recurrence kernel over the chunk's first tokens
        (bitwise the state the pass had there), conv window sliced from the
        recorded input; no extra chunk boundary. Returns the last logits."""
        for s in range(P, L, self.prefill_step):
            e = min(s + self.prefill_step, L)
            out = self.spec.forward(mx.array(prompt[s:e], mx.uint32), self.cache)
            mid = None
            if B is not None and s < B < e:
                mid = self._mid_state(B - s)
            self.spec.commit(self.cache, e - s)
            mx.eval(out.logits, out.normed, [c.state for c in self.cache])
            if mid is not None:
                n = B - s
                pos = len(self.tokens) + n
                row = out.logits[n - 1].astype(mx.float32)
                mx.eval(row, mid)
                self.snaps.append(Snapshot(pos, mid, row))
            self._add_hidden(out, e - s)
            self.tokens += prompt[s:e]
            if e == B:
                self._snapshot(out.logits[-1])
        self._snapshot(out.logits[-1])
        self._trim_snaps()
        return out.logits[-1]

    def _mid_state(self, n):
        """(conv, state) per GDN layer after the first ``n`` tokens of the
        pending (uncommitted) pass, as ``LinearCache.commit`` would compute it."""
        from .gdn import recurrence

        res = []
        for c in self.cache:
            if not isinstance(c, LinearCache):
                continue
            p = c.pending
            n_conv = p["conv_input"].shape[1] - p["T"]
            conv = mx.contiguous(p["conv_input"][:, n : n + n_conv, :])
            if "states" in p:  # capture mode
                st = p["states"][:, n - 1]
            else:
                _, st = recurrence(**{k: p[k][:, :n] for k in ("q", "k", "v", "g", "beta")}, state=p["state0"])
            res.append((conv, st))
        return res

    def _start_drafter(self, prompt, P, last_logits):
        L = len(prompt)
        if self.mtp_plain:
            # MTP KV entry i = pair (H_i, x_{i+1}); valid while both are in the reused prefix
            d, H = self.mtp, self._hidden()
            keep = min(d.cache.offset, max(P - 1, 0))
            d.cache.trim(d.cache.offset - keep)
            if keep < L - 1:
                toks = mx.array(prompt[keep + 1 : L], mx.uint32)
                d.head(d.embed(toks[None]), H[None, keep : L - 1], d.cache)
                mx.eval(d.cache.state)
            d.pend_h, d.pend_tok = H[L - 1 : L], []
            d.n_chain, d.consumed = 0, False
            return
        H = self._hidden() if self.mtp is not None else None
        pre = ForwardOutput(mx.array(prompt, mx.uint32), last_logits[None], H, H)
        self.drafter.start(list(prompt), pre)

    def _pick(self, logits, sp: SamplingParams) -> int:
        if sp.temperature <= 0:
            return mx.argmax(logits, axis=-1).item()
        return mx.random.categorical(filtered_logits(logits[None], sp))[0].item()

    # ----------------------------------------------------------- generate
    def generate(self, prompt: List[int], params: GenParams,
                 on_tokens: Callable[[List[int]], bool] = lambda t: False,
                 cancelled: Callable[[], bool] = lambda: False) -> GenResult:
        try:
            return self._generate(prompt, params, on_tokens, cancelled)
        except BaseException:
            self.reset()  # the cache may be half-updated
            raise

    def _generate(self, prompt, params, on_tokens, cancelled):
        L = len(prompt)
        if L == 0:
            raise RequestError("empty prompt")
        if L >= self.max_context:
            raise ContextError(L, self.max_context)
        budget = self.max_context - L
        max_new = budget if params.max_tokens is None or params.max_tokens < 0 else min(params.max_tokens, budget)
        sp = params.sampling
        sample = sp.temperature > 0
        if sp.seed is not None and sp.seed >= 0:
            mx.random.seed(sp.seed)

        mx.synchronize()
        t0 = time.perf_counter()
        snap = self.reuse_point(prompt) if params.cache_prompt else None
        self._restore(snap)
        P = len(self.tokens)
        # Prefill the rest, also snapshotting before the generation prompt (the
        # last <|im_start|>): the next turn re-renders this assistant turn and
        # may tokenize its start differently.
        B = max((i for i in range(P, L) if prompt[i] == self.im_start), default=None)
        last_logits = snap.logits if snap is not None and snap.pos == L else None
        if P < L:
            last_logits = self._prefill(prompt, P, L, B if B is not None and P < B else None)
        res = GenResult([], L, P, 0.0, 0.0)
        if max_new <= 0:
            res.prompt_s = time.perf_counter() - t0
            return res
        last = self._pick(last_logits, sp)
        self._start_drafter(prompt, P, last_logits)
        t1 = time.perf_counter()
        res.prompt_s = t1 - t0

        res.tokens = [last]
        res.hit_eos = last in self.eos
        res.stopped = bool(on_tokens([last]))
        done = res.hit_eos or res.stopped or max_new <= 1
        end_logits = None
        while not done:
            if cancelled():
                res.cancelled = True
                break
            kk = min(self.k, max_new - len(res.tokens) - 1)
            drafts = self.drafter.draft(last, kk).astype(mx.uint32)
            inp = mx.concatenate([mx.array([last], mx.uint32), drafts])
            out = self.spec.forward(inp, self.cache)
            if sample:
                q = self.drafter.draft_probs() if hasattr(self.drafter, "draft_probs") else None
                accept, fix, bonus = sample_verify(out.logits, drafts, q, sp)
                mx.eval([a for a in (accept, fix, bonus, drafts) if a is not None])
                drafts = drafts.tolist()
                acc = accept.tolist() if accept is not None else []
                n = acc.index(False) if False in acc else len(drafts)
                nxt = fix[n].item() if n < len(drafts) else bonus.item()
            else:
                target = mx.argmax(out.logits, axis=-1)
                mx.eval(target, drafts)
                target, drafts = target.tolist(), drafts.tolist()
                n = 0
                while n < len(drafts) and drafts[n] == target[n]:
                    n += 1
                nxt = target[n]
            new = drafts[:n] + [nxt]
            n_keep = n + 1
            j = next((i for i, t in enumerate(new) if t in self.eos), None)
            if j is not None:
                # stop at EOS; don't let the cache consume accepted drafts after it
                new, n_keep, res.hit_eos = new[: j + 1], j + 1, True
            rerun = self.spec.commit(self.cache, n_keep, out)
            o = rerun or out
            self.drafter.commit(o, n_keep - 1)
            self.tokens += [last] + drafts[: n_keep - 1]
            self._add_hidden(o, n_keep)
            end_logits = o.logits[n_keep - 1]

            res.rounds += 1
            res.draft_n += len(drafts)
            res.draft_accepted += n
            new = new[: max_new - len(res.tokens)]
            res.tokens += new
            last = nxt
            if on_tokens(new):
                res.stopped = True
            done = res.hit_eos or res.stopped or len(res.tokens) >= max_new
        if end_logits is not None:
            self._snapshot(end_logits)
        self._hidden()
        res.gen_s = time.perf_counter() - t1
        return res


# =========================================================================
# Text: detokenize, stop strings, reasoning / tool-call split


def _partial_suffix(text: str, tags) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of a tag."""
    best = 0
    for tag in tags:
        for n in range(min(len(tag) - 1, len(text)), best, -1):
            if text.endswith(tag[:n]):
                best = n
                break
    return best


def _call_id():
    return "call_" + "".join(random.choices(string.ascii_letters + string.digits, k=24))


class TextStream:
    """Turns generated token ids into reasoning / content / tool-call deltas.

    * text: mlx-lm's streaming detokenizer (complete UTF-8 characters only);
      EOS ids are dropped.
    * stop strings: searched in the whole generated text (reasoning
      included, as llama-server does); text that could still become a stop
      string is held back; on a match the text is cut before it.
    * ``think``: ``"open"`` (the prompt ends inside ``<think>``) or
      ``"closed"``; reasoning is the text up to ``</think>`` with leading
      whitespace skipped, content starts after the whitespace following
      ``</think>`` (llama-server's Qwen3 parser). ``None``: plain text.
    * tools: text between the tokenizer's tool-call markers is parsed with
      mlx-lm's tool parser (Qwen3-Coder XML) into OpenAI ``tool_calls``.
    """

    def __init__(self, tokenizer, stop=(), think=None, tools=None, eos=()):
        self.detok = tokenizer.detokenizer
        if hasattr(self.detok, "_maybe_trim_space"):
            # mlx-lm drops a leading space of the first segment (an SPM
            # convention); we want decode(tokens) exactly.
            self.detok._maybe_trim_space = lambda s: s
        self.eos = set(eos)
        self.stop = [s for s in (stop or []) if s]
        self.tools = tools
        self.tc_start = getattr(tokenizer, "tool_call_start", None) or "<tool_call>"
        self.tc_end = getattr(tokenizer, "tool_call_end", None) or "</tool_call>"
        self.tool_parser = getattr(tokenizer, "tool_parser", None)
        self.think = think
        self.state = {"open": "reason_start", "closed": "content_start", None: "content"}[think]
        self.text = ""  # generated text (cut at a stop string)
        self.sent = 0  # chars of self.text handed to the parser
        self.buf = ""
        self.finished = False  # a stop string matched
        self.stop_word = None
        self.reasoning, self.content, self.tool_calls = "", "", []

    def add(self, tokens) -> List[dict]:
        if self.finished:
            return []
        for t in tokens:
            if t not in self.eos:
                self.detok.add_token(t)
        return self._process(self.detok.last_segment, final=False)

    def finish(self) -> List[dict]:
        if self.finished:
            return []
        self.detok.finalize()
        return self._process(self.detok.last_segment, final=True)

    def _process(self, seg, final):
        self.text += seg
        cut = None
        for s in self.stop:
            i = self.text.find(s, max(0, self.sent - len(s) + 1))
            if i >= 0 and (cut is None or i < cut):
                cut, self.stop_word = i, s
        if cut is not None:
            self.text, self.finished, final = self.text[:cut], True, True
            end = cut
        elif final:
            end = len(self.text)
        else:
            end = len(self.text) - _partial_suffix(self.text, self.stop)
        end = max(end, self.sent)
        new, self.sent = self.text[self.sent : end], end
        return self._parse(new, final)

    def _parse(self, new, final):
        self.buf += new
        r, c, calls = "", "", []
        tools = self.tools is not None
        while True:
            st = self.state
            if st in ("reason_start", "content_start"):
                stripped = self.buf.lstrip()
                if not stripped:
                    self.buf = ""
                    break
                self.buf = stripped
                if st == "reason_start":
                    self.state = "reasoning"
                elif self.think and "<think>".startswith(self.buf) and not final:
                    break  # might be the model opening a think block itself
                elif self.think and self.buf.startswith("<think>"):
                    self.buf, self.state = self.buf[len("<think>"):], "reason_start"
                else:
                    self.state = "content"
                continue
            if st == "reasoning":
                ends = ["</think>"] + ([self.tc_start] if tools else [])
                hits = [(self.buf.find(e), e) for e in ends if e in self.buf]
                if hits:
                    i, e = min(hits)
                    r += self.buf[:i]
                    if e == "</think>":
                        self.buf, self.state = self.buf[i + len(e):], "content_start"
                    else:
                        self.buf, self.state = self.buf[i:], "content"
                    continue
                keep = 0 if final else _partial_suffix(self.buf, ends)
                r += self.buf[: len(self.buf) - keep]
                self.buf = self.buf[len(self.buf) - keep:]
                break
            if st == "content":
                if tools and self.tc_start in self.buf:
                    i = self.buf.index(self.tc_start)
                    c += self.buf[:i]
                    self.buf, self.state = self.buf[i + len(self.tc_start):], "tool"
                    continue
                keep = 0 if final or not tools else _partial_suffix(self.buf, [self.tc_start])
                c += self.buf[: len(self.buf) - keep]
                self.buf = self.buf[len(self.buf) - keep:]
                break
            if st == "tool":
                if self.tc_end in self.buf or final:
                    i = self.buf.find(self.tc_end)
                    body = self.buf if i < 0 else self.buf[:i]
                    self.buf = "" if i < 0 else self.buf[i + len(self.tc_end):]
                    call = self._parse_call(body)
                    if call is None:
                        c += self.tc_start + body + (self.tc_end if i >= 0 else "")
                    else:
                        call["index"] = len(self.tool_calls) + len(calls)
                        calls.append(call)
                    self.state = "content_start"
                    if i < 0:
                        break
                    continue
                break
        self.reasoning += r
        self.content += c
        self.tool_calls += calls
        return [d for d in [{"reasoning": r, "content": c, "tool_calls": calls}]
                if d["reasoning"] or d["content"] or d["tool_calls"]]

    def _parse_call(self, body):
        try:
            p = self.tool_parser(body.strip(), self.tools)
            args = p["arguments"]
            return {"id": _call_id(), "type": "function",
                    "function": {"name": p["name"],
                                 "arguments": args if isinstance(args, str)
                                 else json.dumps(args, ensure_ascii=False)}}
        except Exception as e:
            warn_once(("toolparse", str(e)[:40]), f"could not parse tool call ({e}); returned as text")
            return None


# =========================================================================
# Requests: parsing and prompt rendering

IGNORED = {  # parameter -> value that means "off"
    "presence_penalty": 0.0, "frequency_penalty": 0.0, "repeat_penalty": 1.0,
    "logit_bias": None, "logprobs": None, "top_logprobs": None, "typical_p": 1.0,
    "mirostat": 0, "dry_multiplier": 0.0, "xtc_probability": 0.0,
}
REJECTED = ("grammar", "json_schema")
EFFORT_MAP = {"high": "xhigh", "minimal": "low"}


def parse_gen_params(body: dict, defaults: SamplingParams) -> GenParams:
    def num(name, default, cast=float, lo=None):
        v = body.get(name, default)
        if v is None:
            return default
        try:
            v = cast(v)
        except (TypeError, ValueError):
            raise RequestError(f"'{name}' must be a number")
        if lo is not None and v < lo:
            raise RequestError(f"'{name}' must be >= {lo}")
        return v

    n = body.get("n")
    if n not in (None, 1):
        raise RequestError("only n=1 is supported")
    for k in REJECTED:
        if body.get(k):
            raise RequestError(f"'{k}' is not supported by this server")
    rf = body.get("response_format")
    if rf and (rf.get("type") if isinstance(rf, dict) else rf) not in ("text", None):
        raise RequestError("response_format other than text is not supported by this server")
    for k, off in IGNORED.items():
        v = body.get(k)
        if v is not None and v != off and v is not False:
            warn_once(k, f"'{k}' is not supported and is ignored")
    mt = body.get("max_completion_tokens", body.get("max_tokens", body.get("n_predict")))
    if mt is not None:
        try:
            mt = int(mt)
        except (TypeError, ValueError):
            raise RequestError("'max_tokens' must be an integer")
    seed = body.get("seed")
    sp = SamplingParams(
        temperature=num("temperature", defaults.temperature, lo=0.0),
        top_k=num("top_k", defaults.top_k, int),
        top_p=num("top_p", defaults.top_p),
        min_p=num("min_p", defaults.min_p, lo=0.0),
        seed=int(seed) if seed is not None else None,
    )
    if not 0.0 < sp.top_p <= 1.0:
        sp.top_p = 1.0
    if sp.min_p >= 1.0:
        sp.min_p = 0.0
    return GenParams(max_tokens=None if mt is None or mt < 0 else mt, sampling=sp,
                     cache_prompt=bool(body.get("cache_prompt", True)))


def parse_stop(body) -> List[str]:
    s = body.get("stop")
    if s is None:
        return []
    if isinstance(s, str):
        return [s]
    if isinstance(s, list) and all(isinstance(x, str) for x in s):
        return s
    raise RequestError("'stop' must be a string or a list of strings")


def _norm_content(content):
    if content is None or isinstance(content, str):
        return content or ""
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append({"type": "text", "text": p})
            elif isinstance(p, dict) and p.get("type") == "text":
                parts.append({"type": "text", "text": p.get("text", "")})
            else:
                raise RequestError(f"content part type {p.get('type') if isinstance(p, dict) else p!r} "
                                   "is not supported (text only)")
        return parts
    raise RequestError("message content must be a string or a list of parts")


def normalize_messages(messages) -> list:
    if not isinstance(messages, list) or not messages:
        raise RequestError("'messages' must be a non-empty list")
    out = []
    for m in messages:
        if not isinstance(m, dict) or "role" not in m:
            raise RequestError("each message needs a 'role'")
        m = dict(m)
        if m["role"] == "developer":
            m["role"] = "system"
        m["content"] = _norm_content(m.get("content"))
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                fn = tc.get("function", tc)
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        raise RequestError("tool_call arguments must be a JSON object")
                calls.append(tc)
            m["tool_calls"] = calls
        out.append(m)
    return out


def render_chat(tokenizer, body: dict):
    """-> (prompt ids, think mode for TextStream, tools or None)."""
    messages = normalize_messages(body.get("messages"))
    tools = body.get("tools") or None
    tool_choice = body.get("tool_choice")
    if tools is not None:
        if getattr(tokenizer, "tool_parser", None) is None:
            raise RequestError("this model's tokenizer has no tool-call parser; 'tools' is not supported")
        if tool_choice not in (None, "auto", "none"):
            raise RequestError("only tool_choice 'auto' or 'none' is supported")
        if tool_choice == "none":
            tools = None
    kwargs = dict(body.get("chat_template_kwargs") or {})
    effort = body.get("reasoning_effort")
    if effort is not None and "reasoning_effort" not in kwargs:
        if effort == "none":
            kwargs.setdefault("enable_thinking", False)
        else:
            kwargs["reasoning_effort"] = EFFORT_MAP.get(effort, effort)
    if kwargs.get("enable_thinking") is False:
        kwargs.pop("reasoning_effort", None)
    try:
        text = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                             tokenize=False, **kwargs)
    except Exception as e:  # jinja raise_exception, bad structure
        raise RequestError(f"chat template error: {e}")
    ids = tokenizer.encode(text, add_special_tokens=False)
    think = "open" if text.rstrip().endswith("<think>") else "closed"
    return ids, think, tools


def render_completion(tokenizer, body: dict):
    p = body.get("prompt")
    if isinstance(p, list) and len(p) == 1:
        p = p[0]
    if isinstance(p, str):
        return tokenizer.encode(p, add_special_tokens=False)
    if isinstance(p, list) and p and all(isinstance(t, int) for t in p):
        return p
    raise RequestError("'prompt' must be a string, a list of token ids, or a one-element list")


# =========================================================================
# Worker: one thread owns the model and serves jobs in FIFO order


@dataclass
class Job:
    prompt: List[int]
    params: GenParams
    stop: List[str]
    think: Optional[str]
    tools: Optional[list]
    events: "queue.Queue" = field(default_factory=queue.Queue)
    cancelled: bool = False
    t_submit: float = field(default_factory=time.perf_counter)


class Worker:
    def __init__(self, build: Callable[[], Engine], log=True):
        self.jobs: "queue.Queue[Job]" = queue.Queue()
        self.engine: Optional[Engine] = None
        self.error = None
        self.ready = threading.Event()
        self.log = log
        self.thread = threading.Thread(target=self._run, args=(build,), daemon=True, name="mlx-worker")
        self.thread.start()

    def submit(self, job: Job) -> Job:
        self.jobs.put(job)
        return job

    def _run(self, build):
        try:
            self.engine = build()
        except BaseException as e:
            self.error = e
            traceback.print_exc()
            self.ready.set()
            return
        self.ready.set()
        while True:
            job = self.jobs.get()
            if job is None:
                return
            self._serve(job)

    def _serve(self, job: Job):
        eng = self.engine
        ts = TextStream(eng.tokenizer, job.stop, job.think, job.tools, eos=eng.eos)

        def on_tokens(toks):
            for d in ts.add(toks):
                job.events.put(("delta", d))
            return ts.finished

        try:
            res = eng.generate(job.prompt, job.params, on_tokens, lambda: job.cancelled)
            for d in ts.finish():
                job.events.put(("delta", d))
        except RequestError as e:
            job.events.put(("error", e))
            return
        except Exception as e:
            traceback.print_exc()
            job.events.put(("error", RequestError(f"{type(e).__name__}: {e}", "server_error", 500)))
            return
        if ts.tool_calls and (res.hit_eos or ts.finished):
            reason = "tool_calls"
        else:
            reason = "stop" if res.hit_eos or ts.finished else "length"
        job.events.put(("done", (res, reason, ts)))
        if self.log:
            t = res.timings()
            wait = time.perf_counter() - job.t_submit - res.prompt_s - res.gen_s
            acc = (f", accepted {res.draft_accepted}/{res.draft_n}" if res.draft_n else "")
            print(f"[serve] prompt {res.n_prompt} tok (cached {res.n_cached}, new {t['prompt_n']} "
                  f"@ {t['prompt_per_second']:.1f} t/s, {res.prompt_s:.2f}s) | gen {len(res.tokens)} tok "
                  f"@ {t['predicted_per_second']:.2f} t/s{acc} | {reason}"
                  f"{' (client gone)' if res.cancelled else ''}{f' | queued {wait:.1f}s' if wait > 0.5 else ''}",
                  file=sys.stderr, flush=True)


# =========================================================================
# HTTP


def _now():
    return int(time.time())


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "mlx_spec"
    app: "App" = None  # set by make_server

    def log_message(self, fmt, *args):
        if self.app.verbose:
            super().log_message(fmt, *args)

    # ------------------------------------------------------------ helpers
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def send_json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def send_error_body(self, e: RequestError):
        self.send_json(e.code, e.body())

    def _chunk(self, data: bytes):
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def sse(self, obj):
        self._chunk(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")

    def path_is(self, *paths):
        return self.path.split("?", 1)[0].rstrip("/") in paths

    # ------------------------------------------------------------ routes
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        app = self.app
        if self.path_is("/health", "/v1/health"):
            if app.worker.error is not None:
                return self.send_json(500, {"error": {"code": 500, "message": f"model failed to load: {app.worker.error}", "type": "server_error"}})
            if not app.worker.ready.is_set():
                return self.send_json(503, {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}})
            return self.send_json(200, {"status": "ok"})
        if self.path_is("/v1/models", "/models"):
            return self.send_json(200, {"object": "list", "data": [
                {"id": app.model_name, "object": "model", "created": app.created, "owned_by": "mlx_spec",
                 "meta": {"drafter": app.drafter_name, "n_ctx": app.max_context}}]})
        if self.path_is("/props"):
            return self.send_json(200, {"model_path": app.model_name, "total_slots": 1,
                                        "default_generation_settings": {"n_ctx": app.max_context},
                                        "build_info": FINGERPRINT})
        self.send_json(404, {"error": {"code": 404, "message": "File Not Found", "type": "not_found_error"}})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            if self.path_is("/v1/chat/completions", "/chat/completions"):
                kind = "chat"
            elif self.path_is("/v1/completions", "/completions"):
                kind = "text"
            else:
                return self.send_json(404, {"error": {"code": 404, "message": "File Not Found", "type": "not_found_error"}})
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError as e:
                raise RequestError(f"invalid JSON: {e}")
            if not isinstance(body, dict):
                raise RequestError("request body must be a JSON object")
            self.handle_generate(kind, body)
        except RequestError as e:
            self.send_error_body(e)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def handle_generate(self, kind, body):
        app = self.app
        if not app.worker.ready.is_set() or app.worker.engine is None:
            raise RequestError("Loading model", "unavailable_error", 503)
        eng = app.worker.engine
        params = parse_gen_params(body, app.defaults)
        stop = parse_stop(body)
        if kind == "chat":
            ids, think, tools = render_chat(eng.tokenizer, body)
        else:
            ids, think, tools = render_completion(eng.tokenizer, body), None, None
        if len(ids) >= eng.max_context:
            raise ContextError(len(ids), eng.max_context)
        job = app.worker.submit(Job(ids, params, stop, think, tools))
        cid = ("chatcmpl-" if kind == "chat" else "cmpl-") + "".join(
            random.choices(string.ascii_letters + string.digits, k=32))
        if body.get("stream"):
            so = body.get("stream_options") or {}
            self.stream(kind, job, cid, bool(so.get("include_usage")), body)
        else:
            self.complete(kind, job, cid)

    def _base(self, cid, obj):
        return {"created": _now(), "id": cid, "model": self.app.model_name,
                "system_fingerprint": FINGERPRINT, "object": obj}

    def complete(self, kind, job, cid):
        while True:
            ev, val = job.events.get()
            if ev == "error":
                raise val
            if ev == "done":
                break
        res, reason, ts = val
        if kind == "chat":
            msg = {"role": "assistant", "content": ts.content}
            if ts.reasoning:
                msg["reasoning_content"] = ts.reasoning
            if ts.tool_calls:
                msg["tool_calls"] = [{k: v for k, v in c.items() if k != "index"} for c in ts.tool_calls]
                if not ts.content:
                    msg["content"] = None
            choice = {"finish_reason": reason, "index": 0, "message": msg}
            out = {"choices": [choice], **self._base(cid, "chat.completion")}
        else:
            choice = {"text": ts.content, "index": 0, "logprobs": None, "finish_reason": reason}
            out = {"choices": [choice], **self._base(cid, "text_completion")}
        out["usage"] = res.usage()
        out["timings"] = res.timings()
        self.send_json(200, out)

    def stream(self, kind, job, cid, include_usage, body):
        # Wait for the first event so request errors still get a proper 400.
        first = job.events.get()
        if first[0] == "error":
            raise first[1]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self._cors()
        self.end_headers()
        obj = "chat.completion.chunk" if kind == "chat" else "text_completion"
        per_token = bool(body.get("timings_per_token"))

        def chunk(delta=None, reason=None, text=None):
            if kind == "chat":
                ch = {"finish_reason": reason, "index": 0, "delta": delta or {}}
            else:
                ch = {"text": text or "", "index": 0, "logprobs": None, "finish_reason": reason}
            return {"choices": [ch], **self._base(cid, obj)}

        try:
            if kind == "chat":
                self.sse(chunk({"role": "assistant", "content": None}))
            ev = first
            while True:
                kind_ev, val = ev
                if kind_ev == "error":
                    self.sse({"error": val.body()["error"]})
                    break
                if kind_ev == "delta":
                    if kind == "chat":
                        d = {}
                        if val["reasoning"]:
                            d["reasoning_content"] = val["reasoning"]
                        if val["content"]:
                            d["content"] = val["content"]
                        if d:
                            self.sse(chunk(d))
                        for c in val["tool_calls"]:
                            self.sse(chunk({"tool_calls": [c]}))
                    elif val["content"]:
                        self.sse(chunk(text=val["content"]))
                if kind_ev == "done":
                    res, reason, ts = val
                    last = chunk({} if kind == "chat" else None, reason)
                    if include_usage:
                        self.sse(last)
                        last = {"choices": [], **self._base(cid, obj), "usage": res.usage()}
                    last["timings"] = res.timings()
                    self.sse(last)
                    break
                ev = job.events.get()
            self._chunk(b"data: [DONE]\n\n")
            self._chunk(b"")
        except (BrokenPipeError, ConnectionResetError, OSError):
            job.cancelled = True  # the worker stops after the current round
            self.close_connection = True


class App:
    def __init__(self, worker: Worker, model_name: str, drafter_name: str, max_context: int,
                 defaults: SamplingParams, verbose=False):
        self.worker, self.model_name, self.drafter_name = worker, model_name, drafter_name
        self.max_context, self.defaults, self.verbose = max_context, defaults, verbose
        self.created = _now()


class _Server(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):  # client went away: quiet
            super().handle_error(request, client_address)


def make_server(app: App, host: str, port: int) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"app": app})
    srv = _Server((host, port), handler)
    srv.daemon_threads = True
    return srv


# =========================================================================
# CLI


def add_args(p: argparse.ArgumentParser):
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"MLX model dir or HF id (default {DEFAULT_MODEL})")
    p.add_argument("--drafter", default=DEFAULT_DRAFTER,
                   help="mlx_spec config string (mtp:k=3:vocab=65536, ngram:k=3, draft:k=3, "
                        "mtp-hybrid:k=3, ...) or 'none'")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--max-context", type=int, default=32768, help="prompt + generated tokens")
    p.add_argument("--alias", default=None, help="model id reported by /v1/models")
    p.add_argument("--snapshots", type=int, default=6, help="prompt-cache snapshots kept (~154 MB each)")
    p.add_argument("--temp", type=float, default=0.8, help="default temperature (llama-server: 0.8)")
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--min-p", type=float, default=0.05)
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--verbose", action="store_true", help="log every HTTP request")


def _eos_ids(model, tokenizer):
    eos = set(tokenizer.eos_token_ids)
    try:
        from mlx_lm.utils import _download

        cfg = json.loads((Path(_download(model)) / "generation_config.json").read_text())
        e = cfg.get("eos_token_id")
        eos |= set(e if isinstance(e, list) else [e] if e is not None else [])
    except Exception:
        pass
    return eos


def build_engine(model: str, drafter: str, max_context: int, snapshots: int = 6,
                 warmup: bool = True) -> Engine:
    """Load the target (+ drafter) with mlx_spec's benchmark Runner and wrap it."""
    from .bench import Runner, parse_config

    runner = Runner(model)
    if drafter in ("none", "greedy", "spec-none"):
        d, k, rollback = NoDrafter(), 0, "replay"
    else:
        cfg = parse_config(drafter)
        d = runner.make_drafter(cfg)
        k, rollback = int(cfg.get("k", 3)), cfg.get("rollback", "replay")
    eng = Engine(runner.spec, runner.tokenizer, d, k=k, eos=_eos_ids(model, runner.tokenizer),
                 max_context=max_context, rollback=rollback, max_snapshots=snapshots)
    if warmup:
        ids = runner.tokenizer.encode("<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n",
                                      add_special_tokens=False)
        eng.generate(ids, GenParams(max_tokens=8, sampling=SamplingParams(temperature=0.0)))
        eng.generate(ids, GenParams(max_tokens=8, sampling=SamplingParams(temperature=0.7, seed=0)))
        eng.reset()
    return eng


def serve(a):
    t0 = time.time()
    if os.environ.get("SPECDEC_CPU") == "1":  # tiny-model testing without the GPU
        mx.set_default_device(mx.cpu)
        info = mx.device_info()
        mx.device_info = lambda: {"max_recommended_working_set_size": 0, **info}  # Runner wires memory
    name = a.alias or a.model.rstrip("/").split("/")[-1]
    print(f"[serve] loading {a.model} (drafter {a.drafter}) ...", file=sys.stderr, flush=True)
    worker = Worker(lambda: build_engine(a.model, a.drafter, a.max_context, a.snapshots,
                                         warmup=not a.no_warmup))
    defaults = SamplingParams(a.temp, a.top_k, a.top_p, a.min_p)
    app = App(worker, name, a.drafter, a.max_context, defaults, verbose=a.verbose)
    srv = make_server(app, a.host, a.port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[serve] listening on http://{a.host}:{a.port}/v1 (503 until the model is loaded)",
          file=sys.stderr, flush=True)
    worker.ready.wait()
    if worker.error is not None:
        srv.shutdown()
        sys.exit(1)
    print(f"[serve] ready in {time.time() - t0:.0f}s: model '{name}', drafter {a.drafter}, "
          f"max context {a.max_context}", file=sys.stderr, flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
