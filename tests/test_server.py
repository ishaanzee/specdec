"""mlx_spec.server on a tiny random-weight qwen3_5 model with the real Qwen3.8
tokenizer (vocab 248320), float32, on the CPU (no GPU lock needed, ~1-2 min):

    .venv/bin/python -m pytest tests/test_server.py -q

Covers request parsing, the text layer (streaming detokenization across
multi-byte tokens, stop strings, <think> and tool-call splitting), exact
speculative sampling with top-k/top-p/min-p, the engine's prompt cache (output
and MTP cache with reuse == fresh prefill), and the HTTP API (shapes,
streaming == non-streaming, errors, queueing of concurrent requests).
"""

import http.client
import json
import os
import random
import threading
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_spec import SpecModel, greedy_generate
from mlx_spec.mtp import MTPDrafter, MTPHead, load_vocab_subset
from mlx_spec.server import (App, ContextError, Engine, GenParams, Job, RequestError, SamplingParams,
                             TextStream, Worker, filtered_logits, make_server, parse_gen_params,
                             parse_stop, render_chat, sample_verify)

from test_tiny import tiny_model

VOCAB = 248320
TARGET = "mlx-community/Qwen3.8-27B-4bit"


@pytest.fixture(scope="module", autouse=True)
def cpu():
    old = mx.default_device()
    if os.environ.get("SPECDEC_GPU") != "1":
        mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


@pytest.fixture(scope="module")
def tok():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from huggingface_hub import snapshot_download
    from mlx_lm.tokenizer_utils import load

    return load(Path(snapshot_download(TARGET, allow_patterns=["*.json", "*.jinja"])))


@pytest.fixture(scope="module")
def model(cpu):
    return tiny_model(vocab=VOCAB)


@pytest.fixture(scope="module")
def head(model):
    mx.random.seed(3)
    h = MTPHead(model.language_model.args)
    mx.eval(h.parameters())
    return h


def make_engine(model, tok, head=None, **kw):
    spec = SpecModel(model)
    d = MTPDrafter(spec, head, vocab=load_vocab_subset("first:65536")) if head is not None else None
    return Engine(spec, tok, d, k=3, eos=set(tok.eos_token_ids) | {248044}, **kw)


def chat_ids(tok, messages, **kw):
    s = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kw)
    return tok.encode(s, add_special_tokens=False)


GREEDY = SamplingParams(temperature=0.0)

# ---------------------------------------------------------------- text layer


def feed(ts, ids, rng):
    """Feed ids in random chunks; return the concatenated deltas."""
    out = {"reasoning": "", "content": "", "tool_calls": []}
    i = 0
    while i < len(ids) and not ts.finished:
        n = rng.randint(1, 4)
        for d in ts.add(ids[i : i + n]):
            for k in out:
                out[k] += d[k]
        i += n
    for d in ts.finish():
        for k in out:
            out[k] += d[k]
    return out


def test_think_split_and_multibyte(tok):
    reasoning = "Let me think about it.\nThe user wants 42 ✓.\n"
    content = "The answer: 42 ✓ 日本語 🦙🦙 naïve — done."
    gen = tok.encode(reasoning + "</think>\n\n" + content, add_special_tokens=False) + [248046]
    # the emoji really are split over several byte-level tokens
    assert len(tok.encode("🦙", add_special_tokens=False)) > 1
    for seed in range(20):
        ts = TextStream(tok, think="open", eos={248046})
        out = feed(ts, gen, random.Random(seed))
        assert out["reasoning"] == reasoning and ts.reasoning == reasoning
        assert out["content"] == content and ts.content == content
        assert "�" not in out["content"]
    # thinking disabled: the model starts with the answer (leading whitespace dropped)
    ts = TextStream(tok, think="closed")
    out = feed(ts, tok.encode("\n" + content, add_special_tokens=False), random.Random(0))
    assert out == {"reasoning": "", "content": content, "tool_calls": []}
    # plain text (completions): exactly decode(tokens), leading space kept
    ids = tok.encode(" hello 🦙 world", add_special_tokens=False)
    ts = TextStream(tok)
    assert feed(ts, ids, random.Random(1))["content"] == tok.decode(ids) == " hello 🦙 world"


def test_stop_strings(tok):
    text = "alpha beta gamma STOP delta"
    ids = tok.encode(text, add_special_tokens=False)
    for seed in range(20):
        rng = random.Random(seed)
        deltas = []
        ts = TextStream(tok, stop=["STOP", "zzz"])
        i = 0
        while i < len(ids) and not ts.finished:
            n = rng.randint(1, 3)
            deltas += [d["content"] for d in ts.add(ids[i : i + n])]
            i += n
        deltas += [d["content"] for d in ts.finish()]
        assert ts.finished and ts.stop_word == "STOP"
        assert "".join(deltas) == "alpha beta gamma "
        assert not any("S" in d for d in deltas)  # the partial stop string was never sent
    # stop string inside the reasoning, and one spanning reasoning tokens
    ids = tok.encode("I think so. END here</think>\n\nanswer", add_special_tokens=False)
    ts = TextStream(tok, stop=["END"], think="open")
    out = feed(ts, ids, random.Random(0))
    assert out["reasoning"] == "I think so. " and out["content"] == ""
    # no stop: everything comes through, incl. text that only looked like a prefix
    ts = TextStream(tok, stop=["STOPPER"])
    assert feed(ts, tok.encode("a STOP b", add_special_tokens=False), random.Random(0))["content"] == "a STOP b"


TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                   "required": ["city"]}}}]


def test_tool_calls(tok):
    gen = ("I should call the tool.\n</think>\n\nLet me check.\n\n<tool_call>\n<function=get_weather>\n"
           "<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>")
    ids = tok.encode(gen, add_special_tokens=False) + [248046]
    for seed in range(5):
        ts = TextStream(tok, think="open", tools=TOOLS, eos={248046})
        out = feed(ts, ids, random.Random(seed))
        assert out["reasoning"] == "I should call the tool.\n"
        assert out["content"].strip() == "Let me check."
        assert len(out["tool_calls"]) == 1
        c = out["tool_calls"][0]
        assert c["type"] == "function" and c["id"].startswith("call_") and c["index"] == 0
        assert c["function"]["name"] == "get_weather"
        assert json.loads(c["function"]["arguments"]) == {"city": "Paris", "days": 3}
    # without tools the markup stays text
    ts = TextStream(tok, think="open")
    out = feed(ts, ids, random.Random(0))
    assert "<tool_call>" in out["content"] and not out["tool_calls"]


# --------------------------------------------------------- request parsing


def test_parse_params():
    d = SamplingParams()
    p = parse_gen_params({}, d)
    assert p.max_tokens is None and p.sampling.temperature == 0.8 and p.cache_prompt
    p = parse_gen_params({"max_tokens": 5, "temperature": 0, "top_p": 0.5, "seed": 7, "cache_prompt": False}, d)
    assert (p.max_tokens, p.sampling.temperature, p.sampling.top_p, p.sampling.seed, p.cache_prompt) == (5, 0.0, 0.5, 7, False)
    assert parse_gen_params({"max_completion_tokens": 9}, d).max_tokens == 9
    assert parse_gen_params({"max_tokens": -1}, d).max_tokens is None
    for bad in ({"n": 2}, {"response_format": {"type": "json_object"}}, {"grammar": "root ::= x"},
                {"temperature": -1}, {"max_tokens": "x"}):
        with pytest.raises(RequestError):
            parse_gen_params(bad, d)
    assert parse_gen_params({"response_format": {"type": "text"}}, d)
    assert parse_stop({"stop": "x"}) == ["x"] and parse_stop({"stop": ["a", "b"]}) == ["a", "b"]
    assert parse_stop({}) == []
    with pytest.raises(RequestError):
        parse_stop({"stop": 3})


def test_render_chat(tok):
    msgs = [{"role": "user", "content": "hi"}]
    ids, think, tools = render_chat(tok, {"messages": msgs})
    s = tok.decode(ids)
    assert think == "open" and s.endswith("<|im_start|>assistant\n<think>\n") and "xhigh" in s
    ids, think, _ = render_chat(tok, {"messages": msgs, "chat_template_kwargs": {"enable_thinking": False}})
    assert think == "closed" and tok.decode(ids).endswith("<think>\n\n</think>\n\n")
    s = tok.decode(render_chat(tok, {"messages": msgs, "reasoning_effort": "low"})[0])
    assert "Reasoning effort is set to low" in s
    s = tok.decode(render_chat(tok, {"messages": msgs, "reasoning_effort": "high"})[0])
    assert "xhigh" in s
    ids, think, _ = render_chat(tok, {"messages": msgs, "reasoning_effort": "none"})
    assert think == "closed"
    with pytest.raises(RequestError):
        render_chat(tok, {"messages": msgs, "reasoning_effort": "bogus"})
    with pytest.raises(RequestError):
        render_chat(tok, {"messages": []})
    with pytest.raises(RequestError):
        render_chat(tok, {"messages": msgs, "tools": TOOLS, "tool_choice": "required"})
    with pytest.raises(RequestError):
        render_chat(tok, {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]})
    # typed text parts, tools, and an assistant tool call with JSON-string arguments in the history
    hist = [{"role": "user", "content": [{"type": "text", "text": "weather?"}]},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"}]
    ids, think, tools = render_chat(tok, {"messages": hist, "tools": TOOLS})
    s = tok.decode(ids)
    assert tools == TOOLS and "<parameter=city>\nParis\n</parameter>" in s and "<tool_response>\nsunny" in s


# ------------------------------------------------------------------ sampling


def test_filtered_logits():
    x = mx.array([[4.0, 3.0, 2.0, 1.0, 0.0, -1.0]])
    fin = lambda sp: np.isfinite(np.array(filtered_logits(x, sp)))[0].tolist()
    assert fin(SamplingParams(1.0, 2, 1.0, 0.0)) == [1, 1, 0, 0, 0, 0]
    p = np.exp(np.arange(4, -2, -1.0)); p /= p.sum()  # 0.636 0.234 0.086 ...
    assert fin(SamplingParams(1.0, 0, 0.7, 0.0)) == [1, 1, 0, 0, 0, 0]
    assert fin(SamplingParams(1.0, 0, 0.5, 0.0)) == [1, 0, 0, 0, 0, 0]
    assert fin(SamplingParams(1.0, 0, 1.0, 0.1)) == [1, 1, 1, 0, 0, 0]  # p >= 0.1 * p_max
    assert fin(SamplingParams(1.0, 0, 1.0, 0.0)) == [1] * 6
    y = np.array(filtered_logits(x, SamplingParams(0.5, 0, 1.0, 0.0)))
    assert np.allclose(y, np.array(x) / 0.5)


@pytest.mark.parametrize("proposal", ["onehot", "dist"])
def test_sample_verify_exact(proposal):
    """The first emitted token (accepted draft or resample) is distributed as the
    filtered target distribution p, whatever the proposal."""
    mx.random.seed(0)
    V = 8
    logits = mx.random.normal((2, V)) * 1.5
    sp = SamplingParams(0.8, 6, 0.9, 0.05)
    p = np.array(mx.softmax(filtered_logits(logits, sp), axis=-1))[0]
    q = mx.softmax(mx.random.normal((1, V)), axis=-1) if proposal == "dist" else None
    N = 6000
    counts = np.zeros(V)
    for i in range(N):
        d = (mx.random.categorical(mx.log(q)) if q is not None else mx.array([2])).astype(mx.uint32)
        acc, fix, bonus = sample_verify(logits, d, q, sp)
        counts[d.item() if acc.item() else fix.item()] += 1
    tv = 0.5 * np.abs(counts / N - p).sum()
    assert tv < 0.03, (tv, counts / N, p)


# -------------------------------------------------------------------- engine


def test_engine_matches_greedy(model, tok, head):
    ids = chat_ids(tok, [{"role": "user", "content": "Tell me a story."}])
    ref = greedy_generate(model, ids, 40).tokens
    for h in (None, head):
        eng = make_engine(model, tok, h, max_context=4096)
        eng.eos = set()  # compare raw token streams
        r = eng.generate(ids, GenParams(40, GREEDY))
        assert r.tokens == ref and r.n_cached == 0
        if h is not None:
            assert r.draft_n > 0


def _mtp_kv(eng):
    c = eng.mtp.cache
    return np.array(c.keys[..., : c.offset, :]), np.array(c.values[..., : c.offset, :])


@pytest.mark.parametrize("use_mtp", [False, True])
def test_prompt_cache_reuse_equals_fresh(model, tok, head, use_mtp):
    """A 3-turn chat with the history re-rendered by the template, a raw-token
    continuation (exact generation-end snapshot), and an identical resend:
    the reuse engine's output and MTP cache equal a fresh prefill's."""
    h = head if use_mtp else None
    reuse = make_engine(model, tok, h, max_context=4096)
    fresh = make_engine(model, tok, h, max_context=4096)
    msgs = []
    prompts = []
    for turn, q in enumerate(["What is 2+2?", "And times 3?", "Explain briefly."]):
        msgs.append({"role": "user", "content": q})
        ids = chat_ids(tok, msgs, enable_thinking=False)
        prompts.append(ids)
        a = reuse.generate(ids, GenParams(24, GREEDY))
        b = fresh.generate(ids, GenParams(24, GREEDY, cache_prompt=False))
        assert a.tokens == b.tokens, turn
        assert (a.draft_n, a.draft_accepted) == (b.draft_n, b.draft_accepted)
        assert b.n_cached == 0
        if turn:
            assert a.n_cached > len(prompts[turn - 1]) - 8, (turn, a.n_cached)  # at least up to the last turn's gen prompt
        if use_mtp:
            for x, y in zip(_mtp_kv(reuse), _mtp_kv(fresh)):
                assert x.shape == y.shape and np.allclose(x, y, atol=1e-4)
        text = tok.decode([t for t in a.tokens if t not in reuse.eos])
        msgs.append({"role": "assistant", "content": text})

    # raw continuation: prompt + generated tokens + more -> reuse from the generation-end snapshot
    p1 = prompts[-1]
    a1 = reuse.generate(p1, GenParams(16, GREEDY))
    p2 = p1 + a1.tokens + tok.encode("<|im_end|>\n<|im_start|>user\nmore<|im_end|>\n<|im_start|>assistant\n",
                                     add_special_tokens=False)
    a2 = reuse.generate(p2, GenParams(16, GREEDY))
    assert a2.n_cached == len(p1) + len(a1.tokens) - 1
    b2 = fresh.generate(p2, GenParams(16, GREEDY, cache_prompt=False))
    assert a2.tokens == b2.tokens
    # identical resend: nothing to prefill (logits kept with the snapshot)
    a3 = reuse.generate(p2, GenParams(16, GREEDY))
    assert a3.n_cached == len(p2) and a3.tokens == b2.tokens
    if use_mtp:
        fresh.generate(p2, GenParams(16, GREEDY, cache_prompt=False))
        for x, y in zip(_mtp_kv(reuse), _mtp_kv(fresh)):
            assert np.allclose(x, y, atol=1e-4)
    # a prompt that diverges early falls back to a fresh prefill and still matches
    p3 = tok.encode("<|im_start|>user\nunrelated<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    assert reuse.generate(p3, GenParams(8, GREEDY)).tokens == fresh.generate(p3, GenParams(8, GREEDY, cache_prompt=False)).tokens


def test_sampling_reuse_and_seed(model, tok, head):
    reuse = make_engine(model, tok, head, max_context=4096)
    fresh = make_engine(model, tok, head, max_context=4096)
    sp = SamplingParams(0.9, 40, 0.95, 0.05, seed=11)
    ids = chat_ids(tok, [{"role": "user", "content": "hi"}])
    a = reuse.generate(ids, GenParams(20, sp))
    ids2 = ids + a.tokens + tok.encode("<|im_end|>\n<|im_start|>user\nagain<|im_end|>\n<|im_start|>assistant\n",
                                       add_special_tokens=False)
    x = reuse.generate(ids2, GenParams(20, sp))
    y = fresh.generate(ids2, GenParams(20, sp, cache_prompt=False))
    assert x.n_cached > 0 and x.tokens == y.tokens  # same seed, same draws
    z = fresh.generate(ids2, GenParams(20, SamplingParams(0.9, 40, 0.95, 0.05, seed=12), cache_prompt=False))
    assert z.tokens != y.tokens


def test_engine_limits(model, tok):
    eng = make_engine(model, tok, max_context=64)
    ids = chat_ids(tok, [{"role": "user", "content": "x " * 80}])
    with pytest.raises(ContextError):
        eng.generate(ids, GenParams(4, GREEDY))
    ids = chat_ids(tok, [{"role": "user", "content": "hi"}])
    r = eng.generate(ids, GenParams(None, GREEDY))  # until the context is full
    assert len(ids) + len(r.tokens) <= 64 and (r.hit_eos or len(ids) + len(r.tokens) == 64)
    assert eng.generate(ids, GenParams(0, GREEDY)).tokens == []
    # EOS inside an accepted run: stop there, and the cache holds nothing past it
    ref = greedy_generate(model, ids, 12).tokens
    eng2 = make_engine(model, tok, max_context=4096)
    eng2.eos = {ref[5]}
    r = eng2.generate(ids, GenParams(12, GREEDY))
    assert r.tokens == ref[: ref.index(ref[5]) + 1] and r.hit_eos
    assert len(eng2.tokens) == len(ids) + len(r.tokens) - 1


# ---------------------------------------------------------------------- HTTP


@pytest.fixture(scope="module")
def server(model, tok, head):
    eng = make_engine(model, tok, head, max_context=2048)
    worker = Worker(lambda: eng, log=False)
    worker.ready.wait()
    app = App(worker, "tiny", "mtp:k=3", 2048, SamplingParams())
    srv = make_server(app, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


def post(port, path, body, raw=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    c.request("POST", path, body=raw if raw is not None else json.dumps(body),
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def get(port, path):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    c.request("GET", path)
    r = c.getresponse()
    return r.status, json.loads(r.read())


def sse_events(data: bytes):
    evs = []
    for block in data.decode().split("\n\n"):
        if block.startswith("data: "):
            payload = block[len("data: "):]
            evs.append(payload if payload == "[DONE]" else json.loads(payload))
    return evs


CHAT = {"messages": [{"role": "user", "content": "Write a poem."}], "max_tokens": 30, "temperature": 0}


def test_http_basic(server):
    assert get(server, "/health") == (200, {"status": "ok"})
    st, m = get(server, "/v1/models")
    assert st == 200 and m["data"][0]["id"] == "tiny"
    for think in (True, False):
        body = {**CHAT, "chat_template_kwargs": {"enable_thinking": think}}
        st, data = post(server, "/v1/chat/completions", body)
        assert st == 200, data
        r = json.loads(data)
        assert r["object"] == "chat.completion" and r["choices"][0]["finish_reason"] in ("stop", "length")
        msg = r["choices"][0]["message"]
        assert msg["role"] == "assistant" and isinstance(msg["content"], str)
        assert r["usage"]["completion_tokens"] == r["timings"]["predicted_n"] <= 30
        for k in ("prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_per_second",
                  "draft_n", "draft_n_accepted", "cache_n"):
            assert k in r["timings"], k
        # streaming gives the same text
        st, data = post(server, "/v1/chat/completions", {**body, "stream": True,
                                                          "stream_options": {"include_usage": True}})
        assert st == 200
        evs = sse_events(data)
        assert evs[-1] == "[DONE]" and evs[0]["choices"][0]["delta"]["role"] == "assistant"
        chunks = evs[1:-1]
        content = "".join(e["choices"][0]["delta"].get("content", "") for e in chunks if e["choices"])
        reasoning = "".join(e["choices"][0]["delta"].get("reasoning_content", "") for e in chunks if e["choices"])
        assert content == msg["content"] and reasoning == msg.get("reasoning_content", "")
        fin = [e for e in chunks if e["choices"] and e["choices"][0]["finish_reason"]]
        assert len(fin) == 1 and fin[0]["choices"][0]["finish_reason"] == r["choices"][0]["finish_reason"]
        assert chunks[-1]["choices"] == [] and chunks[-1]["usage"] == r["usage"] | {
            "prompt_tokens_details": chunks[-1]["usage"]["prompt_tokens_details"]}
        assert "timings" in chunks[-1]
        if think:  # random weights never close the think block
            assert msg.get("reasoning_content") and msg["content"] == ""


def test_http_stop_and_completions(server):
    body = {**CHAT, "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 40}
    full = json.loads(post(server, "/v1/chat/completions", body)[1])["choices"][0]["message"]["content"]
    assert len(full) > 20
    stop = full[12:15]
    want = full[: full.index(stop)]
    for stream in (False, True):
        st, data = post(server, "/v1/chat/completions", {**body, "stop": [stop], "stream": stream})
        if stream:
            evs = sse_events(data)[1:-1]
            got = "".join(e["choices"][0]["delta"].get("content", "") for e in evs)
            reason = evs[-1]["choices"][0]["finish_reason"]
        else:
            r = json.loads(data)
            got, reason = r["choices"][0]["message"]["content"], r["choices"][0]["finish_reason"]
        assert got == want and reason == "stop"
    # /v1/completions, raw prompt
    cb = {"prompt": "Once upon a time", "max_tokens": 20, "temperature": 0}
    r = json.loads(post(server, "/v1/completions", cb)[1])
    assert r["object"] == "text_completion" and r["usage"]["completion_tokens"] == 20
    evs = sse_events(post(server, "/v1/completions", {**cb, "stream": True})[1])
    assert "".join(e["choices"][0]["text"] for e in evs[:-1]) == r["choices"][0]["text"]
    assert evs[-2]["choices"][0]["finish_reason"] == "length"


def test_http_cache_reuse(server):
    msgs = [{"role": "user", "content": "Count to five."}]
    body = {"messages": msgs, "max_tokens": 16, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    r1 = json.loads(post(server, "/v1/chat/completions", body)[1])
    msgs2 = msgs + [r1["choices"][0]["message"], {"role": "user", "content": "Again."}]
    b2 = {**body, "messages": msgs2}
    r2 = json.loads(post(server, "/v1/chat/completions", b2)[1])
    r3 = json.loads(post(server, "/v1/chat/completions", {**b2, "cache_prompt": False})[1])
    assert r2["timings"]["cache_n"] > 0 and r3["timings"]["cache_n"] == 0
    assert r2["usage"]["prompt_tokens_details"]["cached_tokens"] == r2["timings"]["cache_n"]
    assert r2["choices"][0]["message"]["content"] == r3["choices"][0]["message"]["content"]


def test_http_errors(server):
    st, data = post(server, "/v1/chat/completions", None, raw=b"{not json")
    assert st == 400 and "error" in json.loads(data)
    st, data = post(server, "/v1/chat/completions", {**CHAT, "n": 2})
    assert st == 400 and json.loads(data)["error"]["type"] == "invalid_request_error"
    st, data = post(server, "/v1/chat/completions", {**CHAT, "tools": TOOLS, "tool_choice": "required"})
    assert st == 400 and "tool_choice" in json.loads(data)["error"]["message"]
    st, data = post(server, "/v1/chat/completions", {"messages": [{"role": "user", "content": "x " * 3000}]})
    assert st == 400 and json.loads(data)["error"]["type"] == "exceed_context_size_error"
    st, data = post(server, "/v1/chat/completions", {**CHAT, "stream": True, "reasoning_effort": "bogus"})
    assert st == 400
    assert post(server, "/v1/nope", {})[0] == 404
    # tools are accepted and rendered (random weights won't call them)
    st, data = post(server, "/v1/chat/completions", {**CHAT, "tools": TOOLS})
    assert st == 200


def test_http_concurrent(server):
    bodies = [{**CHAT, "messages": [{"role": "user", "content": f"Topic {i}"}], "max_tokens": 12} for i in range(3)]
    seq = [json.loads(post(server, "/v1/chat/completions", b)[1])["choices"][0]["message"] for b in bodies]
    out = [None] * 3

    def run(i):
        if i == 1:  # one streaming client among them
            evs = sse_events(post(server, "/v1/chat/completions", {**bodies[i], "stream": True})[1])
            out[i] = "".join(e["choices"][0]["delta"].get("reasoning_content", "") for e in evs[1:-1])
        else:
            out[i] = json.loads(post(server, "/v1/chat/completions", bodies[i])[1])

    ts = [threading.Thread(target=run, args=(i,)) for i in range(3)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert out[0]["choices"][0]["message"] == seq[0]
    assert out[1] == seq[1].get("reasoning_content", "")
    assert out[2]["choices"][0]["message"] == seq[2]


def test_openai_client(server):
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(base_url=f"http://127.0.0.1:{server}/v1", api_key="none")
    r = client.chat.completions.create(model="tiny", messages=CHAT["messages"], max_tokens=10, temperature=0,
                                       extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    assert r.choices[0].message.content is not None and r.usage.completion_tokens == 10
    parts = [c.choices[0].delta.content or "" for c in client.chat.completions.create(
        model="tiny", messages=CHAT["messages"], max_tokens=10, temperature=0, stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}) if c.choices]
    assert "".join(parts) == r.choices[0].message.content
