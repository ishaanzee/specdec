#!/usr/bin/env python3
"""Smoke test of `python -m mlx_spec serve` with the real 27B, through the openai client.

Starts the server as a subprocess, runs the checks, stops the server (also on
errors / Ctrl-C). Run it under the GPU lock as ONE command, so gpu.py doesn't
mistake our own server for the user's:

    scripts/gpu.py .venv/bin/python tests/smoke_server_27b.py [--model ...] [--drafter ...]

Checks / measurements (greedy, thinking off unless noted):
  1. speed: the six bench prompts via /v1/completions (template applied by us,
     256 tokens): tok/s, acceptance, and whether the text equals the CLI's
     (results/mlx_mtp.jsonl output_sha256 for the same config)
  2. non-streaming vs streaming chat: same text; time to first token
  3. thinking on (reasoning_effort low): reasoning_content / content split,
     streamed == non-streamed
  4. 3-turn conversation over a ~1.5k-token document: with prompt caching vs
     cache_prompt=false: cache_n, prompt time, TTFT, identical output
  5. sampling (temperature 0.7, top_p 0.9, seed): same seed -> same text
Writes results/mlx_server_smoke.json.
"""

import argparse
import atexit
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mlx-community/Qwen3.8-27B-4bit")
ap.add_argument("--drafter", default="mtp:k=3:vocab=65536")
ap.add_argument("--port", type=int, default=8091)
ap.add_argument("--skip-speed", action="store_true")
a = ap.parse_args()

BASE = f"http://127.0.0.1:{a.port}"
log_path = ROOT / "logs" / "server_smoke_server.log"
proc = subprocess.Popen(
    [sys.executable, "-m", "mlx_spec", "serve", "--model", a.model, "--drafter", a.drafter,
     "--port", str(a.port)],
    cwd=ROOT, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, start_new_session=True)


def stop_server():
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    print(f"[smoke] server stopped (exit {proc.returncode})", flush=True)


atexit.register(stop_server)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))

t0 = time.time()
while True:
    if proc.poll() is not None:
        sys.exit(f"server died, see {log_path}")
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
            if r.status == 200:
                break
    except Exception:
        pass
    if time.time() - t0 > 900:
        sys.exit("server did not come up in 15 min")
    time.sleep(2)
print(f"[smoke] server ready after {time.time() - t0:.0f}s", flush=True)

import openai  # noqa: E402

client = openai.OpenAI(base_url=BASE + "/v1", api_key="none", timeout=1800)
MODEL = client.models.list().data[0].id
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
power = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.split("\n")[0]
out = {"model": a.model, "drafter": a.drafter, "time": time.strftime("%F %T"),
       "power": power.split("'")[1] if "'" in power else power}
print(f"[smoke] power source: {out['power']}", flush=True)


def chat(messages, stream=False, **kw):
    """-> dict(content, reasoning, timings, usage, ttft, wall, finish)."""
    extra = dict(kw.pop("extra_body", {}))
    t = time.perf_counter()
    if not stream:
        r = client.chat.completions.create(model=MODEL, messages=messages, extra_body=extra, **kw)
        m = r.choices[0].message
        return dict(content=m.content or "", reasoning=getattr(m, "reasoning_content", None) or "",
                    timings=r.model_extra["timings"], usage=r.usage.model_dump(), wall=time.perf_counter() - t,
                    finish=r.choices[0].finish_reason)
    content, reasoning, ttft, timings, finish = "", "", None, None, None
    for c in client.chat.completions.create(model=MODEL, messages=messages, stream=True, extra_body=extra,
                                            stream_options={"include_usage": True}, **kw):
        if c.choices:
            d = c.choices[0].delta
            piece = (d.content or "") + (getattr(d, "reasoning_content", None) or "")
            if piece and ttft is None:
                ttft = time.perf_counter() - t
            content += d.content or ""
            reasoning += getattr(d, "reasoning_content", None) or ""
            finish = c.choices[0].finish_reason or finish
        if (c.model_extra or {}).get("timings"):
            timings = c.model_extra["timings"]
    return dict(content=content, reasoning=reasoning, timings=timings, ttft=ttft,
                wall=time.perf_counter() - t, finish=finish)


def fmt(t):
    return (f"prompt {t['prompt_n']} new + {t['cache_n']} cached in {t['prompt_ms'] / 1e3:.2f}s "
            f"({t['prompt_per_second']:.1f} t/s) | gen {t['predicted_n']} @ {t['predicted_per_second']:.2f} t/s"
            + (f", accepted {t['draft_n_accepted']}/{t['draft_n']}" if t.get("draft_n") else ""))


# ------------------------------------------------------------ 1. speed
if not a.skip_speed:
    from mlx_lm.tokenizer_utils import load as load_tok
    from mlx_lm.utils import _download

    tok = load_tok(Path(_download(a.model)))
    prompts = [json.loads(l) for l in open(ROOT / "bench" / "prompts.jsonl") if l.strip()]
    cli = {}
    cfg_name = "_".join(a.drafter.split(":")[1:]).replace("=", "")
    for l in open(ROOT / "results" / "mlx_mtp.jsonl"):
        r = json.loads(l)
        if r["config"] == cfg_name and r["method"] == a.drafter.split(":")[0] and r.get("model", "").endswith(a.model.split("/")[-1]):
            cli[r["prompt_id"]] = r
    rows = []
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True,
                                       enable_thinking=False, tokenize=False)
        r = client.completions.create(model=MODEL, prompt=text, max_tokens=256, temperature=0,
                                      extra_body={"cache_prompt": False})
        t = r.model_extra["timings"]
        sha = hashlib.sha256(r.choices[0].text.encode()).hexdigest()
        ref = cli.get(p["id"])
        row = dict(prompt_id=p["id"], gen_tok_s=round(t["predicted_per_second"], 2), n_gen=t["predicted_n"],
                   accept=round(t["draft_n_accepted"] / max(t["draft_n"], 1), 3),
                   prompt_tok_s=round(t["prompt_per_second"], 1),
                   cli_gen_tok_s=ref and ref["gen_tok_s"], same_text_as_cli=ref and sha == ref["output_sha256"],
                   cli_matches_greedy=ref and ref["matches_baseline"])
        rows.append(row)
        print(f"[speed] {p['id']:<10} {fmt(t)} | CLI {row['cli_gen_tok_s']} t/s, same text as CLI: "
              f"{row['same_text_as_cli']}", flush=True)
    mean = sum(r["gen_tok_s"] for r in rows) / len(rows)
    print(f"[speed] mean {mean:.2f} tok/s (CLI mean "
          f"{sum(r['cli_gen_tok_s'] or 0 for r in rows) / len(rows):.2f})", flush=True)
    out["speed"] = dict(rows=rows, mean_gen_tok_s=round(mean, 2))

# ------------------------------------------- 2. streaming vs non-streaming
q = [{"role": "user", "content": "Explain in a short paragraph why the sky is blue."}]
ns = chat(q, max_tokens=200, temperature=0, extra_body=NOTHINK)
st = chat(q, stream=True, max_tokens=200, temperature=0, extra_body={**NOTHINK, "cache_prompt": False})
print(f"[stream] non-stream: {fmt(ns['timings'])}\n[stream] stream:     {fmt(st['timings'])}, "
      f"TTFT {st['ttft']:.2f}s | same text: {ns['content'] == st['content']}", flush=True)
print("[stream] text:", ns["content"][:200].replace("\n", " "), "...", flush=True)
out["stream"] = dict(same_text=ns["content"] == st["content"], ttft_s=st["ttft"], timings=st["timings"],
                     nonstream_timings=ns["timings"], finish=st["finish"])

# ----------------------------------------------------------- 3. thinking
q = [{"role": "user", "content": "What is 17 * 23? Answer with the number."}]
th = chat(q, max_tokens=1024, temperature=0, reasoning_effort="low")
th_s = chat(q, stream=True, max_tokens=1024, temperature=0, reasoning_effort="low")
print(f"[think] {fmt(th['timings'])} finish {th['finish']}\n[think] reasoning: {th['reasoning'][:160]!r}...\n"
      f"[think] content: {th['content']!r}\n[think] streamed == non-streamed: "
      f"{(th['reasoning'], th['content']) == (th_s['reasoning'], th_s['content'])}", flush=True)
out["thinking"] = dict(reasoning_len=len(th["reasoning"]), content=th["content"], finish=th["finish"],
                       same_stream=(th["reasoning"], th["content"]) == (th_s["reasoning"], th_s["content"]),
                       has_think_tags="<think>" in th["content"] + th["reasoning"] or "</think>" in th["content"])

# ------------------------------------------------- 4. 3-turn prompt caching
doc = (ROOT / "notes" / "mlx-from-scratch.md").read_text()[:6000]
turns = [f"Here is a document:\n\n{doc}\n\nSummarize it in three bullet points.",
         "What does it say about rolling back the gated-delta layers?",
         "Suggest one next step, in two sentences."]
msgs, reuse = [], []
for i, u in enumerate(turns):
    msgs.append({"role": "user", "content": u})
    r = chat(list(msgs), stream=True, max_tokens=200, temperature=0, extra_body=NOTHINK)
    reuse.append(r)
    print(f"[cache] turn {i + 1} reuse:    TTFT {r['ttft']:.2f}s | {fmt(r['timings'])}", flush=True)
    msgs.append({"role": "assistant", "content": r["content"]})
fresh = []
for i in range(len(turns)):
    m = msgs[: 2 * i + 1]
    r = chat(m, stream=True, max_tokens=200, temperature=0, extra_body={**NOTHINK, "cache_prompt": False})
    fresh.append(r)
    same = r["content"] == reuse[i]["content"]
    div = None
    if not same:
        div = next((j for j, (x, y) in enumerate(zip(r["content"], reuse[i]["content"])) if x != y),
                   min(len(r["content"]), len(reuse[i]["content"])))
    print(f"[cache] turn {i + 1} no reuse: TTFT {r['ttft']:.2f}s | {fmt(r['timings'])} | same output: {same}"
          + (f" (diverges at char {div})" if div is not None else ""), flush=True)
out["cache"] = [dict(turn=i + 1, reuse_ttft_s=x["ttft"], fresh_ttft_s=y["ttft"], reuse_timings=x["timings"],
                     fresh_timings=y["timings"], same_output=x["content"] == y["content"])
                for i, (x, y) in enumerate(zip(reuse, fresh))]
# thinking on, 2 turns, history without reasoning_content (what most chat apps send back)
tm = [{"role": "user", "content": "Name a prime number between 10 and 20."}]
r1 = chat(tm, stream=True, max_tokens=600, temperature=0, reasoning_effort="low")
tm += [{"role": "assistant", "content": r1["content"]}, {"role": "user", "content": "And one between 20 and 30?"}]
r2 = chat(tm, stream=True, max_tokens=600, temperature=0, reasoning_effort="low")
print(f"[cache] thinking turn 2: TTFT {r2['ttft']:.2f}s | {fmt(r2['timings'])}", flush=True)
out["cache_thinking_turn2"] = r2["timings"]

# ------------------------------------------------------------ 5. sampling
q = [{"role": "user", "content": "Write a haiku about the sea."}]
s1 = chat(q, max_tokens=60, temperature=0.7, top_p=0.9, seed=42, extra_body=NOTHINK)
s2 = chat(q, max_tokens=60, temperature=0.7, top_p=0.9, seed=42, extra_body={**NOTHINK, "cache_prompt": False})
s3 = chat(q, max_tokens=60, temperature=0.7, top_p=0.9, seed=43, extra_body=NOTHINK)
print(f"[sample] {fmt(s1['timings'])}\n[sample] seed 42 twice identical: {s1['content'] == s2['content']}, "
      f"seed 43 differs: {s1['content'] != s3['content']}\n[sample] {s1['content']!r}", flush=True)
out["sampling"] = dict(timings=s1["timings"], same_seed_same=s1["content"] == s2["content"],
                       other_seed_differs=s1["content"] != s3["content"])

(ROOT / "results" / "mlx_server_smoke.json").write_text(json.dumps(out, indent=1))
print("[smoke] done; results/mlx_server_smoke.json", flush=True)
