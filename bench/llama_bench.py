#!/usr/bin/env python3
"""Shared llama.cpp (llama-server) benchmark harness for the specdec experiments.

USAGE (always under the GPU lock; the whole session runs inside one lock):

  scripts/gpu.py .venv/bin/python bench/llama_bench.py \
      --experiment ngram --method ngram-mod --config-name defaults \
      --model PATH/TO/main.gguf --repeat 3 -- --spec-type ngram-mod --spec-draft-n-max 16

  # Save reference outputs (do this once, with no speculation):
  scripts/gpu.py .venv/bin/python bench/llama_bench.py \
      --experiment baseline --method none --config-name none \
      --model PATH --repeat 3 --save-baseline -- --spec-type none

Everything after `--` is passed to llama-server verbatim (spec flags, -md, ...).
The harness always adds the fixed settings: -ngl 999 -fa on -c 8192 -t 6 -tb 6
-np 1 --no-webui (override with --ctx / --threads; don't, unless you mean it).

What it does: picks a free port, starts llama-server, waits for /health, sends a
tiny warm-up request, then for each round (1..--repeat) and each prompt in
bench/prompts.jsonl POSTs /v1/chat/completions with temperature 0, max_tokens
256, chat_template_kwargs {"enable_thinking": false}, cache_prompt false. It
reads the server's `timings` block (predicted_per_second, prompt_per_second,
predicted_n, draft_n, draft_n_accepted), hashes message.content (sha256), and
compares it with results/baseline_outputs/<model-tag>/<prompt_id>.txt.

Server restarts (--restart): n-gram drafters (ngram-mod, ngram-cache, ...) learn
from previously generated text, so re-running the same prompt on the same server
would let them "remember" the answer and inflate acceptance. Default is
`prompt`: a fresh server for every (round, prompt) pair, so no state leaks across
prompts or repeats. `round` restarts once per round (prompts can learn from earlier
prompts in the same round); `never` uses one server for everything.

Output: one row per (method, config, prompt) appended to
results/<experiment>.jsonl with the README schema plus extras (medians over
repeats; per-run values in `runs`). The server log goes to
logs/llama_<experiment>_<config>.log. A summary table is printed at the end.

--paired: for every (round, prompt), also run a fresh `--spec-type none` server on
the same prompt immediately before or after (order alternates) and report
`speedup` = candidate gen tok/s / reference gen tok/s (median over rounds). Use
this whenever the machine is shared: background load drifts by 20%+ over an hour,
which swamps unpaired comparisons. Doubles the run time.

Other options: --only id1,id2 (subset of prompts), --max-tokens, --model-tag
(baseline directory name; default = gguf filename stem), --dump-raw (print the
first raw JSON response), --no-write (don't append results), --track (default A).
"""
import argparse, hashlib, json, os, signal, socket, statistics, subprocess, sys, time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
SERVER = "/opt/homebrew/bin/llama-server"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Server:
    def __init__(self, model, extra, ctx, threads, log_path):
        self.port = free_port()
        self.cmd = [SERVER, "-m", model, "--host", "127.0.0.1", "--port", str(self.port),
                    "-ngl", "999", "-fa", "on", "-c", str(ctx), "-t", str(threads), "-tb", str(threads),
                    "-np", "1", "--no-webui", "--metrics"] + list(extra)
        self.log_path = log_path
        self.proc = None

    def start(self, timeout=600):
        self.log = open(self.log_path, "a")
        self.log.write(f"\n===== {time.strftime('%F %T')} {' '.join(self.cmd)}\n")
        self.log.flush()
        t0 = time.time()
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        url = f"http://127.0.0.1:{self.port}/health"
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.proc.returncode}; see {self.log_path}")
            try:
                r = requests.get(url, timeout=2)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    self.load_s = time.time() - t0
                    return self
            except (requests.ConnectionError, requests.Timeout, ValueError):
                pass
            time.sleep(0.5)
        self.stop()
        raise RuntimeError("llama-server did not become healthy in time")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=30)
            except Exception:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        if getattr(self, "log", None):
            self.log.close()

    def chat(self, prompt, max_tokens, timeout=1800):
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "top_k": 1, "seed": 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "cache_prompt": False,
            "stream": False,
        }
        t0 = time.time()
        r = requests.post(f"http://127.0.0.1:{self.port}/v1/chat/completions", json=body, timeout=timeout)
        wall = time.time() - t0
        r.raise_for_status()
        return r.json(), wall


def load_prompts(path, only):
    ps = [json.loads(l) for l in open(path) if l.strip()]
    if only:
        keep = set(only.split(","))
        ps = [p for p in ps if p["id"] in keep]
    return ps


def one_run(srv, p, max_tokens, dump_raw):
    resp, wall = srv.chat(p["prompt"], max_tokens)
    if dump_raw:
        print(json.dumps(resp, indent=1)[:6000], file=sys.stderr)
    msg = resp["choices"][0]["message"]
    text = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    t = resp.get("timings", {})
    dn = t.get("draft_n")
    da = t.get("draft_n_accepted")
    return {
        "text": text,
        "reasoning_chars": len(reasoning),
        "think_tag_in_content": "<think>" in text,
        "n_gen": t.get("predicted_n"),
        "n_prompt": t.get("prompt_n"),
        "gen_tok_s": t.get("predicted_per_second"),
        "prompt_tok_s": t.get("prompt_per_second"),
        "draft_n": dn,
        "draft_accepted": da,
        "wall_s": round(wall, 3),
        "finish_reason": resp["choices"][0].get("finish_reason"),
        "timings": t,
    }


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-tag", default=None)
    ap.add_argument("--track", default="A")
    ap.add_argument("--prompts", default=str(ROOT / "bench" / "prompts.jsonl"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--restart", choices=["prompt", "round", "never"], default="prompt")
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--paired", action="store_true",
                    help="run a --spec-type none reference next to every candidate run; adds ref_gen_tok_s and speedup")
    ap.add_argument("--dump-raw", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("extra", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    extra = a.extra[1:] if a.extra and a.extra[0] == "--" else a.extra
    tag = a.model_tag or Path(a.model).name.removesuffix(".gguf")
    base_dir = ROOT / "results" / "baseline_outputs" / tag
    (ROOT / "logs").mkdir(exist_ok=True)
    log_path = ROOT / "logs" / f"llama_{a.experiment}_{a.config_name}.log"
    prompts = load_prompts(a.prompts, a.only)

    runs = {p["id"]: [] for p in prompts}
    refs = {p["id"]: [] for p in prompts}
    srv = None
    load_times = []

    def ref_run(p):
        # paired reference: a no-speculation server run right next to the candidate run,
        # so both see the same background load (this machine is shared).
        rs = Server(a.model, ["--spec-type", "none"], a.ctx, a.threads, log_path).start()
        try:
            rs.chat("Say hi.", 4)
            r = one_run(rs, p, a.max_tokens, False)
        finally:
            rs.stop()
        refs[p["id"]].append(r)
        print(f"[ref] {p['id']:<10} gen {r['gen_tok_s']:.2f} tok/s", file=sys.stderr, flush=True)

    def fresh():
        nonlocal srv
        if srv:
            srv.stop()
        srv = Server(a.model, extra, a.ctx, a.threads, log_path).start()
        load_times.append(srv.load_s)
        srv.chat("Say hi.", 4)  # warm-up (Metal kernels, allocations)

    try:
        dumped = False
        for rnd in range(a.repeat):
            if a.restart == "round" or (a.restart == "never" and srv is None):
                fresh()
            for i, p in enumerate(prompts):
                ref_first = a.paired and (rnd + i) % 2 == 0
                if ref_first:
                    if srv:
                        srv.stop(); srv = None
                    ref_run(p)
                if a.restart == "prompt" or (a.paired and srv is None):
                    fresh()
                r = one_run(srv, p, a.max_tokens, a.dump_raw and not dumped)
                dumped = True
                runs[p["id"]].append(r)
                acc = (f" acc {r['draft_accepted']}/{r['draft_n']}" if r["draft_n"] else "")
                print(f"[{a.method}/{a.config_name}] round {rnd+1} {p['id']:<10} "
                      f"gen {r['gen_tok_s']:.2f} tok/s  n={r['n_gen']}{acc}  "
                      f"prompt {r['prompt_tok_s']:.1f} tok/s", file=sys.stderr, flush=True)
                if a.paired and not ref_first:
                    srv.stop(); srv = None
                    ref_run(p)
    finally:
        if srv:
            srv.stop()

    rows = []
    for p in prompts:
        rs = runs[p["id"]]
        texts = [r["text"] for r in rs]
        sha = hashlib.sha256(texts[0].encode()).hexdigest()
        bfile = base_dir / f"{p['id']}.txt"
        if a.save_baseline:
            base_dir.mkdir(parents=True, exist_ok=True)
            bfile.write_text(texts[0])
        matches, diverge = None, None
        if bfile.exists():
            b = bfile.read_text()
            matches = texts[0] == b
            if not matches:
                diverge = next((i for i, (x, y) in enumerate(zip(texts[0], b)) if x != y), min(len(b), len(texts[0])))
        dn = med([r["draft_n"] for r in rs])
        da = med([r["draft_accepted"] for r in rs])
        rf = refs[p["id"]]
        paired = {}
        if rf:
            ratios = [r["gen_tok_s"] / q["gen_tok_s"] for r, q in zip(rs, rf)]
            paired = {"ref_gen_tok_s": round(med([q["gen_tok_s"] for q in rf]), 3),
                      "speedup": round(med(ratios), 4),
                      "ref_runs": [q["gen_tok_s"] for q in rf],
                      "ref_matches_baseline": all(q["text"] == bfile.read_text() for q in rf) if bfile.exists() else None}
        rows.append({
            "track": a.track, "method": a.method, "config": a.config_name, "prompt_id": p["id"],
            "kind": p.get("kind"),
            "n_gen": rs[0]["n_gen"],
            "gen_tok_s": round(med([r["gen_tok_s"] for r in rs]), 3),
            "prompt_tok_s": round(med([r["prompt_tok_s"] for r in rs]), 2),
            "draft_n": dn, "draft_accepted": da,
            "accept_rate": round(da / dn, 4) if dn else None,
            "output_sha256": sha,
            "matches_baseline": matches,
            "diverge_at_char": diverge,
            "repeats_consistent": all(t == texts[0] for t in texts),
            "repeat": len(rs),
            "runs": [{k: r[k] for k in ("gen_tok_s", "prompt_tok_s", "draft_n", "draft_accepted", "wall_s", "n_gen")} for r in rs],
            "reasoning_chars": rs[0]["reasoning_chars"],
            "think_tag_in_content": rs[0]["think_tag_in_content"],
            "finish_reason": rs[0]["finish_reason"],
            "model": tag,
            "server_args": extra,
            "fixed_args": f"-ngl 999 -fa on -c {a.ctx} -t {a.threads} -tb {a.threads} -np 1",
            "restart": a.restart,
            "server_load_s": round(med(load_times), 1),
            "llama_cpp": "brew 0.4.1 b10964 b29c606e2",
            **paired,
            "time": time.strftime("%F %T"),
        })

    if not a.no_write:
        out = ROOT / "results" / f"{a.experiment}.jsonl"
        out.parent.mkdir(exist_ok=True)
        with open(out, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    print(f"\n{a.method} / {a.config_name}  (median of {a.repeat}, model {tag})")
    print(f"{'prompt':<11}{'gen tok/s':>10}{'prompt t/s':>11}{'accept':>14}{'match':>7}{'ref t/s':>9}{'speedup':>9}")
    for row in rows:
        acc = f"{row['draft_accepted']}/{row['draft_n']}" if row["draft_n"] else "-"
        extra_cols = f"{row['ref_gen_tok_s']:>9.2f}{row['speedup']:>9.3f}" if "speedup" in row else ""
        print(f"{row['prompt_id']:<11}{row['gen_tok_s']:>10.2f}{row['prompt_tok_s']:>11.1f}{acc:>14}{str(row['matches_baseline']):>7}{extra_cols}")
    tail = ""
    if all("speedup" in r for r in rows):
        tail = f"{'':>32}{statistics.mean(r['ref_gen_tok_s'] for r in rows):>9.2f}{statistics.mean(r['speedup'] for r in rows):>9.3f}"
    print(f"{'mean':<11}{statistics.mean(r['gen_tok_s'] for r in rows):>10.2f}{tail}")
    if any(r["reasoning_chars"] or r["think_tag_in_content"] for r in rows):
        print("WARNING: thinking output detected (reasoning_content or <think> in content)")


if __name__ == "__main__":
    main()
