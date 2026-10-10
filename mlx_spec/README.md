# mlx_spec: speculative decoding from scratch in MLX (track B)

mlx-lm supplies the Qwen3.8 module tree (`mlx_lm.models.qwen3_5`), weight
loading, tokenizer and cache classes. Everything else is here: the greedy
baseline, the draft/verify/accept loop, cache rollback for the hybrid
architecture, and the drafters. Design notes, measurements and results are in
`notes/mlx-from-scratch.md`.

```
scripts/gpu.py .venv/bin/python -m mlx_spec generate --config draft:k=3 --prompt-id code_edit --profile
scripts/gpu.py .venv/bin/python -m mlx_spec bench --configs greedy ngram:k=3 draft:k=3 --repeat 2
scripts/gpu.py .venv/bin/python -m mlx_spec verify-cost
scripts/gpu.py .venv/bin/python -m pytest tests/test_tiny.py -q                       # seconds
.venv/bin/python scripts/extract_mtp.py                                              # once: MTP weights -> models/mtp-mlx/
SPECDEC_CPU=1 .venv/bin/python -m pytest tests/test_mtp_tiny.py -q                    # MTP cache sync, CPU
scripts/gpu.py .venv/bin/python -m mlx_spec bench --configs mtp:k=3:vocab=65536
.venv/bin/python -m pytest tests/test_sampling.py -q                                  # CPU only, 2 min
.venv/bin/python -m mlx_spec summary                                                  # results table
SPECDEC_27B=1 scripts/gpu.py .venv/bin/python -m pytest tests/test_equivalence_27b.py -q -s
```

| file | what |
|---|---|
| `gdn.py` | `RollbackGDN` (wraps each gated-delta layer), `LinearCache` (its rewindable cache), the per-token-state capture kernel |
| `model.py` | `SpecModel`: `forward(tokens, cache) -> ForwardOutput(logits, hidden, normed)`, `prefill`, `commit(cache, n_keep)` |
| `drafters.py` | `Drafter` protocol, `NgramDrafter`, `DraftModelDrafter` (optionally adaptive k), `HybridDrafter`, `NoDrafter` |
| `mtp.py` | Qwen3.8's MTP layer: `MTPHead`, `load_mtp_head` (weights from `scripts/extract_mtp.py`), `MTPDrafter` (best: `mtp:k=3:vocab=65536`, 2.69x) |
| `decode.py` | `greedy_generate` (baseline), `speculative_generate` (the loop) |
| `bench.py` | results rows in the shared schema, verify-cost curve, mismatch diagnosis |
| `server.py` | OpenAI-compatible HTTP server with prompt caching across turns (section "Server" below) |

Sampling: `speculative_generate(..., temperature=t)` does exact speculative
sampling (accept with `min(1, p/q)`, resample from `max(p - q, 0)`), with
`q` from `drafter.draft_probs()` or one-hot for deterministic drafters;
`tests/test_sampling.py` checks the output distribution. Greedy is the
benchmarked path.

## Rollback in one paragraph

After verifying `T = 1 + k` tokens, `SpecModel.commit(cache, n_keep)` keeps
`n_keep = 1 + accepted`. Attention layers: `KVCache.trim(T - n_keep)`.
Gated-delta layers can't be trimmed, so the layer records the pass's
recurrence inputs and `LinearCache.commit` recomputes the state for the kept
tokens with the same Metal kernel (`replay`, default; bitwise equal to the
state the verify pass had at that position), or picks it from per-token
states written by a modified kernel (`capture`), or re-runs the model
(`reforward`, reference). The conv window is sliced from the recorded conv
input in all modes.

## Adding a drafter (MTP, DFlash, ...)

Implement three methods (see the docstring at the top of `drafters.py` for the
exact invariants):

```python
class MyDrafter:
    name = "mine"
    def start(self, prompt: list[int], prefill: ForwardOutput) -> None: ...
    def draft(self, last: int, k: int) -> mx.array: ...       # <= k tokens after `last`
    def commit(self, out: ForwardOutput, n_accepted: int) -> None: ...
```

`out.hidden` (pre-final-norm) and `out.normed` (post-norm) are the target's
hidden states for every verified position, computed in the verify pass at no
extra cost; rows `0..n_accepted` are committed. That's what MTP / EAGLE /
DFlash heads consume. Then register it in `bench.Runner.make_drafter` so the
CLI and benchmark can use it (`--configs mtp:k=2`). `mtp.MTPDrafter` is a
worked example (hidden-state conditioned, own KV cache trimmed at commit).

## Server

`python -m mlx_spec serve` is an OpenAI-compatible HTTP server around the
speculative loop (`server.py`), shaped like llama-server's API so a chat app
can switch by changing the port. The two 27B models don't fit in 36 GB
together, so stop llama-server first. Then:

```
scripts/gpu.py .venv/bin/python -m mlx_spec serve           # MTP k=3 vocab 65536, http://127.0.0.1:8081
.venv/bin/python -m mlx_spec serve --model models/qwen38-27b-q4km-mlx --port 8081
```

`scripts/gpu.py` waits until no other model job holds the lock, and other
`gpu.py` jobs wait while the server runs. You can also run the server
without it. Startup takes about 30 s: load, MTP head, warmup. `/health`
returns 503 until the model is ready.

Options: `--model` (MLX dir or HF id, default `mlx-community/Qwen3.8-27B-4bit`),
`--drafter` (any config string from the CLI above, e.g. `mtp:k=3:vocab=65536`
(default), `mtp:k=2`, `ngram:k=3`, `draft:k=3`, or `none`), `--port` (8081),
`--host` (127.0.0.1; use 0.0.0.0 for other devices on the LAN),
`--max-context` (32768 tokens, prompt + output), `--alias` (model id shown by
`/v1/models`), `--snapshots` (prompt-cache snapshots, 6), and default sampling
settings `--temp 0.8 --top-k 40 --top-p 0.95 --min-p 0.05` (llama-server's
defaults, used when a request doesn't set them).

**Pointing a chat app at it:** use base URL `http://127.0.0.1:8081/v1`. The
app's own URL setting probably reads `http://127.0.0.1:8080/v1` today, for
llama-server. Any API key works, and the model name is ignored. Example:

```
curl http://127.0.0.1:8081/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Hi!"}], "stream": true,
  "temperature": 0, "chat_template_kwargs": {"enable_thinking": false}}'
```

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8081/v1", api_key="none")
r = c.chat.completions.create(model="qwen", messages=[{"role": "user", "content": "Hi!"}],
                              reasoning_effort="low")
print(r.choices[0].message.reasoning_content, r.choices[0].message.content, r.model_extra["timings"])
```

**Endpoints:** `POST /v1/chat/completions` and `POST /v1/completions`, both
with or without `stream`. `GET /v1/models`, `GET /health`, and a minimal
`GET /props`. CORS is open, as in llama-server.

**Responses** follow llama-server's shapes:

- `usage` includes `prompt_tokens_details.cached_tokens`.
- `timings` holds `cache_n, prompt_n, prompt_ms, prompt_per_second,
  predicted_n, predicted_ms, predicted_per_second, draft_n, draft_n_accepted`.
  It is on the response, or on the last SSE chunk when streaming.
- The first SSE chunk is `{"role": "assistant", "content": null}`.
- `stream_options.include_usage` adds a final chunk with an empty `choices`
  list and `usage`. The stream ends with `data: [DONE]`.
- `finish_reason` is `stop` (EOS or a stop string), `length`, or `tool_calls`.

**Request fields:**

| field | behaviour |
|---|---|
| `messages` | text content, or a list of `text` parts; images are rejected |
| `max_tokens` / `max_completion_tokens` / `n_predict` | default: until `--max-context` |
| `temperature` | 0 = greedy; > 0 = exact speculative sampling |
| `top_k`, `top_p`, `min_p` | supported exactly: applied to the target's logits inside the verify step, in llama.cpp's order (top_k, top_p, min_p, then temperature), so the output distribution equals sampling the model alone with those settings |
| `seed` | reseeds MLX's RNG; same seed + same request gives the same output |
| `stop` | string or list; searched in the generated text (thinking included, as in llama-server) and held back while a partial match is possible; the stop string is not returned |
| `chat_template_kwargs` | passed to the template, e.g. `{"enable_thinking": false}`, `{"reasoning_effort": "low"}`, `{"preserve_thinking": false}` |
| `reasoning_effort` | `xhigh` (the template's default), `medium`, or `low`. OpenAI names are mapped: `high` → `xhigh`, `minimal` → `low`, `none` → `enable_thinking: false` |
| `tools`, `tool_choice` | `auto`/`none` supported. The template renders the tools; `<tool_call>` XML in the output is parsed by mlx-lm's Qwen3-Coder parser into `tool_calls` (streamed as one delta per completed call). `required` or a named function returns 400 (no grammar-constrained decoding) |
| `cache_prompt` | default true; false prefills the whole prompt (llama-server's field) |
| `stream`, `stream_options.include_usage` | as in OpenAI |
| `n` > 1, `response_format` json, `grammar`, `json_schema` | 400 error |
| penalties, `logit_bias`, `logprobs` | ignored, with a warning logged once |

**Thinking.** The Qwen3.8 template opens `<think>` in the prompt unless
`enable_thinking` is false. As with llama-server `--jinja`, the text up to
`</think>` becomes `reasoning_content` (streamed as
`delta.reasoning_content`), and the rest becomes `content`, with leading
whitespace dropped. Streaming holds back text only while it could still be
`</think>`, `<tool_call>`, or a stop string. Text is detokenized with
mlx-lm's streaming detokenizer, so multi-byte characters split across tokens
come out whole.

**One request at a time.** One worker thread owns the model. Concurrent
requests queue in FIFO order, and a streaming client that disconnects stops
its generation after the current round.

**Prompt caching across turns.** Chat apps resend the whole conversation
every turn, and the server reuses what it already computed:

- The 16 attention layers' KV caches can be trimmed to any length. The 48
  gated-delta layers' recurrent and conv state can't.
- So every request snapshots that state (array references, about 154 MB
  each) at three points: before the generation prompt (the last
  `<|im_start|>`), at the end of the prompt, and at the end of the output.
  Each snapshot also keeps the logits of its last token.
- A new request restores the longest snapshot that is a prefix of its
  tokens, trims the KV caches to it, and prefills the rest.
- The MTP head's own KV cache is trimmed to the reused prefix. Missing
  (hidden, token) pairs are fed from the stored target hidden states, so
  drafting is the same as after a fresh prefill.

When the app sends the history back verbatim, the next turn prefills only
the new user message. With thinking on, apps usually drop the reasoning, so
the re-rendered assistant turn doesn't match what was generated. The server
then falls back to the snapshot before that turn and prefills the
assistant's answer again.

Correctness:

- `tests/test_server.py` (tiny fp32 model, real tokenizer) checks that
  output and MTP KV cache with reuse equal a fresh prefill, over 3 turns, a
  raw continuation, an identical resend, and a divergent prompt. It also
  checks that sampled output with a fixed seed is the same either way.
- Mutations are caught: not restoring the GDN state, not trimming the MTP
  cache, or an off-by-one in the MTP pending row each fail the test.
- On the 27B in bf16, reuse changes the batch shapes the prefix was computed
  with, so it can flip near-tie tokens, exactly like the speculative-vs-greedy
  case in `notes/mlx-from-scratch.md`.

**Tests.**

```
.venv/bin/python -m pytest tests/test_server.py -q                    # CPU, ~30 s, no lock needed
scripts/gpu.py .venv/bin/python tests/smoke_server_27b.py             # real 27B through the openai client
SPECDEC_CPU=1 .venv/bin/python tests/smoke_server_27b.py --model <tiny dir> --drafter ngram:k=3 --skip-speed
```

The smoke test starts and stops its own server inside the one `gpu.py`
command, on port 8091. Results go to `results/mlx_server_smoke.json`.
