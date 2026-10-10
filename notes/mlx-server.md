# mlx_spec serve: status (2026-10-10)

Code: `mlx_spec/server.py`, tests `tests/test_server.py` (CPU, 23 tests, pass),
27B smoke `tests/smoke_server_27b.py` (results `results/mlx_server_smoke.json`,
log `logs/server_smoke.log`). Usage: `mlx_spec/README.md`, section "Server".

## First 27B smoke run (AC power, before the prefill fix below)
- Through the server (6 bench prompts, greedy, 256 tok): mean 19.8 tok/s
  (code_edit 23.2, prose 14.0). TTFT ~0.6 s for short prompts.
- 3-turn chat over a 1.9k-token document: turn 2/3 TTFT 0.65 / 0.63 s with
  prompt caching (2061 / 2288 tokens cached, 28 / 25 new) vs 22.3 / 24.9 s
  with `cache_prompt=false`. Output identical with and without reuse, all
  3 turns. Thinking on (history without reasoning): 48 cached / 47 new.
- Streaming == non-streaming; reasoning_content split correct (17*23 ->
  reasoning + "391"); seed reproducible.

## Speed gap (server 19.8 vs CLI 22.3 tok/s): not server overhead
Same locked session (`logs/server_overhead.log`, AC): CLI
`speculative_generate` vs Engine on the main thread, Engine + text layer, and
Engine in a worker thread are within ~1%: code_edit 24.4-24.7, prose
14.7-15.1 tok/s. The CLI itself ran ~11% below the 2026-10-09 rows that day
(27.7 -> 24.5), so the gap came from comparing against older CLI rows (machine
state), not from the server.

## Text differences vs CLI (summarize / explain / math): bf16 near-ties
The server used to prefill chat prompts in two pieces (cut at the last
`<|im_start|>` to snapshot there). With one-piece prefill the Engine output
equals the CLI exactly on all three. With the cut they diverged at tokens
147 / 42 / 8 (the known near-tie points), where the top-2 logits are 23.125/23.125,
24.0/24.125 and 27.75/27.75: numerics, not a bug.
Fix (done, CPU-tested incl. mutation check): prefill in the same chunks as the
CLI and recover the snapshot at the generation prompt mid-chunk by replaying
the GDN recurrence (as rollback does) -> `Engine._prefill`, `_mid_state`.

## Left
- A 27B smoke rerun with the fix was queued (`logs/server_smoke.log`); check
  that the speed section reports `same text as CLI: True` for all 6 prompts.
- Sampling at temperature 0.7 has low acceptance with MTP's greedy drafts
  (10/42 accepted, 8.7 tok/s in a short test); greedy is the fast path.
