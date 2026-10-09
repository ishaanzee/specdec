"""Where does the verify-cost staircase come from? Time mx.quantized_matmul
alone for M = 1..32 rows on Qwen3.8-27B's MLP shapes (4-bit, group 64).

    scripts/gpu.py .venv/bin/python bench/mlx_qmm_curve.py
"""

import json
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
SHAPES = {"mlp_up (5120->17408)": (17408, 5120), "mlp_down (17408->5120)": (5120, 17408),
          "lm_head (5120->248320)": (248320, 5120)}
REPS = 20


def main():
    rows = []
    for name, (n_out, n_in) in SHAPES.items():
        # stack several copies so one call reads ~1-2 GB like a slice of the model
        n_copies = max(1, int(1.5e9 // (n_out * n_in // 2)))
        ws = []
        for _ in range(n_copies):
            w = mx.random.normal((n_out, n_in), dtype=mx.bfloat16) * 0.02
            ws.append(mx.quantize(w, group_size=64, bits=4))
        mx.eval(ws)
        gb = n_copies * n_out * n_in / 2 / 1e9
        base = None
        for M in range(1, 33):
            x = mx.random.normal((1, M, n_in), dtype=mx.bfloat16)
            mx.eval(x)

            def run():
                return [mx.quantized_matmul(x, q, s, b, transpose=True, group_size=64, bits=4)
                        for q, s, b in ws]

            mx.eval(run())
            ts = []
            for _ in range(REPS):
                mx.synchronize()
                t = time.perf_counter()
                mx.eval(run())
                ts.append(time.perf_counter() - t)
            ms = 1e3 * sorted(ts)[REPS // 2]
            base = base or ms
            rows.append({"shape": name, "M": M, "ms": round(ms, 3), "x_M1": round(ms / base, 3),
                         "GB_per_s": round(gb / (ms / 1e3), 1)})
            print(f"{name:<24} M={M:<3} {ms:8.2f} ms  {ms / base:5.2f}x  {gb / (ms / 1e3):6.1f} GB/s", flush=True)
        del ws
    with open(ROOT / "results" / "mlx_qmm_curve.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
