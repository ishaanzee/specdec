#!/usr/bin/env python3
"""Run bench/llama_bench.py against a different llama-server binary (without touching the shared harness).

  LLAMA_SERVER=vendor/llama.cpp/build-mm/bin/llama-server \
      .venv/bin/python bench/llama_bench_server.py <llama_bench.py args...>
"""
import importlib.util, os, sys
from pathlib import Path

here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("llama_bench", here / "llama_bench.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.SERVER = str(Path(os.environ["LLAMA_SERVER"]).resolve())
print(f"[llama_bench_server] using {mod.SERVER}", file=sys.stderr)
mod.main()
