#!/usr/bin/env python3
"""Run a command while holding the machine-wide GPU lock.

Only one model fits in memory at a time and parallel runs ruin timings, so every
command that loads a model goes through here: scripts/gpu.py <cmd> [args...]
"""
import fcntl, os, subprocess, sys, time

LOCK = "/tmp/specdec-gpu.lock"

def main():
    if len(sys.argv) < 2:
        sys.exit("usage: gpu.py <cmd> [args...]")
    # "a" so waiting processes don't wipe the holder's line before they get the lock.
    with open(LOCK, "a") as f:
        t0 = time.time()
        fcntl.flock(f, fcntl.LOCK_EX)
        waited = time.time() - t0
        if waited > 1:
            print(f"[gpu.py] waited {waited:.0f}s for GPU lock", file=sys.stderr)
        f.truncate(0)
        f.write(f"{os.getpid()} {' '.join(sys.argv[1:])}\n")
        f.flush()
        sys.exit(subprocess.call(sys.argv[1:]))

if __name__ == "__main__":
    main()
