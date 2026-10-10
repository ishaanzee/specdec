#!/usr/bin/env python3
"""Run a command while holding the machine-wide GPU lock.

Only one model fits in memory at a time and parallel runs ruin timings, so every
command that loads a model goes through here: scripts/gpu.py <cmd> [args...]

Waiters are served first come, first served (a ticket queue in QUEUE_DIR), and the
lock is not taken while the user's own model server is running (any llama-server or
`mlx_spec serve` process that isn't ours), since that already uses most of memory.
Set SPECDEC_IGNORE_FOREIGN=1 to skip that check.
"""
import fcntl, os, subprocess, sys, time

LOCK = "/tmp/specdec-gpu.lock"
QUEUE_DIR = "/tmp/specdec-gpu-queue"
FOREIGN = r"llama-server|mlx_spec serve"


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def first_in_queue(ticket):
    tickets = sorted(os.listdir(QUEUE_DIR))
    for t in tickets:
        if alive(int(t.split("-")[1])):
            return t == ticket
        os.unlink(os.path.join(QUEUE_DIR, t))  # stale ticket from a dead waiter
    return False


def foreign_servers():
    if os.environ.get("SPECDEC_IGNORE_FOREIGN") == "1":
        return []
    out = subprocess.run(["pgrep", "-f", FOREIGN], capture_output=True, text=True).stdout
    return [int(p) for p in out.split() if int(p) != os.getpid()]


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: gpu.py <cmd> [args...]")
    os.makedirs(QUEUE_DIR, exist_ok=True)
    ticket = f"{time.time_ns()}-{os.getpid()}"
    open(os.path.join(QUEUE_DIR, ticket), "w").close()
    t0, said = time.time(), set()
    try:
        # "a" so waiting processes don't wipe the holder's line before they get the lock.
        with open(LOCK, "a") as f:
            while True:
                if first_in_queue(ticket):
                    fcntl.flock(f, fcntl.LOCK_EX)
                    servers = foreign_servers()
                    if not servers:
                        break
                    fcntl.flock(f, fcntl.LOCK_UN)
                    if "foreign" not in said:
                        print(f"[gpu.py] waiting: user's model server running (pids {servers})", file=sys.stderr)
                        said.add("foreign")
                time.sleep(5)
            os.unlink(os.path.join(QUEUE_DIR, ticket))
            waited = time.time() - t0
            if waited > 1:
                print(f"[gpu.py] waited {waited:.0f}s for GPU lock", file=sys.stderr)
            f.truncate(0)
            f.write(f"{os.getpid()} {' '.join(sys.argv[1:])}\n")
            f.flush()
            sys.exit(subprocess.call(sys.argv[1:]))
    finally:
        try:
            os.unlink(os.path.join(QUEUE_DIR, ticket))
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
