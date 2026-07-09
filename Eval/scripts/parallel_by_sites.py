"""
Run WebArena/VAB evaluations in parallel (no site-based isolation).

This script **does not attempt to isolate tasks by site**. It assumes tasks are
independent and can run concurrently. Concurrency is only bounded by `--max_parallel`.

Approach:
1) Build the list of task indices from --test_indices or --test_start_idx/--test_end_idx.
2) (Optional) skip finished tasks by checking <result_dir>/render_<idx>.html.
3) Split indices into chunks of size `--chunk_size`; each chunk is executed sequentially
   inside one `run.py` process via `--test_indices a,b,c`.
4) Run the resulting `run.py` commands concurrently with a global limit `--max_parallel`.

Usage example:
  python scripts/parallel_by_sites.py \
    --test_config_base_dir config_files/wa/test_webarena_lite \
    --result_dir cache/my_run \
    --max_parallel 8 \
    --chunk_size 1 \
    -- \
    --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json \
    --provider openai --model gpt-4o-mini --mode chat

Everything after `--` is forwarded to run.py.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def _http_get(url: str, *, timeout_s: float) -> tuple[int, str]:
    with urlopen(url, timeout=timeout_s) as resp:  # nosec - URL is user-provided
        status = getattr(resp, "status", 200)
        body = resp.read().decode("utf-8", errors="ignore")
        return int(status), body


def _reset_instance_via_server(
    reset_server_url: str,
    *,
    timeout_s: float,
    poll_interval_s: float,
    request_timeout_s: float,
    max_retries: int,
) -> None:
    """
    Trigger a full docker instance reset via the reset_server (WebArena-Env-Setup/07_serve_reset.sh),
    then poll /status until Ready.
    """
    base = reset_server_url.rstrip("/")
    reset_url = f"{base}/reset"
    status_url = f"{base}/status"

    # Trigger reset (418 means another reset is already running).
    last_err: Exception | None = None
    for _attempt in range(max_retries):
        try:
            code, _body = _http_get(reset_url, timeout_s=request_timeout_s)
            if code in (200, 418):
                break
            last_err = RuntimeError(f"reset server returned status={code} for GET {reset_url}")
        except Exception as e:
            last_err = e
        time.sleep(min(2.0, poll_interval_s))
    else:
        raise RuntimeError(f"Failed to trigger reset via {reset_url}: {repr(last_err)}")

    # Poll status until ready or timeout.
    start = time.time()
    while True:
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for reset server to become ready: {status_url}")
        try:
            code, body = _http_get(status_url, timeout_s=request_timeout_s)
            if code == 500:
                raise RuntimeError(f"Reset server reports failure (500) at {status_url}: {body}")
            if code == 200:
                b = (body or "").lower()
                if ("ready" in b) and ("duty" in b):
                    return
        except URLError:
            # transient network issue, keep polling
            pass
        time.sleep(poll_interval_s)


def _parse_indices(spec: str | None, *, start: int, end: int) -> list[int]:
    if spec:
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        return [int(p) for p in parts]
    return list(range(start, end))


def _is_finished(result_dir: str, idx: int) -> bool:
    # run.py marks completion by writing render_<task_id>.html (task_id == idx for standard configs)
    return Path(result_dir, f"render_{idx}.html").exists()


def _ensure_config_exists(base_dir: str, idx: int) -> None:
    config_path = Path(base_dir) / f"{idx}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

def _chunk_indices(indices: list[int], *, chunk_size: int) -> list[list[int]]:
    if chunk_size <= 0:
        raise ValueError("--chunk_size must be a positive integer")
    return [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]


def _strip_known_arg(tokens: list[str], flag: str) -> list[str]:
    """Remove occurrences of `flag VALUE` from tokens (best-effort)."""
    out: list[str] = []
    skip_next = False
    for i, tok in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if tok == flag:
            if i + 1 < len(tokens):
                skip_next = True
            continue
        out.append(tok)
    return out


def _run_with_limit(
    commands: list[list[str]],
    *,
    max_parallel: int,
    poll_s: float = 0.5,
) -> int:
    """Run commands with concurrency limit. Return 0 if all succeed, else first non-zero."""
    running: list[subprocess.Popen] = []
    pending = list(commands)

    while pending or running:
        while pending and len(running) < max_parallel:
            cmd = pending.pop(0)
            running.append(subprocess.Popen(cmd))
            time.sleep(0.05)  # small stagger to reduce burst load

        # reap finished
        for p in list(running):
            ret = p.poll()
            if ret is None:
                continue
            running.remove(p)
            if ret != 0:
                for other in running:
                    try:
                        other.terminate()
                    except Exception:
                        pass
                return ret

        if running:
            time.sleep(poll_s)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_config_base_dir", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=910)
    parser.add_argument(
        "--test_indices",
        type=str,
        default=None,
        help="Comma-separated indices override start/end, e.g. '0,5,7'",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=0,
        help=(
            "How many task indices to run sequentially in one run.py process. "
            "Use 0 to auto-balance across --max_parallel. Default: 0."
        ),
    )
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=8,
        help="Max concurrent run.py processes within a batch.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use for launching run.py.",
    )
    parser.add_argument(
        "--run_script",
        type=str,
        default="run.py",
        help="Path to run.py (relative to Eval/).",
    )
    parser.add_argument(
        "--skip_finished",
        action="store_true",
        help="Skip tasks that already have render_<idx>.html in result_dir.",
    )
    parser.add_argument(
        "--reset_server_url",
        type=str,
        default=None,
        help="Reset server base URL, e.g. http://127.0.0.1:7565",
    )
    parser.add_argument(
        "--reset_before_run",
        action="store_true",
        help="Trigger ONE global reset before starting any parallel jobs.",
    )
    parser.add_argument(
        "--reset_timeout_s",
        type=float,
        default=600.0,
        help="Max seconds to wait for reset server to become Ready.",
    )
    parser.add_argument(
        "--reset_poll_interval_s",
        type=float,
        default=2.0,
        help="Polling interval (seconds) for reset server /status.",
    )
    parser.add_argument(
        "--reset_request_timeout_s",
        type=float,
        default=10.0,
        help="HTTP timeout (seconds) for reset server requests.",
    )
    parser.add_argument(
        "--reset_max_retries",
        type=int,
        default=3,
        help="Max retries for triggering reset endpoint.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print the computed batches and commands.",
    )

    # Forward everything after `--` to run.py
    args, forward = parser.parse_known_args()
    # If the caller uses `--` as a separator, argparse will keep it in the unknown args.
    # We should not forward that literal token to run.py.
    if forward and forward[0] == "--":
        forward = forward[1:]

    indices = _parse_indices(args.test_indices, start=args.test_start_idx, end=args.test_end_idx)
    indices = sorted(set(indices))
    if not indices:
        print("[parallel_by_sites] No indices to run.", file=sys.stderr)
        return 1

    indices_to_run: list[int] = []
    for idx in indices:
        if args.skip_finished and _is_finished(args.result_dir, idx):
            continue
        indices_to_run.append(idx)
        _ensure_config_exists(args.test_config_base_dir, idx)

    if not indices_to_run:
        print("[parallel_by_sites] All selected tasks appear finished; nothing to run.")
        return 0

    max_parallel = max(1, int(args.max_parallel))
    requested_chunk_size = int(args.chunk_size)
    if requested_chunk_size == 0:
        # Auto-balance: aim for ~max_parallel chunks.
        # chunk_size = ceil(num_tasks / max_parallel)
        chunk_size = max(1, int(math.ceil(len(indices_to_run) / max_parallel)))
    else:
        chunk_size = requested_chunk_size
    chunks = _chunk_indices(sorted(indices_to_run), chunk_size=chunk_size)

    print(
        f"[parallel_by_sites] tasks={len(indices_to_run)} chunks={len(chunks)} "
        f"max_parallel={max_parallel} chunk_size={chunk_size}"
    )

    # Ensure forwarded args do not override the ones we control
    for flag in [
        "--test_config_base_dir",
        "--result_dir",
        "--test_start_idx",
        "--test_end_idx",
        "--test_indices",
    ]:
        forward = _strip_known_arg(forward, flag)

    if args.reset_server_url and args.reset_before_run:
        if args.dry_run:
            print(f"[parallel_by_sites] DRY_RUN: would reset via {args.reset_server_url}")
        else:
            print(f"[parallel_by_sites] Resetting environment via {args.reset_server_url} ...")
            _reset_instance_via_server(
                args.reset_server_url,
                timeout_s=float(args.reset_timeout_s),
                poll_interval_s=float(args.reset_poll_interval_s),
                request_timeout_s=float(args.reset_request_timeout_s),
                max_retries=int(args.reset_max_retries),
            )
            print("[parallel_by_sites] Reset complete; starting batches.")

    commands: list[list[str]] = []
    for chunk in chunks:
        idx_str = ",".join(str(i) for i in chunk)
        cmd = [
            args.python,
            args.run_script,
            "--test_config_base_dir",
            args.test_config_base_dir,
            "--result_dir",
            args.result_dir,
            "--test_indices",
            idx_str,
        ]
        cmd.extend(forward)
        commands.append(cmd)

    if args.dry_run:
        for cmd in commands:
            print("DRY_RUN:", " ".join(cmd))
        return 0

    print("[parallel_by_sites] Running ...")
    rc = _run_with_limit(commands, max_parallel=max_parallel)
    if rc != 0:
        print(f"[parallel_by_sites] Failed with code {rc}", file=sys.stderr)
        return rc

    print("[parallel_by_sites] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

