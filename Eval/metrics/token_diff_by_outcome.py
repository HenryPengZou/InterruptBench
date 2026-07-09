"""
Adapt Token Efficiency / Token consumption difference between interrupted and uninterrupted runs.

Per-task token consumption:
- Tokens are computed from each action's `raw_prediction` text in
  `RESULT_DIR/trajectories/<task_id>.json` (actions-only trajectory).
- We exclude STOP actions (action_type==17) from action/token accounting, since STOP is
  created by the runner (e.g., early stop / termination placeholder), not the agent.

Interrupt handling:
- For the interrupt run, we measure tokens ONLY for actions AFTER interruption injection:
  let k = interrupt.interrupt_at_action (number of replayed actions), then we take actions[k:].

Early-stop penalty (requested):
- If a task is early stopped (detected by a STOP action with answer containing "Early stop"),
  then the number of actions for that task is regarded as max_step (default 30).
- Token consumption is extended accordingly:
  - Let existing_actions = number of counted actions (post-interrupt for interrupt run, full for no-interrupt).
  - Let avg_last3 = average token count of the last 3 counted actions (or fewer if <3).
  - Extended token = (max_step - existing_actions) * avg_last3
  - Total token = existing_token + extended_token

We report:
- Overall averages: avg_tokens_no_interrupt, avg_tokens_with_interrupt, diff
- Outcome buckets (based on score>=1.0):
  - no_success & int_fail
  - no_fail & int_success
  - both_success
  - both_fail
  - and marginals: no_success/no_fail, int_success/int_fail

Usage:
  python token_diff_by_outcome.py \
    --no_interrupt_dir <dir> \
    --interrupt_dir <dir> \
    [--task_count 165] [--max_step 30] [--missing_policy penalize|ignore]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

def _ensure_tiktoken_cache_dir() -> None:
    """
    tiktoken may try to write cache files under DATA_GYM_CACHE_DIR (default: /tmp/data-gym-cache).
    On some systems this directory can be owned by root / not writable, causing PermissionError.

    We proactively redirect caches to a user-writable directory if needed.
    """

    def _is_writable_dir(p: Path) -> bool:
        try:
            p.mkdir(parents=True, exist_ok=True)
            test = p / ".write_test"
            test.write_text("ok")
            test.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    # Respect explicit user configuration if it looks writable.
    env_tiktoken = os.environ.get("TIKTOKEN_CACHE_DIR")
    if env_tiktoken and _is_writable_dir(Path(env_tiktoken)):
        return

    env_dg = os.environ.get("DATA_GYM_CACHE_DIR")
    if env_dg and _is_writable_dir(Path(env_dg)):
        return

    # If default /tmp/data-gym-cache is not writable, switch to ~/.cache.
    default_dg = Path("/tmp/data-gym-cache")
    if _is_writable_dir(default_dg):
        return

    xdg_cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    dg_cache = xdg_cache / "data-gym-cache"
    tk_cache = xdg_cache / "tiktoken"
    # Ensure both exist; even if one fails, we'll still set vars best-effort.
    dg_ok = _is_writable_dir(dg_cache)
    tk_ok = _is_writable_dir(tk_cache)
    if dg_ok:
        os.environ["DATA_GYM_CACHE_DIR"] = str(dg_cache)
    if tk_ok:
        os.environ["TIKTOKEN_CACHE_DIR"] = str(tk_cache)


_ensure_tiktoken_cache_dir()

import tiktoken  # noqa: E402  (must come after cache env setup)


STOP_ACTION_TYPE = 17  # browser_env.actions.ActionTypes.STOP


@dataclass(frozen=True)
class PerTask:
    task_id: int
    success_no: bool
    success_int: bool
    tokens_no: float
    tokens_int: float
    diff: float  # tokens_int - tokens_no
    actions_no: int
    actions_int: int
    early_stop_no: bool
    early_stop_int: bool
    missing_no: bool
    missing_int: bool


def _read_json(path: Path) -> dict | list | None:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _trim_trailing_placeholder_stops(actions: list[dict]) -> list[dict]:
    """
    run.py may append a placeholder STOP action with empty answer/raw_prediction on termination.
    Trim such trailing placeholders so they don't affect early-stop detection or slicing.
    """
    out = list(actions)
    while out:
        a = out[-1]
        if not isinstance(a, dict) or int(a.get("action_type", -1)) != STOP_ACTION_TYPE:
            break
        answer = a.get("answer", "")
        raw = a.get("raw_prediction", "")
        if (answer is None or str(answer) == "") and (raw is None or str(raw) == ""):
            out.pop()
            continue
        break
    return out


def _is_early_stopped(all_actions_in_traj: list[dict]) -> bool:
    """
    Detect early stop by scanning STOP actions (even though STOP isn't counted).
    """
    for a in all_actions_in_traj:
        if not isinstance(a, dict) or int(a.get("action_type", -1)) != STOP_ACTION_TYPE:
            continue
        ans = a.get("answer", "")
        if ans is not None and "Early stop" in str(ans):
            return True
    return False


def _extract_interrupt_k(action_score_json: dict | None, traj_json: dict | None) -> int:
    for src in (action_score_json, traj_json):
        if not isinstance(src, dict):
            continue
        intr = src.get("interrupt", None)
        if isinstance(intr, dict):
            try:
                return max(0, int(intr.get("interrupt_at_action", 0)))
            except Exception:
                pass
    return 0


@lru_cache(maxsize=128)
def _encoding_for(provider: str, model: str):
    """
    Best-effort token encoding:
    - If tiktoken knows the model, use it.
    - Otherwise fall back to cl100k_base (works reasonably well as an approximation).
    Note: Bedrock/Claude tokenization isn't exactly cl100k_base; this is an approximation.
    """
    _ = provider  # reserved for future provider-specific handling
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str, *, provider: str, model: str) -> int:
    if not text:
        return 0
    enc = _encoding_for(provider or "", model or "")
    try:
        return len(enc.encode(text))
    except Exception:
        # extremely defensive fallback
        return len(text.split())


def _load_score_success(result_dir: Path, task_id: int) -> tuple[float, bool]:
    p = result_dir / "actions" / f"{task_id}.json"
    if not p.exists():
        return 0.0, False
    data = _read_json(p)
    if not isinstance(data, dict):
        return 0.0, False
    s = _safe_float(data.get("score", 0.0), default=0.0)
    return s, (s >= 1.0)


def _load_actions_and_tokens(
    result_dir: Path,
    task_id: int,
    *,
    mode: str,  # "no_interrupt" | "interrupt"
    max_step: int,
    missing_policy: str,  # "penalize" | "ignore"
) -> tuple[float, int, bool, bool]:
    """
    Returns (tokens, actions_count, early_stop, missing).
    - actions_count excludes STOP actions.
    - tokens sums token counts of `raw_prediction` of counted actions.
    - early_stop detected from STOP actions in the raw trajectory actions.
    """
    traj_path = result_dir / "trajectories" / f"{task_id}.json"
    if not traj_path.exists():
        if missing_policy == "ignore":
            return 0.0, 0, False, True
        return 0.0, int(max_step), False, True

    traj_json = _read_json(traj_path)
    if not isinstance(traj_json, dict):
        if missing_policy == "ignore":
            return 0.0, 0, False, True
        return 0.0, int(max_step), False, True

    provider = str(traj_json.get("provider", "") or "")
    model = str(traj_json.get("model", "") or "")

    raw_actions = traj_json.get("actions", [])
    if not isinstance(raw_actions, list):
        raw_actions = []
    all_actions = [a for a in raw_actions if isinstance(a, dict)]
    all_actions = _trim_trailing_placeholder_stops(all_actions)

    early_stop = _is_early_stopped(all_actions)

    # Determine interrupt k (replay length).
    action_score_json = _read_json(result_dir / "actions" / f"{task_id}.json")
    k = _extract_interrupt_k(action_score_json if isinstance(action_score_json, dict) else None, traj_json)

    # Count only non-STOP actions.
    non_stop = [a for a in all_actions if int(a.get("action_type", -1)) != STOP_ACTION_TYPE]
    if mode == "interrupt":
        non_stop = non_stop[int(k) :] if k > 0 else non_stop

    token_per_action: list[int] = []
    for a in non_stop:
        raw = a.get("raw_prediction", "")
        if raw is None:
            raw = ""
        token_per_action.append(_count_tokens(str(raw), provider=provider, model=model))

    actions_count = len(token_per_action)
    tokens_existing = float(sum(token_per_action))

    if early_stop:
        # Apply penalty: extend actions to max_step and fill extra actions with avg of last 3.
        if actions_count <= 0:
            avg_last3 = 0.0
        else:
            last3 = token_per_action[-3:]
            avg_last3 = float(sum(last3) / len(last3))
        extra_actions = max(0, int(max_step) - int(actions_count))
        tokens_total = tokens_existing + float(extra_actions) * avg_last3
        return float(tokens_total), int(max_step), True, False

    return tokens_existing, actions_count, False, False


def _summary_stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"n": 0.0}
    s = sorted(xs)
    n = len(s)

    def q(p: float) -> float:
        if n == 1:
            return float(s[0])
        idx = int(round((n - 1) * p))
        idx = max(0, min(n - 1, idx))
        return float(s[idx])

    return {
        "n": float(n),
        "mean": float(sum(s) / n),
        "median": float(statistics.median(s)),
        "p25": q(0.25),
        "p75": q(0.75),
        "min": float(s[0]),
        "max": float(s[-1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_interrupt_dir", type=str, required=True)
    parser.add_argument("--interrupt_dir", type=str, required=True)
    parser.add_argument("--task_count", type=int, default=165)
    parser.add_argument("--max_step", type=int, default=30)
    parser.add_argument(
        "--missing_policy",
        type=str,
        choices=["penalize", "ignore"],
        default="penalize",
        help="If missing trajectories, either penalize (keep in denom) or ignore (drop from aggregates).",
    )
    parser.add_argument("--print_task_ids", action="store_true")
    args = parser.parse_args()

    no_dir = Path(args.no_interrupt_dir)
    int_dir = Path(args.interrupt_dir)
    if not no_dir.exists():
        raise SystemExit(f"--no_interrupt_dir not found: {no_dir}")
    if not int_dir.exists():
        raise SystemExit(f"--interrupt_dir not found: {int_dir}")

    task_count = int(args.task_count)
    max_step = int(args.max_step)
    missing_policy = str(args.missing_policy)

    per: list[PerTask] = []
    for tid in range(task_count):
        score_no, success_no = _load_score_success(no_dir, tid)
        score_int, success_int = _load_score_success(int_dir, tid)

        tokens_no, actions_no, early_no, missing_no = _load_actions_and_tokens(
            no_dir,
            tid,
            mode="no_interrupt",
            max_step=max_step,
            missing_policy=missing_policy,
        )
        tokens_int, actions_int, early_int, missing_int = _load_actions_and_tokens(
            int_dir,
            tid,
            mode="interrupt",
            max_step=max_step,
            missing_policy=missing_policy,
        )

        if missing_policy == "ignore" and (missing_no or missing_int):
            # Skip from diff aggregates; still record meta.
            per.append(
                PerTask(
                    task_id=tid,
                    success_no=success_no,
                    success_int=success_int,
                    tokens_no=tokens_no,
                    tokens_int=tokens_int,
                    diff=0.0,
                    actions_no=actions_no,
                    actions_int=actions_int,
                    early_stop_no=early_no,
                    early_stop_int=early_int,
                    missing_no=missing_no,
                    missing_int=missing_int,
                )
            )
            continue

        per.append(
            PerTask(
                task_id=tid,
                success_no=success_no,
                success_int=success_int,
                tokens_no=float(tokens_no),
                tokens_int=float(tokens_int),
                diff=float(tokens_int - tokens_no),
                actions_no=int(actions_no),
                actions_int=int(actions_int),
                early_stop_no=early_no,
                early_stop_int=early_int,
                missing_no=missing_no,
                missing_int=missing_int,
            )
        )

    # Overall (diff-included tasks)
    per_used = [t for t in per if not (missing_policy == "ignore" and (t.missing_no or t.missing_int))]
    overall_tokens_no = [t.tokens_no for t in per_used]
    overall_tokens_int = [t.tokens_int for t in per_used]
    overall_diff = [t.diff for t in per_used]

    print("## overall")
    print(f"n\t{len(per_used)}")
    if per_used:
        print(f"avg_tokens_no_interrupt\t{sum(overall_tokens_no)/len(overall_tokens_no):.4f}")
        print(f"avg_tokens_with_interrupt\t{sum(overall_tokens_int)/len(overall_tokens_int):.4f}")
        print(f"avg_diff_with_minus_without\t{sum(overall_diff)/len(overall_diff):.4f}")
    print(f"early_stop_no_interrupt\t{sum(1 for t in per_used if t.early_stop_no)}")
    print(f"early_stop_with_interrupt\t{sum(1 for t in per_used if t.early_stop_int)}")
    print(f"missing_no_interrupt\t{sum(1 for t in per if t.missing_no)}")
    print(f"missing_with_interrupt\t{sum(1 for t in per if t.missing_int)}")
    print("")

    # Buckets
    buckets: dict[str, list[PerTask]] = {
        "no_success__int_fail": [],
        "no_fail__int_success": [],
        "both_success": [],
        "both_fail": [],
        "no_success": [],
        "no_fail": [],
        "int_success": [],
        "int_fail": [],
    }

    for t in per_used:
        (buckets["no_success"] if t.success_no else buckets["no_fail"]).append(t)
        (buckets["int_success"] if t.success_int else buckets["int_fail"]).append(t)
        if t.success_no and (not t.success_int):
            buckets["no_success__int_fail"].append(t)
        elif (not t.success_no) and t.success_int:
            buckets["no_fail__int_success"].append(t)
        elif t.success_no and t.success_int:
            buckets["both_success"].append(t)
        else:
            buckets["both_fail"].append(t)

    def report(name: str, tasks: list[PerTask]) -> None:
        diffs = [t.diff for t in tasks]
        s = _summary_stats(diffs)
        print(f"## {name}")
        print(f"n\t{int(s.get('n', 0.0))}")
        if s.get("n", 0.0) > 0:
            print(f"diff_mean\t{s['mean']:.4f}")
            print(f"diff_median\t{s['median']:.4f}")
            print(f"diff_p25\t{s['p25']:.4f}")
            print(f"diff_p75\t{s['p75']:.4f}")
            print(f"diff_min\t{s['min']:.4f}")
            print(f"diff_max\t{s['max']:.4f}")
            print(f"tokens_no_mean\t{sum(t.tokens_no for t in tasks)/len(tasks):.4f}")
            print(f"tokens_int_mean\t{sum(t.tokens_int for t in tasks)/len(tasks):.4f}")
        if args.print_task_ids and tasks:
            print("task_ids\t" + ",".join(str(t.task_id) for t in tasks))
        print("")

    # Same order as action_diff_by_outcome.py
    report("no_interrupt_success__interrupt_fail", buckets["no_success__int_fail"])
    report("no_interrupt_fail__interrupt_success", buckets["no_fail__int_success"])
    report("both_success", buckets["both_success"])
    report("both_fail", buckets["both_fail"])
    report("no_interrupt_success", buckets["no_success"])
    report("no_interrupt_fail", buckets["no_fail"])
    report("interrupt_fail", buckets["int_fail"])
    report("interrupt_success", buckets["int_success"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

