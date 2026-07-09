"""
Analyze action count differences by outcome buckets between two runs:
- no-interrupt run (baseline)
- interrupt run

We compute per-task:
- success_no  = (score_no  >= 1.0)
- success_int = (score_int >= 1.0)
- actions_no  = total actions (from trajectories/<task_id>.json)
- actions_int = post-interrupt actions = max(0, total_actions_int - k)
  where k = interrupt.interrupt_at_action

Optional penalties:
- Early stop penalty: if a trajectory contains a STOP action (action_type==17) with
  answer containing "Early stop", treat actions as max_step (default 30).

We then aggregate the action differences (actions_int - actions_no) for:
- no_success & int_fail
- no_fail & int_success
- both_success
- both_fail
and the marginals:
- no_success, no_fail
- int_success, int_fail

Usage:
  python action_diff_by_outcome.py \
    --no_interrupt_dir <dir> \
    --interrupt_dir <dir> \
    [--task_count 165] [--max_step 30] [--penalize_early_stop] [--missing_policy penalize|ignore]
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path


STOP_ACTION_TYPE = 17  # browser_env.actions.ActionTypes.STOP


@dataclass(frozen=True)
class PerTask:
    task_id: int
    success_no: bool
    success_int: bool
    actions_no: int
    actions_int: int
    diff: int  # actions_int - actions_no
    missing_no: bool
    missing_int: bool
    early_stop_no: bool
    early_stop_int: bool


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


def _trim_trailing_placeholder_stops(actions: list[dict]) -> list[dict]:
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


def _is_early_stopped(actions: list[dict]) -> bool:
    for a in actions:
        if not isinstance(a, dict) or int(a.get("action_type", -1)) != STOP_ACTION_TYPE:
            continue
        ans = a.get("answer", "")
        if ans is not None and "Early stop" in str(ans):
            return True
    return False


def _load_score(result_dir: Path, task_id: int, *, missing_policy: str) -> tuple[float, bool]:
    p = result_dir / "actions" / f"{task_id}.json"
    if not p.exists():
        if missing_policy == "ignore":
            return 0.0, True
        return 0.0, True
    data = _read_json(p)
    if not isinstance(data, dict):
        return 0.0, True
    return _safe_float(data.get("score", 0.0), default=0.0), False


def _load_actions_count(
    result_dir: Path,
    task_id: int,
    *,
    mode: str,  # "no_interrupt" | "interrupt"
    max_step: int,
    penalize_early_stop: bool,
    missing_policy: str,
) -> tuple[int, bool, bool, int]:
    """
    Returns: (actions_count, missing, early_stop, interrupt_k_used)
    """
    traj_path = result_dir / "trajectories" / f"{task_id}.json"
    if not traj_path.exists():
        if missing_policy == "ignore":
            return 0, True, False, 0
        return int(max_step), True, False, 0

    traj_json = _read_json(traj_path)
    if not isinstance(traj_json, dict):
        if missing_policy == "ignore":
            return 0, True, False, 0
        return int(max_step), True, False, 0

    raw_actions = traj_json.get("actions", [])
    if not isinstance(raw_actions, list):
        raw_actions = []
    actions = [a for a in raw_actions if isinstance(a, dict)]
    actions = _trim_trailing_placeholder_stops(actions)

    # interrupt k comes either from this trajectory or the actions/<id>.json
    action_score_json = _read_json(result_dir / "actions" / f"{task_id}.json")
    k = _extract_interrupt_k(action_score_json if isinstance(action_score_json, dict) else None, traj_json)

    early_stop = _is_early_stopped(actions)
    if penalize_early_stop and early_stop:
        return int(max_step), False, True, int(k)

    total = int(len(actions))
    if mode == "interrupt":
        return max(0, total - int(k)), False, False, int(k)
    return total, False, False, 0


def _summary_stats(xs: list[int]) -> dict[str, float]:
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
        "p10": q(0.10),
        "p25": q(0.25),
        "p75": q(0.75),
        "p90": q(0.90),
        "min": float(s[0]),
        "max": float(s[-1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_interrupt_dir", type=str, required=True)
    parser.add_argument("--interrupt_dir", type=str, required=True)
    parser.add_argument("--task_count", type=int, default=165)
    parser.add_argument("--max_step", type=int, default=30)
    parser.add_argument("--penalize_early_stop", action="store_true", default=True)
    parser.add_argument(
        "--missing_policy",
        type=str,
        choices=["penalize", "ignore"],
        default="penalize",
        help="If missing trajectories, either set actions=max_step or ignore in aggregates.",
    )
    parser.add_argument(
        "--print_task_ids",
        action="store_true",
        help="Also print task id lists per bucket (may be long).",
    )
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
    penalize_early_stop = bool(args.penalize_early_stop)

    per_tasks: list[PerTask] = []
    for tid in range(task_count):
        score_no, _missing_score_no = _load_score(no_dir, tid, missing_policy=missing_policy)
        score_int, _missing_score_int = _load_score(int_dir, tid, missing_policy=missing_policy)
        success_no = score_no >= 1.0
        success_int = score_int >= 1.0

        actions_no, missing_no, early_no, _k0 = _load_actions_count(
            no_dir,
            tid,
            mode="no_interrupt",
            max_step=max_step,
            penalize_early_stop=penalize_early_stop,
            missing_policy=missing_policy,
        )
        actions_int, missing_int, early_int, _k1 = _load_actions_count(
            int_dir,
            tid,
            mode="interrupt",
            max_step=max_step,
            penalize_early_stop=penalize_early_stop,
            missing_policy=missing_policy,
        )

        # If ignoring missing, we drop tasks missing in either side from diff-based buckets.
        if missing_policy == "ignore" and (missing_no or missing_int):
            per_tasks.append(
                PerTask(
                    task_id=tid,
                    success_no=success_no,
                    success_int=success_int,
                    actions_no=actions_no,
                    actions_int=actions_int,
                    diff=0,
                    missing_no=missing_no,
                    missing_int=missing_int,
                    early_stop_no=early_no,
                    early_stop_int=early_int,
                )
            )
            continue

        diff = int(actions_int) - int(actions_no)
        per_tasks.append(
            PerTask(
                task_id=tid,
                success_no=success_no,
                success_int=success_int,
                actions_no=int(actions_no),
                actions_int=int(actions_int),
                diff=int(diff),
                missing_no=missing_no,
                missing_int=missing_int,
                early_stop_no=early_no,
                early_stop_int=early_int,
            )
        )

    # Buckets
    buckets: dict[str, list[PerTask]] = {
        "no_success_int_fail": [],
        "no_fail_int_success": [],
        "both_success": [],
        "both_fail": [],
        "no_success": [],
        "no_fail": [],
        "int_success": [],
        "int_fail": [],
    }

    for t in per_tasks:
        if missing_policy == "ignore" and (t.missing_no or t.missing_int):
            # still count marginals on available info? user asked categories; keep them out by default
            continue

        (buckets["no_success"] if t.success_no else buckets["no_fail"]).append(t)
        (buckets["int_success"] if t.success_int else buckets["int_fail"]).append(t)

        if t.success_no and (not t.success_int):
            buckets["no_success_int_fail"].append(t)
        elif (not t.success_no) and t.success_int:
            buckets["no_fail_int_success"].append(t)
        elif t.success_no and t.success_int:
            buckets["both_success"].append(t)
        else:
            buckets["both_fail"].append(t)

    def report_bucket(name: str, tasks: list[PerTask]) -> None:
        diffs = [t.diff for t in tasks]
        no_actions = [t.actions_no for t in tasks]
        int_actions = [t.actions_int for t in tasks]
        early_no = sum(1 for t in tasks if t.early_stop_no)
        early_int = sum(1 for t in tasks if t.early_stop_int)
        miss_no = sum(1 for t in tasks if t.missing_no)
        miss_int = sum(1 for t in tasks if t.missing_int)

        sd = _summary_stats(diffs)
        sa0 = _summary_stats(no_actions)
        sa1 = _summary_stats(int_actions)

        print(f"## {name}")
        print(f"n\t{int(sd.get('n', 0.0))}")
        if sd.get("n", 0.0) > 0:
            print(f"diff_mean\t{sd['mean']:.4f}")
            print(f"diff_median\t{sd['median']:.4f}")
            print(f"diff_p25\t{sd['p25']:.4f}")
            print(f"diff_p75\t{sd['p75']:.4f}")
            print(f"diff_min\t{sd['min']:.4f}")
            print(f"diff_max\t{sd['max']:.4f}")
            print(f"actions_no_mean\t{sa0['mean']:.4f}")
            print(f"actions_int_mean\t{sa1['mean']:.4f}")
        print(f"early_stop_no\t{early_no}")
        print(f"early_stop_int\t{early_int}")
        print(f"missing_no\t{miss_no}")
        print(f"missing_int\t{miss_int}")
        if args.print_task_ids and tasks:
            ids = [t.task_id for t in tasks]
            print("task_ids\t" + ",".join(str(i) for i in ids))
        print("")

    # Print in the order user requested
    report_bucket("no_interrupt_success__interrupt_fail", buckets["no_success_int_fail"])
    report_bucket("no_interrupt_fail__interrupt_success", buckets["no_fail_int_success"])
    report_bucket("both_success", buckets["both_success"])
    report_bucket("both_fail", buckets["both_fail"])
    report_bucket("no_interrupt_success", buckets["no_success"])
    report_bucket("no_interrupt_fail", buckets["no_fail"])
    report_bucket("interrupt_fail", buckets["int_fail"])
    report_bucket("interrupt_success", buckets["int_success"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

