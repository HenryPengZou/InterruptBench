"""
Adapt Action Efficiency metric.

Definition (as requested):
- Per-task action count:
  - For the interrupt run: number of actions AFTER the interruption is injected
    (i.e., post-interrupt index). If the task replays k actions before injection,
    then post_actions = max(0, total_actions - k).
  - For the non-interrupt run: total number of actions (k=0).
- Early-stop penalty:
  - If the agent is "early stopped" for a task, then the action count for that task
    is set to max_step (default: 30).
  - We detect early stop by the presence of a STOP action (action_type==17) whose
    "answer" contains the substring "Early stop".

Output:
- avg_actions_no_interrupt
- avg_actions_with_interrupt
- difference = (with_interrupt - no_interrupt)

Usage:
  python adapt_action_efficiency.py \
    --no_interrupt_dir <result_dir_without_interrupt> \
    --interrupt_dir <result_dir_with_interrupt> \
    [--task_count 165] [--max_step 30] [--missing_policy penalize|ignore]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


STOP_ACTION_TYPE = 17  # browser_env.actions.ActionTypes.STOP


@dataclass(frozen=True)
class TaskCounts:
    actions: int
    early_stopped: bool
    missing: bool
    interrupt_k: int
    total_actions_observed: int


def _read_json(path: Path) -> dict | list | None:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_interrupt_k(*, action_score_json: dict | None, traj_json: dict | None) -> int:
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
    """
    run.py may append a placeholder STOP action with empty answer for terminated episodes:
    create_stop_action("")
    This is not an LLM-decided action; trim it from the end to keep counts stable.
    """
    out = list(actions)
    while out:
        a = out[-1]
        if not isinstance(a, dict):
            break
        if int(a.get("action_type", -1)) != STOP_ACTION_TYPE:
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
        if not isinstance(a, dict):
            continue
        if int(a.get("action_type", -1)) != STOP_ACTION_TYPE:
            continue
        ans = a.get("answer", "")
        if ans is None:
            continue
        if "Early stop" in str(ans):
            return True
    return False


def _load_task_counts(
    result_dir: Path,
    *,
    task_id: int,
    max_step: int,
    mode: str,  # "interrupt" | "no_interrupt"
    missing_policy: str,  # "penalize" | "ignore"
) -> TaskCounts | None:
    """
    Return per-task action count, or None if missing_policy=="ignore" and task is missing.
    """
    actions_score_path = result_dir / "actions" / f"{task_id}.json"
    traj_path = result_dir / "trajectories" / f"{task_id}.json"

    action_score_json = _read_json(actions_score_path) if actions_score_path.exists() else None
    traj_json = _read_json(traj_path) if traj_path.exists() else None

    if not isinstance(traj_json, dict):
        if missing_policy == "ignore":
            return None
        return TaskCounts(
            actions=int(max_step),
            early_stopped=False,
            missing=True,
            interrupt_k=0,
            total_actions_observed=0,
        )

    raw_actions = traj_json.get("actions", [])
    if not isinstance(raw_actions, list):
        raw_actions = []
    actions_list = [a for a in raw_actions if isinstance(a, dict)]
    actions_list = _trim_trailing_placeholder_stops(actions_list)

    k = _extract_interrupt_k(action_score_json=action_score_json if isinstance(action_score_json, dict) else None, traj_json=traj_json)
    total_actions = len(actions_list)

    early_stopped = _is_early_stopped(actions_list)
    if early_stopped:
        return TaskCounts(
            actions=int(max_step),
            early_stopped=True,
            missing=False,
            interrupt_k=int(k),
            total_actions_observed=int(total_actions),
        )

    if mode == "interrupt":
        post_actions = max(0, total_actions - int(k))
        return TaskCounts(
            actions=int(post_actions),
            early_stopped=False,
            missing=False,
            interrupt_k=int(k),
            total_actions_observed=int(total_actions),
        )

    # no_interrupt
    return TaskCounts(
        actions=int(total_actions),
        early_stopped=False,
        missing=False,
        interrupt_k=0,
        total_actions_observed=int(total_actions),
    )


def _avg(xs: list[int]) -> float:
    return (sum(xs) / len(xs)) if xs else 0.0


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
        help="If a task trajectory is missing, either count it as max_step or ignore it.",
    )
    args = parser.parse_args()

    no_dir = Path(args.no_interrupt_dir)
    intr_dir = Path(args.interrupt_dir)
    if not no_dir.exists():
        raise SystemExit(f"--no_interrupt_dir not found: {no_dir}")
    if not intr_dir.exists():
        raise SystemExit(f"--interrupt_dir not found: {intr_dir}")

    task_count = int(args.task_count)
    max_step = int(args.max_step)
    if task_count <= 0:
        raise SystemExit("--task_count must be positive")
    if max_step <= 0:
        raise SystemExit("--max_step must be positive")

    missing_policy = str(args.missing_policy)

    counts_no: list[int] = []
    counts_intr: list[int] = []
    early_no = 0
    early_intr = 0
    missing_no = 0
    missing_intr = 0

    for tid in range(task_count):
        c0 = _load_task_counts(
            no_dir,
            task_id=tid,
            max_step=max_step,
            mode="no_interrupt",
            missing_policy=missing_policy,
        )
        c1 = _load_task_counts(
            intr_dir,
            task_id=tid,
            max_step=max_step,
            mode="interrupt",
            missing_policy=missing_policy,
        )

        if c0 is None:
            missing_no += 1
        else:
            counts_no.append(int(c0.actions))
            if c0.early_stopped:
                early_no += 1
            if c0.missing:
                missing_no += 1

        if c1 is None:
            missing_intr += 1
        else:
            counts_intr.append(int(c1.actions))
            if c1.early_stopped:
                early_intr += 1
            if c1.missing:
                missing_intr += 1

    avg_no = _avg(counts_no)
    avg_intr = _avg(counts_intr)
    diff = avg_intr - avg_no

    # Print a compact, parseable summary.
    print(f"avg_actions_no_interrupt\t{avg_no:.4f}")
    print(f"avg_actions_with_interrupt\t{avg_intr:.4f}")
    print(f"diff_with_minus_without\t{diff:.4f}")
    print(f"early_stop_no_interrupt\t{early_no}")
    print(f"early_stop_with_interrupt\t{early_intr}")
    print(f"missing_no_interrupt\t{missing_no}")
    print(f"missing_with_interrupt\t{missing_intr}")
    print(f"denominator_no_interrupt\t{len(counts_no)}")
    print(f"denominator_with_interrupt\t{len(counts_intr)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

