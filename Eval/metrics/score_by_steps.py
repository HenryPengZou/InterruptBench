"""
Compute step-wise overall accuracy (success@N) from Eval result directories.

This metric matches the user's definition:
- If a task has NOT completed by step n, its score at step n is 0.
- If a task completed before step n, treat it as completed at step n (i.e., carry forward
  its final success/failure).

Step definition (IMPORTANT):
- By default, "step n" means the **post-interrupt action index**:
  - Let `k = interrupt.interrupt_at_action` (the number of replayed actions before injecting
    the user update).
  - Let `stop_step_total` be the 1-based index of the first STOP action in the full action list.
  - We report `stop_step = max(0, stop_step_total - k)`.
  - Therefore, the first action after the interrupt is step=1.
  - If the task would have ended before the interrupt (STOP within replay), step becomes 0.

In this repo, a task "completes" when the action sequence reaches a STOP action
(ActionTypes.STOP == 17) in `RESULT_DIR/trajectories/<task_id>.json`.
Final task score is read from `RESULT_DIR/actions/<task_id>.json` (key: "score").

Usage:
  python score_by_steps.py <result_dir>[,<result_dir2>,...] [--max_action N] [--task_count 165]
  python score_by_steps.py <result_dir> --step_origin global   # use global action index (legacy)

Example:
  python score_by_steps.py test_result_transformed_1update_interrupt_80 --max_action 30
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path


STOP_ACTION_TYPE = 17  # browser_env.actions.ActionTypes.STOP


@dataclass(frozen=True)
class TaskRun:
    score: float
    stop_step: int  # step index (depends on --step_origin); large int if unknown
    interrupt_k: int  # number of replay actions before interrupt (0 if none/unknown)


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _read_json(path: Path) -> dict | list | None:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _parse_task_id_from_filename(p: Path) -> int | None:
    try:
        return int(p.stem)
    except Exception:
        return None


def _extract_stop_step_from_trajectory(traj_json: dict) -> int | None:
    """
    Return 1-based step index of first STOP action within traj_json["actions"].
    If no STOP is found, fall back to len(actions) (if present) to approximate completion.
    """
    actions = traj_json.get("actions", None)
    if not isinstance(actions, list):
        return None

    for i, a in enumerate(actions):
        if isinstance(a, dict) and int(a.get("action_type", -1)) == STOP_ACTION_TYPE:
            return i + 1

    # No explicit STOP found. Some legacy/partial logs might omit it; best-effort fallback.
    return len(actions) if actions else None


def _extract_interrupt_k(*, action_score_json: dict | None, traj_json: dict | None) -> int:
    """
    Return interrupt_at_action (k) if present; otherwise 0.

    In run.py, k is the number of replayed actions before injecting the update.
    """
    for src in (action_score_json, traj_json):
        if not isinstance(src, dict):
            continue
        intr = src.get("interrupt", None)
        if isinstance(intr, dict):
            try:
                k = int(intr.get("interrupt_at_action", 0))
                return max(0, k)
            except Exception:
                continue
    return 0


def _to_step_origin(
    *,
    stop_step_total: int,
    interrupt_k: int,
    step_origin: str,
) -> int:
    """
    Convert a total (global) STOP step index into the configured step origin.
    - global: return stop_step_total (1-based)
    - post_interrupt: return max(0, stop_step_total - k)
    """
    if step_origin == "global":
        return stop_step_total
    if step_origin == "post_interrupt":
        return max(0, stop_step_total - max(0, int(interrupt_k)))
    raise ValueError(f"Unknown step_origin: {step_origin}")


def _load_single_result_dir(result_dir: Path) -> dict[int, TaskRun]:
    """
    Load per-task (score, stop_step) from one RESULT_DIR.
    Missing files are simply skipped; caller decides how to treat missing tasks.
    """
    out: dict[int, TaskRun] = {}

    actions_dir = result_dir / "actions"
    traj_dir = result_dir / "trajectories"

    # First read scores (+ interrupt metadata).
    scores: dict[int, float] = {}
    interrupt_k_from_actions: dict[int, int] = {}
    if actions_dir.exists() and actions_dir.is_dir():
        for p in actions_dir.iterdir():
            if p.suffix != ".json":
                continue
            tid = _parse_task_id_from_filename(p)
            if tid is None:
                continue
            data = _read_json(p)
            if not isinstance(data, dict):
                continue
            s = _safe_float(data.get("score", 0.0), default=0.0)
            scores[tid] = s
            interrupt_k_from_actions[tid] = _extract_interrupt_k(
                action_score_json=data, traj_json=None
            )

    # Then read stop steps (if trajectories exist).
    stop_steps_total: dict[int, int] = {}
    interrupt_k_from_traj: dict[int, int] = {}
    if traj_dir.exists() and traj_dir.is_dir():
        for p in traj_dir.iterdir():
            if p.suffix != ".json":
                continue
            tid = _parse_task_id_from_filename(p)
            if tid is None:
                continue
            data = _read_json(p)
            if not isinstance(data, dict):
                continue
            ss = _extract_stop_step_from_trajectory(data)
            if ss is None:
                continue
            stop_steps_total[tid] = int(ss)
            interrupt_k_from_traj[tid] = _extract_interrupt_k(
                action_score_json=None, traj_json=data
            )

    # Merge (score + stop_step). If stop_step missing, use a large sentinel.
    INF = 10**9
    for tid, s in scores.items():
        k = int(interrupt_k_from_actions.get(tid, interrupt_k_from_traj.get(tid, 0)))
        ss_total = int(stop_steps_total.get(tid, INF))
        out[tid] = TaskRun(score=float(s), stop_step=ss_total, interrupt_k=k)

    # Also include tasks that have trajectory but no actions score file (rare): treat score=0.
    for tid, ss_total in stop_steps_total.items():
        if tid not in out:
            k = int(interrupt_k_from_traj.get(tid, 0))
            out[tid] = TaskRun(score=0.0, stop_step=int(ss_total), interrupt_k=k)

    return out


def _combine_runs(dirs: list[Path]) -> dict[int, TaskRun]:
    """
    Combine multiple result dirs like Eval/score.py:
    - choose the best score per task (max score)
    - if scores tie, prefer smaller stop_step (earlier completion)
    """
    combined: dict[int, TaskRun] = {}
    for d in dirs:
        cur = _load_single_result_dir(d)
        for tid, tr in cur.items():
            prev = combined.get(tid, None)
            if prev is None:
                combined[tid] = tr
                continue
            if tr.score > prev.score:
                combined[tid] = tr
                continue
            if tr.score == prev.score and tr.stop_step < prev.stop_step:
                combined[tid] = tr
                continue
    return combined


def _infer_max_action(result_dir: Path, fallback: int) -> int:
    """
    Try to infer max_action from RESULT_DIR/config.json (args.max_steps),
    otherwise return fallback.
    """
    cfg = _read_json(result_dir / "config.json")
    if isinstance(cfg, dict):
        v = cfg.get("max_steps", None)
        try:
            vv = int(v)
            if vv > 0:
                return vv
        except Exception:
            pass
    return fallback


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result_dirs",
        type=str,
        help="Comma-separated Eval result directories (each should contain actions/ and trajectories/).",
    )
    parser.add_argument(
        "--max_action",
        type=int,
        default=0,
        help="Max step N to report (1..N). If 0, infer from config.json or max observed stop_step.",
    )
    parser.add_argument(
        "--task_count",
        type=int,
        default=165,
        help="Total number of tasks for overall accuracy denominator (default: 165).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output path to write a TSV: step<TAB>overall_accuracy",
    )
    parser.add_argument(
        "--step_origin",
        type=str,
        choices=["post_interrupt", "global"],
        default="post_interrupt",
        help=(
            "Definition of step n. "
            "'post_interrupt' counts actions after interrupt injection (default). "
            "'global' counts from the start of the full action list."
        ),
    )
    args = parser.parse_args()

    dirs = [Path(s.strip()) for s in args.result_dirs.split(",") if s.strip()]
    if not dirs:
        raise SystemExit("No result dirs provided.")
    for d in dirs:
        if not d.exists():
            raise SystemExit(f"Result dir not found: {d}")

    combined_raw = _combine_runs(dirs)
    step_origin = str(args.step_origin)

    # Convert raw (global) stop_step into the requested step origin.
    INF = 10**9
    combined: dict[int, TaskRun] = {}
    for tid, tr in combined_raw.items():
        if tr.stop_step >= INF:
            combined[tid] = tr
            continue
        ss = _to_step_origin(
            stop_step_total=int(tr.stop_step),
            interrupt_k=int(tr.interrupt_k),
            step_origin=step_origin,
        )
        combined[tid] = TaskRun(score=float(tr.score), stop_step=int(ss), interrupt_k=int(tr.interrupt_k))

    # Determine max_action.
    max_action = int(args.max_action)
    if max_action <= 0:
        # Prefer config.json from the first dir if available.
        max_action = _infer_max_action(dirs[0], fallback=0)
    if max_action <= 0:
        # Fallback to max observed stop_step among tasks we can read.
        max_observed = 0
        for tr in combined.values():
            if tr.stop_step < 10**9:
                max_observed = max(max_observed, tr.stop_step)
        max_action = max_observed if max_observed > 0 else 1

    task_count = int(args.task_count)
    if task_count <= 0:
        raise SystemExit("--task_count must be positive")

    lines: list[str] = []
    for n in range(1, max_action + 1):
        success_within_n = 0
        for tid in range(task_count):
            tr = combined.get(tid, None)
            if tr is None:
                continue  # missing => treated as 0
            if tr.score >= 1.0 and tr.stop_step <= n:
                success_within_n += 1

        overall_acc = success_within_n / task_count * 100.0
        lines.append(f"{n}\t{overall_acc:.2f}")

    out_text = "\n".join(lines) + "\n"
    print(out_text, end="")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(out_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

