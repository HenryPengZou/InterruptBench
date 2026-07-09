"""
Generate config dir + interrupt_spec for *multi-round* interrupt evaluation.

Why this exists
---------------
`Eval/run.py` currently supports a single interrupt per run:
  - replay saved actions up to K
  - inject one intent update
  - continue and save the resulting trajectory

To support *multiple* interrupts (sequential intent updates) following the workflow:
  0) baseline run on transformed initial intent, save trajectories
  1) interrupt at the middle of the trajectory, apply update #1, finish & save
  2) based on the previous interrupt run, pick the middle of the *post-interrupt* actions,
     apply update #2, finish & save
  3) repeat until all updates are used

...we iterate across *multiple runs*. For stage `s` (1-based):
  - replay from result_dir of stage s-1 (baseline is stage 0)
  - choose the interrupt point:
      s==1: k = floor(num_actions / 2)
      s> 1: k_prev = previous stage's interrupt_at_action
            k = k_prev + floor((num_actions - k_prev) / 2)
  - inject update `updates[s-1]`

Additionally, for stage s>=2, we generate a stage-specific config directory where the
task `intent` already includes all *prior* updates (as tags), so when `run.py` resumes
after replay it sees the cumulative "latest" user intent.

This script is *offline*: it only reads JSON files and writes JSON files.
Running evaluation (run.py / parallel_by_sites.py) is done separately.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _as_int(x: Any, *, field: str) -> int:
    if isinstance(x, bool) or x is None:
        raise ValueError(f"Invalid {field}: {x!r}")
    if isinstance(x, int):
        return x
    if isinstance(x, str) and x.strip():
        return int(x.strip())
    raise ValueError(f"Invalid {field}: {x!r}")


def _as_str(x: Any, *, field: str) -> str:
    if not isinstance(x, str):
        raise ValueError(f"Invalid {field} (must be str): {x!r}")
    return x


def _as_nonempty_str(x: Any, *, field: str) -> str:
    s = _as_str(x, field=field).strip()
    if not s:
        raise ValueError(f"{field} is empty")
    return s


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


@dataclass(frozen=True)
class RawTask:
    task_id: int
    transformed_initial_intent: str
    updates: list[str]


def load_raw_tasks(raw_path: Path) -> dict[int, RawTask]:
    data = _load_json(raw_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {raw_path}, got {type(data).__name__}")

    out: dict[int, RawTask] = {}
    for it in data:
        if not isinstance(it, dict):
            raise ValueError(f"Raw item must be object, got {type(it).__name__}")
        tid = _as_int(it.get("task_id"), field="task_id")
        t_intent = _as_nonempty_str(
            it.get("transformed_initial_intent"),
            field="transformed_initial_intent",
        )
        updates_raw = it.get("updates")
        if not isinstance(updates_raw, list):
            raise ValueError(f"updates must be a list[str] for task_id={tid}, got {type(updates_raw).__name__}")
        updates: list[str] = []
        for u in updates_raw:
            if not isinstance(u, str):
                raise ValueError(f"update must be str for task_id={tid}, got {u!r}")
            s = u.strip()
            if s:
                updates.append(s)
        if not updates:
            raise ValueError(f"updates is empty after stripping for task_id={tid}")
        out[tid] = RawTask(task_id=tid, transformed_initial_intent=t_intent, updates=updates)
    return out


def _read_actions_trajectory(traj_path: Path) -> tuple[int, dict | None]:
    """
    Read `result_dir/trajectories/<task_id>.json` created by `Eval/run.py --save_trajectory`.
    Returns (num_actions, interrupt_meta).
    """
    data = _load_json(traj_path)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid trajectory JSON object: {traj_path}")
    num_actions = data.get("num_actions", None)
    if isinstance(num_actions, int):
        n = int(num_actions)
    else:
        actions = data.get("actions", [])
        if not isinstance(actions, list):
            raise ValueError(f"Invalid trajectory actions list: {traj_path}")
        n = int(len(actions))
    interrupt_meta = data.get("interrupt", None)
    if interrupt_meta is not None and not isinstance(interrupt_meta, dict):
        # tolerate non-dict but discard
        interrupt_meta = None
    return n, interrupt_meta


def _format_prior_updates(
    base_intent: str,
    *,
    prior: list[tuple[int, str]],
) -> str:
    intent = base_intent
    for k, upd in prior:
        intent += f"\n\n[User interrupt @action={int(k)}] {upd}"
    return intent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_interrupt_file", type=str, required=True)
    parser.add_argument(
        "--base_config_dir",
        type=str,
        required=True,
        help="Per-task config dir (e.g., transformed-intent config dir), containing <task_id>.json.",
    )
    parser.add_argument(
        "--result_dirs",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Result dirs. Recommended: chronological order baseline, stage1, stage2, ... "
            "For stage=1 you typically pass [baseline]. For stage>=2 you can either:\n"
            "  - pass the full list [baseline, stage1, ..., stage-1] (script will read history from them), OR\n"
            "  - pass only the previous stage result dir [stage-1] IF that run saved multi-interrupt history "
            "(run.py will carry task_spec/config `multi_interrupt` into trajectory interrupt meta)."
        ),
    )
    parser.add_argument(
        "--stage",
        type=int,
        required=True,
        help="Which update stage to generate (1-based). stage=1 uses updates[0].",
    )
    parser.add_argument("--out_config_dir", type=str, required=True)
    parser.add_argument("--out_interrupt_spec", type=str, required=True)
    parser.add_argument("--update_mode", choices=["append", "replace"], default="append")
    parser.add_argument("--extra_steps", type=int, default=0)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing configs/trajectories instead of skipping those task_ids.",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw_interrupt_file)
    base_config_dir = Path(args.base_config_dir)
    result_dirs = [Path(p) for p in args.result_dirs]
    stage = int(args.stage)
    out_config_dir = Path(args.out_config_dir)
    out_spec_path = Path(args.out_interrupt_spec)

    if stage <= 0:
        raise SystemExit("--stage must be >= 1")
    if args.extra_steps < 0:
        raise SystemExit("--extra_steps must be >= 0")
    if not raw_path.exists():
        raise SystemExit(f"raw_interrupt_file not found: {raw_path}")
    if not base_config_dir.exists():
        raise SystemExit(f"base_config_dir not found: {base_config_dir}")
    if not result_dirs:
        raise SystemExit("Need at least 1 --result_dirs entry")

    raw_tasks = load_raw_tasks(raw_path)

    # Stage S replays from:
    # - result_dirs[S-1] when full chronological list is provided, else
    # - the last provided result_dir (commonly: previous stage only).
    replay_result_dir = result_dirs[stage - 1] if len(result_dirs) >= stage else result_dirs[-1]
    replay_traj_dir = replay_result_dir / "trajectories"
    if not replay_traj_dir.exists():
        raise SystemExit(f"Replay trajectories dir not found: {replay_traj_dir}")

    # For prior updates, we prefer reading from the replay trajectory's embedded history
    # (interrupt_meta.multi_interrupt.prior_updates + the previous stage's update itself).
    # If that's missing and we have the full chronological `result_dirs`, we fall back to
    # reading stage1..stage-1 from disk.
    prior_stage_dirs = result_dirs[1:stage] if len(result_dirs) >= stage else []  # stage=1 -> []

    tasks_spec: dict[str, Any] = {}
    manifest: dict[str, Any] = {
        "stage": stage,
        "raw_interrupt_file": str(raw_path),
        "base_config_dir": str(base_config_dir),
        "replay_result_dir": str(replay_result_dir),
        "result_dirs": [str(p) for p in result_dirs],
        "num_tasks_in_raw": len(raw_tasks),
        "written_tasks": 0,
        "skipped_tasks": 0,
        "skips": [],
    }

    out_config_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over task_ids in raw (the intended scope for interrupt experiments).
    for tid, raw_task in sorted(raw_tasks.items(), key=lambda kv: kv[0]):
        updates = raw_task.updates
        if stage > len(updates):
            manifest["skipped_tasks"] += 1
            manifest["skips"].append(
                {"task_id": tid, "reason": f"stage>{len(updates)} (no update #{stage})"}
            )
            continue

        cfg_in = base_config_dir / f"{tid}.json"
        if not cfg_in.exists():
            msg = f"missing base config: {cfg_in}"
            if args.strict:
                raise SystemExit(msg)
            manifest["skipped_tasks"] += 1
            manifest["skips"].append({"task_id": tid, "reason": msg})
            continue

        replay_traj = replay_traj_dir / f"{tid}.json"
        if not replay_traj.exists():
            msg = f"missing replay trajectory: {replay_traj}"
            if args.strict:
                raise SystemExit(msg)
            manifest["skipped_tasks"] += 1
            manifest["skips"].append({"task_id": tid, "reason": msg})
            continue

        # Compute interrupt position k for this stage.
        n_actions_prev, interrupt_prev = _read_actions_trajectory(replay_traj)
        if n_actions_prev < 0:
            n_actions_prev = 0

        if stage == 1:
            k = int(n_actions_prev // 2)
            k_prev = None
        else:
            if not interrupt_prev or "interrupt_at_action" not in interrupt_prev:
                msg = f"missing interrupt meta in prev trajectory (stage={stage-1} expected): {replay_traj}"
                if args.strict:
                    raise SystemExit(msg)
                manifest["skipped_tasks"] += 1
                manifest["skips"].append({"task_id": tid, "reason": msg})
                continue
            k_prev = int(interrupt_prev.get("interrupt_at_action", 0))
            # Clamp k_prev then compute midpoint of post-interrupt segment.
            if k_prev < 0:
                k_prev = 0
            if k_prev > n_actions_prev:
                k_prev = n_actions_prev
            post_len = int(n_actions_prev - k_prev)
            k = int(k_prev + (post_len // 2))

        # Clamp k into [0, n_actions_prev]
        if k < 0:
            k = 0
        if k > n_actions_prev:
            k = n_actions_prev

        # Build prior updates list from earlier stage result dirs.
        prior: list[tuple[int, str]] = []
        if stage >= 2 and interrupt_prev:
            # 1) From embedded history on the replay trajectory (preferred).
            embedded: list[tuple[int, str]] = []
            mi = interrupt_prev.get("multi_interrupt", None)
            if isinstance(mi, dict):
                pu = mi.get("prior_updates", None)
                if isinstance(pu, list):
                    for item in pu:
                        if not isinstance(item, dict):
                            continue
                        kk = item.get("k", None)
                        uu = item.get("update", None)
                        if isinstance(kk, int) and isinstance(uu, str) and uu.strip():
                            embedded.append((int(kk), uu.strip()))
            # Add the previous stage's own update (this run is replaying from stage-1).
            prev_update = str(interrupt_prev.get("update_intent", "")).strip()
            if prev_update:
                embedded.append((int(k_prev or 0), prev_update))
            prior = embedded

        if (stage >= 2) and (len(prior) != (stage - 1)) and prior_stage_dirs:
            # 2) Fallback: reconstruct from explicit stage dirs (baseline, stage1, ...).
            prior = []
            for j, rd in enumerate(prior_stage_dirs, start=1):
                tj = rd / "trajectories" / f"{tid}.json"
                if not tj.exists():
                    msg = f"missing prior stage trajectory (stage={j}): {tj}"
                    if args.strict:
                        raise SystemExit(msg)
                    prior = []
                    break
                _, im = _read_actions_trajectory(tj)
                if not im:
                    msg = f"missing prior interrupt meta (stage={j}): {tj}"
                    if args.strict:
                        raise SystemExit(msg)
                    prior = []
                    break
                kj = int(im.get("interrupt_at_action", 0))
                uj = str(im.get("update_intent", "")).strip()
                if not uj:
                    msg = f"empty prior update_intent (stage={j}): {tj}"
                    if args.strict:
                        raise SystemExit(msg)
                    prior = []
                    break
                prior.append((kj, uj))

        if (stage >= 2) and (len(prior) != (stage - 1)):
            msg = (
                f"cannot reconstruct full prior history for stage={stage} "
                f"(need {stage-1}, got {len(prior)})"
            )
            if args.strict:
                raise SystemExit(msg)
            manifest["skipped_tasks"] += 1
            manifest["skips"].append({"task_id": tid, "reason": msg})
            continue

        # Write stage config file with cumulative prior updates already embedded.
        cfg = _load_json(cfg_in)
        if not isinstance(cfg, dict):
            msg = f"invalid base config JSON object: {cfg_in}"
            if args.strict:
                raise SystemExit(msg)
            manifest["skipped_tasks"] += 1
            manifest["skips"].append({"task_id": tid, "reason": msg})
            continue

        # Keep original intent for debugging if not already present.
        orig_intent = cfg.get("intent", "")
        if "true_intent" not in cfg:
            cfg["true_intent"] = orig_intent if isinstance(orig_intent, str) else str(orig_intent)

        base_intent = raw_task.transformed_initial_intent
        cfg["intent"] = _format_prior_updates(base_intent, prior=prior)
        cfg["multi_interrupt"] = {
            "stage": stage,
            "prior_updates": [{"k": kk, "update": uu} for kk, uu in prior],
        }

        cfg_out = out_config_dir / f"{tid}.json"
        _dump_json(cfg_out, cfg)

        # Build interrupt spec for this stage.
        update_intent = updates[stage - 1]
        tasks_spec[str(tid)] = {
            "interrupt_at_action": int(k),
            "update_mode": str(args.update_mode),
            "update_intent": str(update_intent),
            "extra_steps": int(args.extra_steps),
            # extra metadata (ignored by run.py)
            "multi_interrupt": {
                "stage": stage,
                "k_prev": k_prev,
                "num_actions_prev": int(n_actions_prev),
                "strategy": "mid" if stage == 1 else "post_mid",
                "prior_updates": [{"k": kk, "update": uu} for kk, uu in prior],
            },
        }

        manifest["written_tasks"] += 1

    spec_obj = {"tasks": tasks_spec}
    _dump_json(out_spec_path, spec_obj)
    _dump_json(out_config_dir / f"_multi_interrupt_manifest_stage{stage}.json", manifest)

    print(
        f"Wrote stage={stage} interrupt_spec to {out_spec_path} "
        f"and configs to {out_config_dir} (tasks={manifest['written_tasks']}, skipped={manifest['skipped_tasks']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())