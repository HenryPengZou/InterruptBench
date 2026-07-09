"""
Create a new task-config directory whose `intent` field is replaced by
`transformed_initial_intent` from a raw interrupt file.

Typical workflow:
1) Baseline run with transformed intents (to produce replay trajectories)
2) Interrupt run (replay + inject updates)

This script helps step (1) by generating a new config directory.

Input:
- base config dir: e.g. Eval/config_files/wa/test_webarena_lite/
- raw interrupt json: e.g. Eval/interrupt_config/raw/1update_opus.json
  Each item contains:
    - task_id
    - transformed_initial_intent

Output:
- out dir containing `<task_id>.json` files (copied from base, but with `intent` replaced)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _as_task_id(x: Any) -> int:
    if isinstance(x, bool) or x is None:
        raise ValueError(f"Invalid task_id: {x!r}")
    if isinstance(x, int):
        return x
    if isinstance(x, str) and x.strip():
        return int(x.strip())
    raise ValueError(f"Invalid task_id: {x!r}")


def _as_nonempty_str(x: Any, *, field: str) -> str:
    if not isinstance(x, str):
        raise ValueError(f"Invalid {field} (must be str): {x!r}")
    s = x.strip()
    if not s:
        raise ValueError(f"{field} is empty")
    return s


def load_transformed_map(raw_path: Path) -> dict[int, str]:
    with open(raw_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {raw_path}, got {type(data).__name__}")
    out: dict[int, str] = {}
    for it in data:
        if not isinstance(it, dict):
            raise ValueError(f"Each item must be an object, got {type(it).__name__}")
        tid = _as_task_id(it.get("task_id"))
        t_intent = _as_nonempty_str(it.get("transformed_initial_intent"), field="transformed_initial_intent")
        out[tid] = t_intent
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config_dir", type=str, required=True)
    parser.add_argument("--raw_interrupt_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="If set, fail when any transformed intent is missing for an index in the base dir.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_config_dir)
    raw_path = Path(args.raw_interrupt_file)
    out_dir = Path(args.out_dir)

    if not base_dir.exists():
        raise SystemExit(f"base_config_dir not found: {base_dir}")
    if not raw_path.exists():
        raise SystemExit(f"raw_interrupt_file not found: {raw_path}")

    tmap = load_transformed_map(raw_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    # iterate over base_dir/*.json (task configs are named <idx>.json)
    base_files = sorted(base_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if not base_files:
        raise SystemExit(f"No *.json found under {base_dir}")

    written = 0
    skipped = 0
    for p in base_files:
        if not p.stem.isdigit():
            # skip non-task jsons
            continue
        tid = int(p.stem)
        with open(p, "r") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"Invalid config JSON object: {p}")

        if tid not in tmap:
            if args.strict:
                raise SystemExit(f"Missing transformed_initial_intent for task_id={tid} in {raw_path}")
            skipped += 1
            continue

        # Preserve original intent for scoring/debugging (unless already present).
        orig_intent = cfg.get("intent", "")
        if not isinstance(orig_intent, str):
            orig_intent = str(orig_intent)
        if "true_intent" not in cfg:
            cfg["true_intent"] = orig_intent

        # Replace the actual instruction used by the agent.
        cfg["intent"] = tmap[tid]

        out_path = out_dir / p.name
        with open(out_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
            f.write("\n")
        written += 1

    print(f"Wrote {written} configs to {out_dir} (skipped {skipped})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

