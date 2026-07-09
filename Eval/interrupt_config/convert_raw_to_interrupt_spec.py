"""
Convert raw interrupt datasets under `interrupt_config/raw/` into `run.py`-compatible
`interrupt_spec.json` format.

Raw format (per item):
{
  "task_id": 0,
  "intent": "...",                       # optional for conversion
  "transformed_initial_intent": "...",    # optional for conversion
  "updates": ["...", "..."]              # REQUIRED
}

Target format (run.py expects):
{
  "tasks": {
    "0": {
      "interrupt_at_action": 5,
      "update_mode": "append",
      "update_intent": "...\n...",
      "extra_steps": 0
    }
  }
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _as_task_id(x: Any) -> str:
    if isinstance(x, bool) or x is None:
        raise ValueError(f"Invalid task_id: {x!r}")
    if isinstance(x, int):
        return str(x)
    if isinstance(x, str) and x.strip():
        # normalize "000" -> "000" (keep as provided)
        return x.strip()
    raise ValueError(f"Invalid task_id: {x!r}")


def _as_updates(x: Any) -> list[str]:
    if not isinstance(x, list):
        raise ValueError(f"Invalid updates (must be list[str]): {x!r}")
    out: list[str] = []
    for u in x:
        if not isinstance(u, str):
            raise ValueError(f"Invalid update (must be str): {u!r}")
        s = u.strip()
        if s:
            out.append(s)
    if not out:
        raise ValueError("updates is empty after stripping")
    return out


def convert_items(
    items: list[dict[str, Any]],
    *,
    interrupt_at_action: int,
    update_mode: str,
    extra_steps: int,
) -> dict[str, Any]:
    if interrupt_at_action < 0:
        raise ValueError("--interrupt_at_action must be >= 0")
    if extra_steps < 0:
        raise ValueError("--extra_steps must be >= 0")
    if update_mode not in ("append", "replace"):
        raise ValueError("--update_mode must be one of: append, replace")

    tasks: dict[str, Any] = {}
    for it in items:
        if not isinstance(it, dict):
            raise ValueError(f"Each item must be an object, got: {type(it).__name__}")
        tid = _as_task_id(it.get("task_id"))
        updates = _as_updates(it.get("updates"))
        update_intent = "\n".join(updates)
        tasks[tid] = {
            "interrupt_at_action": int(interrupt_at_action),
            "update_mode": update_mode,
            "update_intent": update_intent,
            "extra_steps": int(extra_steps),
        }
    return {"tasks": tasks}


def convert_file(
    in_path: Path,
    out_path: Path,
    *,
    interrupt_at_action: int,
    update_mode: str,
    extra_steps: int,
) -> None:
    with open(in_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {in_path}, got {type(data).__name__}")

    spec = convert_items(
        data,
        interrupt_at_action=interrupt_at_action,
        update_mode=update_mode,
        extra_steps=extra_steps,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _default_out_path(out_dir: Path, in_path: Path) -> Path:
    # e.g. raw/1update_opus.json -> interrupt_spec_1update_opus.json
    return out_dir / f"interrupt_spec_{in_path.stem}.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[],
        help="Input raw JSON files. Default: all *.json under interrupt_config/raw/.",
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "raw"),
        help="Directory to scan when --inputs is empty.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Where to write converted interrupt_spec_*.json files.",
    )
    parser.add_argument("--interrupt_at_action", type=int, default=5)
    parser.add_argument("--update_mode", choices=["append", "replace"], default="append")
    parser.add_argument("--extra_steps", type=int, default=0)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if args.inputs:
        in_paths = [Path(p) for p in args.inputs]
    else:
        in_paths = sorted(raw_dir.glob("*.json"))

    if not in_paths:
        raise SystemExit(f"No input files found (raw_dir={raw_dir})")

    for in_path in in_paths:
        out_path = _default_out_path(out_dir, in_path)
        convert_file(
            in_path,
            out_path,
            interrupt_at_action=int(args.interrupt_at_action),
            update_mode=str(args.update_mode),
            extra_steps=int(args.extra_steps),
        )
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

