"""
Create a new dataset JSON file (list of task dicts) whose `intent` field is replaced by
`transformed_initial_intent` from a raw interrupt file.

This matches the format of files like:
  - Eval/config_files/wa/test_webarena_lite.json

Raw interrupt format (per item):
{
  "task_id": 0,
  "transformed_initial_intent": "...",
  "updates": [...]
}

Output:
- a single JSON file: list[dict], with `intent` overridden for tasks that appear in raw.
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
        t_intent = _as_nonempty_str(
            it.get("transformed_initial_intent"),
            field="transformed_initial_intent",
        )
        out[tid] = t_intent
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dataset_json", type=str, required=True)
    parser.add_argument("--raw_interrupt_file", type=str, required=True)
    parser.add_argument("--out_dataset_json", type=str, required=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="If set, fail when any task_id in base dataset has no transformed intent in raw.",
    )
    args = parser.parse_args()

    base_path = Path(args.base_dataset_json)
    raw_path = Path(args.raw_interrupt_file)
    out_path = Path(args.out_dataset_json)

    if not base_path.exists():
        raise SystemExit(f"base_dataset_json not found: {base_path}")
    if not raw_path.exists():
        raise SystemExit(f"raw_interrupt_file not found: {raw_path}")

    tmap = load_transformed_map(raw_path)

    with open(base_path, "r") as f:
        base = json.load(f)
    if not isinstance(base, list):
        raise SystemExit(f"Expected base_dataset_json to be a JSON array, got {type(base).__name__}")

    updated = 0
    missing = 0
    out_list: list[dict] = []
    for it in base:
        if not isinstance(it, dict):
            raise SystemExit(f"Base dataset item is not an object: {type(it).__name__}")
        if "task_id" not in it:
            raise SystemExit("Base dataset item missing 'task_id'")
        tid = _as_task_id(it["task_id"])
        # Always preserve the original intent as `true_intent` (unless already present).
        orig_intent = it.get("intent", "")
        if not isinstance(orig_intent, str):
            orig_intent = str(orig_intent)

        out_item = dict(it)
        if "true_intent" not in out_item:
            out_item["true_intent"] = orig_intent

        if tid in tmap:
            out_item["intent"] = tmap[tid]
            updated += 1
        else:
            missing += 1
            if args.strict:
                raise SystemExit(
                    f"Missing transformed_initial_intent for task_id={tid} in {raw_path}"
                )

        out_list.append(out_item)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_list, f, indent=4, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {out_path} (updated {updated}, missing {missing})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

