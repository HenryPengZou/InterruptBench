"""Script to run end-to-end evaluation on the benchmark.

Modified from https://github.com/web-arena-x/webarena/blob/main/run.py.
"""
import argparse
import glob
import json
import logging
import math
import os
import random
import subprocess
import tempfile
import time
import html as _html
from pathlib import Path
from typing import List

import openai
import requests
import torch
import numpy as np
from PIL import Image

from agent import (
    PromptAgent,
    construct_agent,
)
from agent.prompts import *
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import is_equivalent
from browser_env.auto_login import get_site_comb_from_filepath
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from evaluation_harness import evaluator_router, image_utils

DATASET = os.environ["DATASET"]

LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

def _json_safe(obj):
    """Convert common numpy-ish objects into JSON-serializable forms."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # numpy scalar
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    return obj


def _json_safe_loose(obj):
    """
    Best-effort JSON serializer:
    - handles numpy-ish objects like `_json_safe`
    - converts unknown / non-serializable objects to `repr(obj)`
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # numpy scalar
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            # JSON keys must be strings
            try:
                kk = str(k)
            except Exception:
                kk = repr(k)
            out[kk] = _json_safe_loose(v)
        return out
    if isinstance(obj, list):
        return [_json_safe_loose(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe_loose(v) for v in obj]
    return repr(obj)


def _state_info_jsonable(state_info: dict) -> dict:
    """
    Convert StateInfo to a JSON-friendly dict.

    NOTE: `ScriptBrowserEnv`'s `info` often contains a live Playwright `page` object
    which is not JSON-serializable. We replace it with a minimal representation.
    """
    obs = state_info.get("observation", None)
    info = state_info.get("info", None)

    info_out = {}
    if isinstance(info, dict):
        for k, v in info.items():
            if k == "page":
                # Replace page object with URL if available.
                try:
                    url = getattr(v, "url", None)
                except Exception:
                    url = None
                info_out["page"] = {"url": url} if url is not None else None
            else:
                info_out[k] = _json_safe_loose(v)
    else:
        info_out = _json_safe_loose(info)

    return {
        "observation": _json_safe_loose(obs),
        "info": info_out,
    }


def _trajectory_jsonable(trajectory: "Trajectory") -> list:
    """
    Convert a full in-memory Trajectory (StateInfo/Action alternating list) into
    a JSON-friendly list.
    """
    out: list = []
    for item in trajectory:
        if isinstance(item, dict) and ("observation" in item) and ("info" in item):
            out.append({"type": "state", **_state_info_jsonable(item)})
        elif isinstance(item, dict) and ("action_type" in item):
            out.append({"type": "action", **_json_safe_loose(item)})
        else:
            out.append({"type": "unknown", "value": _json_safe_loose(item)})
    return out


def _action_from_jsonable(action_dict: dict) -> Action:
    """Convert JSON-loaded action back to env-consumable Action."""
    a = dict(action_dict)
    # coords is stored as list; env action executor expects numpy array
    if isinstance(a.get("coords", None), list):
        a["coords"] = np.array(a["coords"], dtype=np.float32)
    # action_type might be numpy scalar or string in some edge cases
    if "action_type" in a:
        try:
            a["action_type"] = int(a["action_type"])
        except Exception:
            pass
    return a  # type: ignore[return-value]


def _save_actions_only(
    trajectory: Trajectory,
    out_path: Path,
    *,
    task_id: int | str,
    config_file: str,
    intent: str,
    provider: str,
    model: str,
    mode: str,
    action_set_tag: str,
    interrupt_meta: dict | None = None,
) -> None:
    """Persist only the action sequence (JSON-friendly) for replay."""
    actions = trajectory[1::2]  # Action at odd indices
    payload = {
        "version": 1,
        "task_id": task_id,
        "config_file": str(config_file),
        "intent": intent,
        "provider": provider,
        "model": model,
        "mode": mode,
        "action_set_tag": action_set_tag,
        "num_actions": len(actions),
        "actions": _json_safe(actions),
        "interrupt": interrupt_meta,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def _load_replay_actions(replay_path: Path) -> list[Action]:
    with open(replay_path, "r") as f:
        data = json.load(f)
    actions_raw = data.get("actions", [])
    if not isinstance(actions_raw, list):
        raise ValueError(f"Invalid replay file: {replay_path} (actions is not a list)")
    return [_action_from_jsonable(a) for a in actions_raw]


def _freeze_page_url(info: dict) -> None:
    """
    `info["page"]` is a live Playwright object that changes over time.
    Store a snapshot URL string into `info["page_url"]` so we can export
    trajectories later with correct per-step URLs.
    """
    if not isinstance(info, dict):
        return
    if "page_url" in info:
        return
    try:
        page = info.get("page", None)
        url = getattr(page, "url", None)
    except Exception:
        url = None
    info["page_url"] = url


def _truncate_text(s: str, *, max_chars: int) -> str:
    if max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"\n\n... (truncated, {len(s)} chars total)"


def _save_full_trajectory_split_html(
    trajectory: Trajectory,
    out_path: Path,
    *,
    task_id: int | str,
    config_file: str,
    provider: str,
    model: str,
    mode: str,
    action_set_tag: str,
    intent_before_interrupt: str | None,
    intent_after_interrupt: str | None,
    split_at: int | None,
    interrupt_meta: dict | None,
    max_obs_chars: int = 20000,
) -> None:
    """
    Save the full (state+action) trajectory as HTML (much more readable than JSON).
    Clearly separates `before_interrupt` / `after_interrupt` at `split_at`.

    NOTE: We do NOT embed screenshots here; we only render text observations and action dicts
    to keep file sizes reasonable.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = f"""
<!DOCTYPE html>
<head>
  <meta charset="utf-8"/>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }}
    pre {{ white-space: pre-wrap; word-wrap: break-word; background: #f6f8fa; padding: 8px; border-radius: 6px; }}
    details {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 8px; margin: 10px 0; }}
    summary {{ cursor: pointer; font-weight: 600; }}
    .meta {{ color: #374151; }}
    .marker {{ padding: 10px; margin: 16px 0; border-radius: 10px; }}
    .before {{ background: #fff7ed; border: 1px solid #fdba74; }}
    .after  {{ background: #ecfeff; border: 1px solid #67e8f9; }}
  </style>
</head>
<html><body>
  <h1>Full trajectory (interrupt split)</h1>
  <div class="meta"><b>task_id</b>: {_html.escape(str(task_id))}</div>
  <div class="meta"><b>config_file</b>: {_html.escape(str(config_file))}</div>
  <div class="meta"><b>provider/model/mode</b>: {_html.escape(str(provider))} / {_html.escape(str(model))} / {_html.escape(str(mode))}</div>
  <div class="meta"><b>action_set_tag</b>: {_html.escape(str(action_set_tag))}</div>
  <div class="meta"><b>split_at</b>: {_html.escape(str(split_at))}</div>
  <h2>Intent (before interrupt)</h2>
  <pre>{_html.escape(str(intent_before_interrupt or ""))}</pre>
  <h2>Intent (after interrupt)</h2>
  <pre>{_html.escape(str(intent_after_interrupt or ""))}</pre>
  <h2>Interrupt meta</h2>
  <pre>{_html.escape(json.dumps(_json_safe_loose(interrupt_meta or {}), indent=2, ensure_ascii=False))}</pre>
  <div class="marker before"><b>Before interrupt (includes replay segment)</b></div>
"""

    parts: list[str] = [header]
    for idx, item in enumerate(trajectory):
        if split_at is not None and idx == split_at:
            parts.append('<div class="marker after"><b>After interrupt (post-update execution)</b></div>')

        if isinstance(item, dict) and ("observation" in item) and ("info" in item):
            info = item.get("info", {}) if isinstance(item.get("info", {}), dict) else {}
            page_url = None
            try:
                page_url = info.get("page_url", None)
            except Exception:
                page_url = None

            obs = item.get("observation", {})
            text_obs = ""
            try:
                if isinstance(obs, dict) and "text" in obs:
                    text_obs = str(obs.get("text", ""))
                else:
                    text_obs = str(obs)
            except Exception:
                text_obs = repr(obs)
            text_obs = _truncate_text(text_obs, max_chars=max_obs_chars)

            summary = f"State[{idx}]"
            if page_url:
                summary += f" URL={page_url}"
            parts.append(
                "<details open>"
                f"<summary>{_html.escape(summary)}</summary>"
                f"<pre>{_html.escape(text_obs)}</pre>"
                "</details>"
            )
            continue

        if isinstance(item, dict) and ("action_type" in item):
            action_pretty = json.dumps(_json_safe_loose(item), indent=2, ensure_ascii=False)
            parts.append(
                "<details open>"
                f"<summary>{_html.escape(f'Action[{idx}]')}</summary>"
                f"<pre>{_html.escape(action_pretty)}</pre>"
                "</details>"
            )
            continue

        # Unknown item type (shouldn't happen)
        parts.append(
            "<details>"
            f"<summary>{_html.escape(f'Item[{idx}] (unknown)')}</summary>"
            f"<pre>{_html.escape(repr(item))}</pre>"
            "</details>"
        )

    parts.append("</body></html>\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(parts))


def _load_interrupt_spec(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r") as f:
        spec = json.load(f)
    if not isinstance(spec, dict):
        raise ValueError("--interrupt_spec must be a JSON object")
    return spec


def _resolve_interrupt_at_action(task_spec: dict, *, num_replay_actions: int) -> tuple[int, dict]:
    """
    Resolve the effective interrupt point (K) for replay.

    Supports two ways to specify the interruption point in `interrupt_spec`:
    - Fixed steps: `interrupt_at_action: <int>`
    - Percentage of the saved action sequence:
        - `interrupt_at_pct: 0.2`  (0..1)
        - `interrupt_at_pct: "20%"` (string percent)
        - `interrupt_at_pct: 20` (treated as 20%)

    The effective K is computed as:
      K = floor(pct * num_replay_actions)
    and then clamped into [0, num_replay_actions].
    """
    raw_pct = None
    if isinstance(task_spec, dict):
        raw_pct = task_spec.get("interrupt_at_pct", None)
        if raw_pct is None:
            raw_pct = task_spec.get("interrupt_at_percent", None)

    meta: dict = {}
    k: int

    if raw_pct is not None:
        # Parse pct from str/number.
        pct: float
        if isinstance(raw_pct, str):
            s = raw_pct.strip()
            if s.endswith("%"):
                s = s[:-1].strip()
                pct = float(s) / 100.0
            else:
                pct = float(s)
        elif isinstance(raw_pct, (int, float)):
            pct = float(raw_pct)
            # Heuristic: treat 20 as 20%
            if pct > 1.0:
                pct = pct / 100.0
        else:
            raise ValueError(f"interrupt_at_pct must be number or string, got: {type(raw_pct).__name__}")

        if not (pct >= 0.0):
            raise ValueError(f"interrupt_at_pct must be >= 0, got: {raw_pct!r}")

        k = int(math.floor(pct * float(num_replay_actions)))
        meta = {
            "mode": "pct",
            "pct_raw": raw_pct,
            "pct": pct,
        }
    else:
        raw_k = 0
        if isinstance(task_spec, dict):
            raw_k = int(task_spec.get("interrupt_at_action", 0))
        k = raw_k
        meta = {"mode": "fixed", "interrupt_at_action": raw_k}

    # Clamp into valid range
    if k < 0:
        k = 0
    if k > int(num_replay_actions):
        k = int(num_replay_actions)
    meta["resolved_k"] = k
    meta["num_replay_actions"] = int(num_replay_actions)
    return k, meta


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
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.get(reset_url, timeout=request_timeout_s)
            if r.status_code in (200, 418):
                break
            last_err = RuntimeError(
                f"reset server returned status={r.status_code} for GET {reset_url}"
            )
        except Exception as e:
            last_err = e
        time.sleep(min(2.0, poll_interval_s))
    else:
        raise RuntimeError(
            f"Failed to trigger reset via {reset_url}: {repr(last_err)}"
        )

    # Poll status until ready or timeout.
    start = time.time()
    while True:
        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for reset server to become ready: {status_url}"
            )
        try:
            s = requests.get(status_url, timeout=request_timeout_s)
            if s.status_code == 500:
                raise RuntimeError(
                    f"Reset server reports failure (500) at {status_url}: {s.text}"
                )
            if s.status_code == 200:
                body = (s.text or "").lower()
                if ("ready" in body) and ("duty" in body):
                    return
                # "reset ongoing" -> keep waiting
        except Exception:
            # transient network issue, keep polling
            pass
        time.sleep(poll_interval_s)


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the browser"
    )

    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=[
            "accessibility_tree",
            "accessibility_tree_with_captioner",
            "html",
            "image",
            "image_som",
            # WebRL plain-text setting (supported by ScriptBrowserEnv)
            "webrl",
        ],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=2048)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)

    parser.add_argument("--max_steps", type=int, default=50)

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="agents/prompts/state_action_agent.json",
    )
    parser.add_argument(
        "--parsing_failure_th",
        help="When consecutive parsing failures exceed this threshold, the agent will terminate early.",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When consecutive repeated actions exceed this threshold, the agent will terminate early.",
        type=int,
        default=5,
    )

    parser.add_argument("--test_config_base_dir", type=str)

    parser.add_argument(
        "--eval_captioning_model_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run eval captioning model on. By default, runs it on CPU.",
    )
    parser.add_argument(
        "--eval_captioning_model",
        type=str,
        default="Salesforce/blip2-flan-t5-xl",
        choices=["Salesforce/blip2-flan-t5-xl"],
        help="Captioning backbone for VQA-type evals.",
    )
    parser.add_argument(
        "--captioning_model",
        type=str,
        default="Salesforce/blip2-flan-t5-xl",
        choices=["Salesforce/blip2-flan-t5-xl", "llava-hf/llava-1.5-7b-hf"],
        help="Captioning backbone for accessibility tree alt text.",
    )

    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="AWS region for Bedrock (e.g., us-east-2). If not set, will use BEDROCK_REGION/AWS_REGION/AWS_DEFAULT_REGION.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=3840,
    )

    # example config
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=910)
    parser.add_argument(
        "--test_indices",
        type=str,
        default=None,
        help=(
            "Optional comma-separated list of test indices to run, e.g. '0,5,7'. "
            "When set, overrides --test_start_idx/--test_end_idx."
        ),
    )

    # logging related
    parser.add_argument("--result_dir", type=str, default="")
    parser.add_argument(
        "--save_trajectory",
        action="store_true",
        help="Save per-task action sequence for later replay (result_dir/trajectories/<task_id>.json).",
    )
    parser.add_argument(
        "--trajectory_dir",
        type=str,
        default=None,
        help="Where to save trajectories. Default: <result_dir>/trajectories",
    )
    parser.add_argument(
        "--interrupt_spec",
        type=str,
        default=None,
        help=(
            "Path to a JSON spec that defines user interrupts. "
            "When set together with --replay_trajectory_dir, the runner will replay "
            "recorded actions up to interrupt_at_action and then continue with updated intent."
        ),
    )
    parser.add_argument(
        "--replay_trajectory_dir",
        type=str,
        default=None,
        help="Directory containing saved per-task trajectories (trajectories/<task_id>.json) for replay.",
    )
    parser.add_argument(
        "--reset_server_url",
        type=str,
        default=None,
        help=(
            "If set, will trigger a full environment reset via GET <reset_server_url>/reset "
            "and poll <reset_server_url>/status until Ready."
        ),
    )
    parser.add_argument(
        "--reset_before_run",
        action="store_true",
        help=(
            "Trigger reset_server ONCE before the evaluation loop starts. "
            "Recommended for interrupt/replay evaluation to avoid cross-run contamination."
        ),
    )
    parser.add_argument(
        "--reset_before_each_task",
        action="store_true",
        help="Trigger reset_server before every task (strongest isolation; slow).",
    )
    parser.add_argument(
        "--reset_before_replay_only",
        action="store_true",
        help="Only reset before tasks that are running in replay mode (i.e., have interrupt_spec entry + replay_trajectory_dir).",
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
    
    # optional: self-deployed model endpoint (OpenAI-compatible)
    parser.add_argument("--planner_ip", type=str, default=None)
    args = parser.parse_args()

    # check the whether the action space is compatible with the observation space
    if (
        args.action_set_tag == "id_accessibility_tree"
        and args.observation_type
        not in [
            "accessibility_tree",
            "accessibility_tree_with_captioner",
            "image_som",
        ]
    ):
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )

    return args


def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to stop early"""

    # reach the max step
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
            [
                action["action_type"] == ActionTypes.NONE
                for action in last_k_actions
            ]
        ):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k:
            if all(
                [
                    is_equivalent(action, last_action)
                    for action in last_k_actions
                ]
            ):
                return True, f"Same action for {k} times"

    else:
        # check the action sequence
        if (
            sum([is_equivalent(action, last_action) for action in action_seq])
            >= k
        ):
            return True, f"Same typing action for {k} times"

    return False, ""


def test(
    args: argparse.Namespace,
    config_file_list: list[str]
) -> None:
    scores = []
    base_max_steps = args.max_steps
    actions_dir = Path(args.result_dir) / "actions"
    actions_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = (
        Path(args.trajectory_dir)
        if args.trajectory_dir is not None
        else Path(args.result_dir) / "trajectories"
    )
    full_trajectory_dir = trajectory_dir / "full"

    interrupt_spec = _load_interrupt_spec(args.interrupt_spec)
    replay_dir = Path(args.replay_trajectory_dir) if args.replay_trajectory_dir else None

    early_stop_thresholds = {
        "parsing_failure": args.parsing_failure_th,
        "repeating_action": args.repeating_action_failure_th,
    }

    if args.observation_type in [
        "accessibility_tree_with_captioner",
        "image_som",
    ]:
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        caption_image_fn = image_utils.get_captioning_fn(
            device, dtype, args.captioning_model
        )
    else:
        caption_image_fn = None

    # Load a (possibly different) captioning model for running VQA evals.
    if DATASET == 'visualwebarena':
        if (
            caption_image_fn
            and args.eval_captioning_model == args.captioning_model
        ):
            eval_caption_image_fn = caption_image_fn
        else:
            eval_caption_image_fn = image_utils.get_captioning_fn(
                args.eval_captioning_model_device,
                torch.float16
                if (
                    torch.cuda.is_available()
                    and args.eval_captioning_model_device == "cuda"
                )
                else torch.float32,
                args.eval_captioning_model,
            )
    else:
        caption_image_fn = None
        eval_caption_image_fn = None

    agent = construct_agent(
        args,
        captioning_fn=caption_image_fn
        if args.observation_type == "accessibility_tree_with_captioner"
        else None,
    )  # NOTE: captioning_fn here is used for captioning input images.

    env = ScriptBrowserEnv(
        headless=not args.render,
        slow_mo=args.slow_mo,
        observation_type=args.observation_type,
        current_viewport_only=args.current_viewport_only,
        viewport_size={
            "width": args.viewport_width,
            "height": args.viewport_height,
        },
        save_trace_enabled=args.save_trace_enabled,
        sleep_after_execution=args.sleep_after_execution,
        # NOTE: captioning_fn here is used for LLM + captioning baselines.
        # This can be different from the captioning model used for evals.
        captioning_fn=caption_image_fn,
    )

    # Optional: reset the whole webarena docker instance once before running tasks.
    # This matches the use case "same task reruns contaminate each other", while
    # assuming different task_ids within one run are independent.
    if args.reset_server_url and args.reset_before_run:
        try:
            env.close()
        except Exception:
            pass
        logger.info(
            f"[Reset] Triggering full instance reset (before run) via {args.reset_server_url}"
        )
        _reset_instance_via_server(
            args.reset_server_url,
            timeout_s=args.reset_timeout_s,
            poll_interval_s=args.reset_poll_interval_s,
            request_timeout_s=args.reset_request_timeout_s,
            max_retries=args.reset_max_retries,
        )
        logger.info("[Reset] Instance is ready (before run)")

    for config_file in config_file_list:
        task_id = None
        interrupt_meta: dict | None = None
        split_at: int | None = None
        intent_before_interrupt: str | None = None
        task_max_steps = base_max_steps
        try:
            render_helper = RenderHelper(
                config_file, args.result_dir, args.action_set_tag
            )

            # Load task.
            with open(config_file) as f:
                _c = json.load(f)
                intent = _c["intent"]
                task_id = _c["task_id"]
                image_paths = _c.get("image", None)
                images = []

                # automatically login
                if _c["storage_state"]:
                    cookie_file_name = os.path.basename(_c["storage_state"])
                    comb = get_site_comb_from_filepath(cookie_file_name)
                    temp_dir = tempfile.mkdtemp()
                    # subprocess to renew the cookie
                    subprocess.run(
                        [
                            "python",
                            "browser_env/auto_login.py",
                            "--auth_folder",
                            temp_dir,
                            "--site_list",
                            *comb,
                        ]
                    )
                    _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
                    assert os.path.exists(_c["storage_state"])
                    # update the config file
                    config_file = f"{temp_dir}/{os.path.basename(config_file)}"
                    with open(config_file, "w") as f:
                        json.dump(_c, f)

                # Load input images for the task, if any.
                if image_paths is not None:
                    if isinstance(image_paths, str):
                        image_paths = [image_paths]
                    for image_path in image_paths:
                        # Load image either from the web or from a local path.
                        if image_path.startswith("http"):
                            input_image = Image.open(requests.get(image_path, stream=True).raw)
                        else:
                            input_image = Image.open(image_path)

                        images.append(input_image)

            logger.info(f"[Config file]: {config_file}")
            logger.info(f"[Intent]: {intent}")

            agent.reset(config_file)
            trajectory: Trajectory = []

            # Determine whether this task will run in replay+interrupt mode
            task_spec = None
            if isinstance(interrupt_spec.get("tasks", None), dict) and task_id is not None:
                task_spec = interrupt_spec["tasks"].get(str(task_id), None)
            will_replay = bool(task_spec and replay_dir is not None)

            # Optional: reset the whole webarena docker instance to a clean state
            if args.reset_server_url and (
                args.reset_before_each_task
                or (args.reset_before_replay_only and will_replay)
            ):
                # Close any existing playwright resources before nuking docker services
                try:
                    env.close()
                except Exception:
                    pass
                logger.info(f"[Reset] Triggering full instance reset via {args.reset_server_url}")
                _reset_instance_via_server(
                    args.reset_server_url,
                    timeout_s=args.reset_timeout_s,
                    poll_interval_s=args.reset_poll_interval_s,
                    request_timeout_s=args.reset_request_timeout_s,
                    max_retries=args.reset_max_retries,
                )
                logger.info("[Reset] Instance is ready")

            obs, info = env.reset(options={"config_file": config_file})
            _freeze_page_url(info)
            state_info: StateInfo = {"observation": obs, "info": info}
            trajectory.append(state_info)

            meta_data = {"action_history": ["None"]}

            # Optional: replay recorded actions first, then inject user interrupt (intent update)
            # Spec format (minimal):
            # {
            #   "tasks": {
            #     "582": {
            #       "interrupt_at_action": 5,     // OR "interrupt_at_pct": "20%"
            #       "interrupt_at_pct": "20%",    // optional: percentage of saved actions
            #       "update_mode": "append" | "replace",
            #       "update_intent": "new task text",
            #       "extra_steps": 0
            #     }
            #   }
            # }
            replay_ended: bool = False
            if will_replay:
                # Determine interrupt point (fixed step or percentage of saved trajectory)
                update_mode = task_spec.get("update_mode", "append")
                update_intent = task_spec.get("update_intent", "")
                extra_steps = int(task_spec.get("extra_steps", 0))

                replay_file = replay_dir / f"{task_id}.json"
                if not replay_file.exists():
                    raise FileNotFoundError(
                        f"Replay file not found for task {task_id}: {replay_file}"
                    )
                replay_actions = _load_replay_actions(replay_file)

                # replay first K actions to reach the interruption point
                requested_k, interrupt_point = _resolve_interrupt_at_action(
                    task_spec, num_replay_actions=len(replay_actions)
                )
                k = requested_k
                logger.info(
                    f"[Replay] task_id={task_id}, replay_actions={len(replay_actions)}, replay_until={k}"
                )
                # Strict mode for interrupt replay:
                # If the episode ends (STOP/terminated) BEFORE reaching k, abort this task and report
                # task_id / baseline length / k, then skip to next task.
                replay_ended = False
                baseline_len = int(len(replay_actions))
                for i in range(k):
                    action = replay_actions[i]
                    trajectory.append(action)
                    action_str = get_action_description(
                        action,
                        state_info["info"]["observation_metadata"],
                        action_set_tag=args.action_set_tag,
                        prompt_constructor=agent.prompt_constructor
                        if isinstance(agent, PromptAgent)
                        else None,
                    )
                    meta_data["action_history"].append(action_str)

                    if action["action_type"] == ActionTypes.STOP:
                        replay_ended = True
                        msg = (
                            f"[ReplayEarlyEnd] task_id={task_id} baseline_len={baseline_len} "
                            f"k={k} ended_at_action_index={i} reason=stop_action"
                        )
                        logger.error(msg)
                        raise RuntimeError(msg)

                    obs, _, terminated, _, info = env.step(action)
                    _freeze_page_url(info)
                    state_info = {"observation": obs, "info": info}
                    trajectory.append(state_info)
                    if terminated:
                        # Preserve previous behavior (append placeholder STOP) for debugging exports,
                        # but still abort this task in strict mode.
                        trajectory.append(create_stop_action(""))
                        replay_ended = True
                        msg = (
                            f"[ReplayEarlyEnd] task_id={task_id} baseline_len={baseline_len} "
                            f"k={k} ended_at_action_index={i} reason=terminated"
                        )
                        logger.error(msg)
                        raise RuntimeError(msg)

                # inject user interrupt as an intent update
                if update_intent and (not replay_ended):
                    # Record a split point so exported trajectories can clearly distinguish
                    # replay/before-interrupt vs after-interrupt.
                    split_at = len(trajectory)
                    intent_before_interrupt = intent
                    if update_mode == "replace":
                        intent = update_intent
                    else:
                        # default: append
                        intent = (
                            f"{intent}\n\n[User interrupt @action={k}] {update_intent}"
                        )
                    interrupt_meta = {
                        "interrupt_at_action": k,
                        "interrupt_at_action_requested": requested_k,
                        "interrupt_point": interrupt_point,
                        "update_mode": update_mode,
                        "update_intent": update_intent,
                        "extra_steps": extra_steps,
                        "replay_file": str(replay_file),
                    }
                    # Optional: carry multi-interrupt bookkeeping metadata through result files.
                    # This is useful for multi-round interrupt pipelines that need to reconstruct
                    # the full update history from the *latest* run only.
                    try:
                        mi = None
                        if isinstance(task_spec, dict):
                            mi = task_spec.get("multi_interrupt", None)
                        if mi is None and isinstance(_c, dict):
                            mi = _c.get("multi_interrupt", None)
                        if mi is not None:
                            interrupt_meta["multi_interrupt"] = _json_safe_loose(mi)
                    except Exception:
                        pass
                    logger.info(f"[Interrupt injected] {interrupt_meta}")

                # allow extra budget after interrupt (optional)
                if extra_steps > 0:
                    task_max_steps = base_max_steps + extra_steps

            # If replay finished the episode (STOP/terminated), trajectory ends with an Action.
            # Prompt-based agents expect the last element to be a StateInfo; skip post-interrupt loop.
            if replay_ended:
                logger.info("[Replay] Episode ended during replay; skipping agent execution loop.")
            while (not replay_ended):
                early_stop_flag, stop_info = early_stop(
                    trajectory, task_max_steps, early_stop_thresholds
                )

                if early_stop_flag:
                    action = create_stop_action(f"Early stop: {stop_info}")
                else:
                    try:
                        action = agent.next_action(
                            trajectory,
                            intent,
                            images=images,
                            meta_data=meta_data,
                        )
                    except Exception as e:
                        # Don't crash the whole run if a single task fails to build a prompt / parse an action.
                        action = create_stop_action(
                            f"ERROR: {type(e).__name__}: {str(e)}"
                        )

                trajectory.append(action)

                action_str = get_action_description(
                    action,
                    state_info["info"]["observation_metadata"],
                    action_set_tag=args.action_set_tag,
                    prompt_constructor=agent.prompt_constructor
                    if isinstance(agent, PromptAgent)
                    else None,
                )
                render_helper.render(
                    action, state_info, meta_data, args.render_screenshot
                )
                meta_data["action_history"].append(action_str)

                if action["action_type"] == ActionTypes.STOP:
                    break

                obs, _, terminated, _, info = env.step(action)
                _freeze_page_url(info)
                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)

                if terminated:
                    # add a action place holder
                    trajectory.append(create_stop_action(""))
                    break

            # Save actions-only trajectory for replay
            if args.save_trajectory and task_id is not None:
                try:
                    _save_actions_only(
                        trajectory,
                        trajectory_dir / f"{task_id}.json",
                        task_id=task_id,
                        config_file=str(config_file),
                        intent=intent,
                        provider=args.provider,
                        model=args.model,
                        mode=args.mode,
                        action_set_tag=args.action_set_tag,
                        interrupt_meta=interrupt_meta,
                    )
                    # In interrupt mode, additionally export the full trajectory with a clear split.
                    if split_at is not None:
                        _save_full_trajectory_split_html(
                            trajectory,
                            full_trajectory_dir / f"{task_id}.html",
                            task_id=task_id,
                            config_file=str(config_file),
                            provider=args.provider,
                            model=args.model,
                            mode=args.mode,
                            action_set_tag=args.action_set_tag,
                            intent_before_interrupt=intent_before_interrupt,
                            intent_after_interrupt=intent,
                            split_at=split_at,
                            interrupt_meta=interrupt_meta,
                        )
                except Exception as e:
                    logger.info(f"[Save trajectory error] {repr(e)}")

            # NOTE: eval_caption_image_fn is used for running eval_vqa functions.
            evaluator = evaluator_router(
                config_file, captioning_fn=eval_caption_image_fn
            )
            score = evaluator(
                trajectory=trajectory,
                config_file=config_file,
                page=env.page
            )

            scores.append(score)

            if score == 1:
                logger.info(f"[Result] (PASS) {config_file}")
            else:
                logger.info(f"[Result] (FAIL) {config_file}")

            # Write a score file for score.py (expects RESULT_DIR/actions/<task_id>.json)
            try:
                out = {
                    "task_id": task_id,
                    "score": float(score),
                    "provider": args.provider,
                    "model": args.model,
                    "mode": args.mode,
                    "region": getattr(args, "region", None),
                    "intent": intent,
                    "config_file": str(config_file),
                    "interrupt": interrupt_meta,
                }
                with open(actions_dir / f"{task_id}.json", "w") as f:
                    json.dump(out, f, indent=2)
            except Exception as e:
                logger.info(f"[Write actions score error] {repr(e)}")

            if args.save_trace_enabled:
                env.save_trace(
                    Path(args.result_dir) / "traces" / f"{task_id}.zip"
                )
        except openai.OpenAIError as e:
            logger.info(f"[OpenAI Error] {repr(e)}")
        except Exception as e:
            logger.info(f"[Unhandled Error] {repr(e)}]")
            import traceback

            # write to error file
            with open(Path(args.result_dir) / "error.txt", "a") as f:
                f.write(f"[Config file]: {config_file}\n")
                f.write(f"[Unhandled Error] {repr(e)}\n")
                f.write(traceback.format_exc())  # write stack trace to file
            # Still write an actions file so score.py doesn't crash; mark score=0.0
            try:
                if task_id is not None:
                    out = {
                        "task_id": task_id,
                        "score": 0.0,
                        "provider": args.provider,
                        "model": args.model,
                        "mode": args.mode,
                        "region": getattr(args, "region", None),
                        "config_file": str(config_file),
                        "error": repr(e),
                    }
                    with open(actions_dir / f"{task_id}.json", "w") as f:
                        json.dump(out, f, indent=2)
            except Exception:
                pass

        render_helper.close()

    env.close()
    if len(scores):
        logger.info(f"Average score: {sum(scores) / len(scores)}")


def prepare(args: argparse.Namespace) -> None:
    # convert prompt python files to json
    from agent.prompts import to_json

    to_json.run()

    # prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        result_dir = (
            f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
        )
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)

    # log the log file
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")


def get_unfinished(config_files: list[str], result_dir: str) -> list[str]:
    result_files = glob.glob(f"{result_dir}/*.html")
    task_ids = [
        os.path.basename(f).split(".")[0].split("_")[1] for f in result_files
    ]
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        if task_id not in task_ids:
            unfinished_configs.append(config_file)
    return unfinished_configs


def dump_config(args: argparse.Namespace) -> None:
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = config()
    args.sleep_after_execution = 2.5
    prepare(args)

    test_config_base_dir = args.test_config_base_dir

    test_file_list = []
    if args.test_indices:
        parts = [p.strip() for p in str(args.test_indices).split(",") if p.strip()]
        indices: list[int] = [int(p) for p in parts]
        for i in indices:
            test_file_list.append(os.path.join(test_config_base_dir, f"{i}.json"))
    else:
        st_idx = args.test_start_idx
        ed_idx = args.test_end_idx
        for i in range(st_idx, ed_idx):
            test_file_list.append(os.path.join(test_config_base_dir, f"{i}.json"))
    test_file_list = get_unfinished(test_file_list, args.result_dir)
    print(f"Total {len(test_file_list)} tasks left")
    args.render = False
    args.render_screenshot = True
    args.save_trace_enabled = True

    args.current_viewport_only = True
    dump_config(args)

    test(args, test_file_list)
