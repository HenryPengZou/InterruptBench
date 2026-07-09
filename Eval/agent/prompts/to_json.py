import glob
import importlib
import json
import os
import tempfile
from pathlib import Path

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore


# use the current directory as the root
def run() -> None:
    """Convert all python files in agent/prompts to json files in agent/prompts/jsons

    Python files are easiser to edit
    """
    json_dir = Path("agent/prompts/jsons")
    json_dir.mkdir(parents=True, exist_ok=True)

    # Multi-process safety:
    # - Lock: prevent concurrent writers from truncating the same json file.
    # - Atomic replace: readers never observe partial files.
    lock_path = json_dir / ".to_json.lock"
    with open(lock_path, "a+") as lock_f:
        if fcntl is not None:
            try:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            except Exception:
                # Best-effort: continue without lock if flock fails
                pass

        for p_file in glob.glob("agent/prompts/raw/*.py"):
            base_name = os.path.basename(p_file).replace(".py", "")
            module = importlib.import_module(f"agent.prompts.raw.{base_name}")
            prompt = module.prompt

            final_path = json_dir / f"{base_name}.json"
            # Write to a temp file in the same directory then replace atomically.
            fd, tmp_path = tempfile.mkstemp(
                prefix=f".{base_name}.", suffix=".tmp", dir=str(json_dir)
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(prompt, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, final_path)
            finally:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass

    print("Done convert python files to json")


if __name__ == "__main__":
    run()
