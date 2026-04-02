import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = Path(os.environ.get("VBCSR_BUILD_DIR", REPO_ROOT / "build"))

def pythonpath_entries() -> list[str]:
    entries = [str(REPO_ROOT)]
    if BUILD_DIR.is_dir():
        entries.append(str(BUILD_DIR))
    return entries


def ensure_repo_on_path() -> None:
    for entry in reversed(pythonpath_entries()):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def build_subprocess_env(env: dict[str, str] | None = None) -> dict[str, str]:
    merged = os.environ.copy() if env is None else env.copy()
    entries = pythonpath_entries()
    existing = merged.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    merged["PYTHONPATH"] = os.pathsep.join(entries)
    merged.setdefault("VBCSR_BUILD_DIR", str(BUILD_DIR))
    return merged


ensure_repo_on_path()
