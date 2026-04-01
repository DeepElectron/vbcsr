import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = Path(os.environ.get("VBCSR_BUILD_DIR", REPO_ROOT / "build"))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if BUILD_DIR.is_dir() and str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))
