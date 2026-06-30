"""Make the repo root importable so `import sarab` and `from tests._synthetic` work
without installing the package first."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
