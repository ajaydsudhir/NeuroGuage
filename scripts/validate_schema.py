from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuroguage.cli import main_validate_schema


if __name__ == "__main__":
    raise SystemExit(main_validate_schema())
