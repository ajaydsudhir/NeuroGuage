from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_CONTRACT_PATH = Path("configs/data_contract.json")


def load_data_contract(contract_path: str | Path = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    path = Path(contract_path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
