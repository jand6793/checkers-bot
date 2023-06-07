import json
from pathlib import Path
from typing import Any


def save(path: Path, data: Any):
    with open(path, "w+") as f:
        json.dump(data, f, indent=4)


def load(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)
