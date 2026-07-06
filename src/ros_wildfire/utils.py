from __future__ import annotations

import os
from typing import Dict, Any


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_py_floats(d: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        if hasattr(v, "detach"):
            out[k] = float(v.detach().cpu().item())
        else:
            out[k] = float(v)
    return out
