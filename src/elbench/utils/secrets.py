from __future__ import annotations

import os
from pathlib import Path


def get_api_key(env_name: str | None) -> str:
    if not env_name:
        return ""
    value = os.getenv(env_name, "").strip()
    if value:
        return value
    return _read_key_file(env_name)


def _read_key_file(env_name: str) -> str:
    for directory in [Path.cwd(), *Path.cwd().parents]:
        key_path = directory / "key.txt"
        if not key_path.exists():
            continue
        for raw_line in key_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                if key.strip() == env_name:
                    return value.strip().strip('"').strip("'")
                continue
            if env_name == "INNOSPARK_API_KEY":
                return line
    return ""
