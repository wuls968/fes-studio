from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
APP_PATH = PROJECT_ROOT / "app.py"
EXPORT_ROOT = PROJECT_ROOT / "exports"
GENERATED_FES_ROOT = PROJECT_ROOT / "generated_fes"
RUNTIME_DIR = PROJECT_ROOT / ".run"

TOOLS_ENV_VARS = ("FES_STUDIO_TOOLS_ROOT", "FES_STUDIO_OPES_METAD_ROOT")


def _normalize_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser()


def postprocess_root_candidates() -> list[Path]:
    home = Path.home()
    candidates: list[Path] = []
    for key in TOOLS_ENV_VARS:
        path = _normalize_path(os.environ.get(key))
        if path is not None:
            candidates.append(path)
    candidates.extend(
        [
            home / "Downloads" / "others" / "opes-metad",
            home / "Downloads" / "opes-metad",
            home / "opes-metad",
            PROJECT_ROOT / "external" / "opes-metad",
            PROJECT_ROOT / "opes-metad",
        ]
    )

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def default_postprocess_root() -> Path:
    candidates = postprocess_root_candidates()
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else PROJECT_ROOT / "opes-metad"


def resolve_tools_root(tools_root: str | Path | None = None) -> Path:
    return _normalize_path(tools_root) or default_postprocess_root()


def default_run_directory() -> str:
    root = default_postprocess_root()
    preferred = [root / "opes", root / "metad", root]
    for candidate in preferred:
        if not candidate.exists() or not candidate.is_dir():
            continue
        if candidate.name in {"opes", "metad"}:
            return str(candidate)
        if any((candidate / name).exists() for name in ("COLVAR", "HILLS", "BIAS", "STATE", "KERNELS")):
            return str(candidate)
    return ""
