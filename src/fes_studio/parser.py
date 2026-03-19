from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .models import FESDataset

ENERGY_HINTS = (
    "file.free",
    "free_energy",
    "free-energy",
    "free",
    "fes",
    "pmf",
    "dg",
)
ERROR_HINTS = ("err", "error", "sigma", "std", "uncert")
AUX_HINTS = ("der_", "grad", "gradient", "force", "bias")


def decode_text(raw: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def load_fes_file(path: str | Path) -> FESDataset:
    path = Path(path)
    dataset = load_fes_text(path.read_bytes(), source_name=path.name)
    dataset.metadata["path"] = str(path)
    return dataset


def load_fes_text(raw: str | bytes, source_name: str = "uploaded_fes.dat") -> FESDataset:
    text = decode_text(raw) if isinstance(raw, bytes) else raw
    comments: list[str] = []
    metadata: dict[str, str] = {}
    fields: list[str] | None = None
    rows: list[list[float]] = []

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#!"):
            comments.append(raw_line)
            tokens = line[2:].strip().split()
            if not tokens:
                continue
            key = tokens[0].upper()
            if key == "FIELDS":
                fields = tokens[1:]
            elif key == "SET" and len(tokens) >= 3:
                metadata[tokens[1]] = " ".join(tokens[2:])
            else:
                metadata[f"directive_{line_no}"] = line[2:].strip()
            continue
        if line.startswith("#") or line.startswith("@"):
            comments.append(raw_line)
            continue

        try:
            values = [float(token) for token in line.split()]
        except ValueError as exc:
            raise ValueError(f"Line {line_no} cannot be parsed as numeric data: {raw_line}") from exc
        rows.append(values)

    if not rows:
        raise ValueError("No numeric data rows were found in the provided file.")

    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("The file has inconsistent column counts across numeric rows.")

    column_names = list(fields) if fields and len(fields) == width else [f"col_{idx + 1}" for idx in range(width)]
    frame = pd.DataFrame(rows, columns=column_names)

    energy_column = _infer_energy_column(column_names)
    error_column = _infer_error_column(column_names, energy_column)
    cv_columns = _infer_cv_columns(column_names, energy_column, error_column)
    regular_grid, grid_shape = _detect_regular_grid(frame, cv_columns)

    return FESDataset(
        source_name=source_name,
        frame=frame,
        comments=comments,
        metadata=metadata,
        column_names=column_names,
        cv_columns=cv_columns,
        energy_column=energy_column,
        error_column=error_column,
        dimension=len(cv_columns),
        regular_grid=regular_grid,
        grid_shape=grid_shape,
    )


def reconfigure_dataset(
    dataset: FESDataset,
    cv_columns: list[str],
    energy_column: str,
    error_column: str | None = None,
) -> FESDataset:
    if not cv_columns:
        raise ValueError("At least one collective variable column is required.")
    if energy_column in cv_columns:
        raise ValueError("The energy column cannot also be a collective variable column.")
    regular_grid, grid_shape = _detect_regular_grid(dataset.frame, cv_columns)
    return replace(
        dataset,
        cv_columns=cv_columns,
        energy_column=energy_column,
        error_column=error_column or None,
        dimension=len(cv_columns),
        regular_grid=regular_grid,
        grid_shape=grid_shape,
    )


def _infer_energy_column(column_names: list[str]) -> str:
    lowered = {name: name.lower() for name in column_names}
    for name, token in lowered.items():
        if any(hint in token for hint in ENERGY_HINTS):
            return name
    return column_names[-1]


def _infer_error_column(column_names: list[str], energy_column: str) -> str | None:
    lowered = {name: name.lower() for name in column_names}
    for name, token in lowered.items():
        if name == energy_column:
            continue
        if any(hint in token for hint in ERROR_HINTS):
            return name
    return None


def _infer_cv_columns(column_names: list[str], energy_column: str, error_column: str | None) -> list[str]:
    if energy_column in column_names:
        energy_idx = column_names.index(energy_column)
        leading = [name for name in column_names[:energy_idx] if not _is_auxiliary(name)]
        if leading:
            return leading

    trailing = [
        name
        for name in column_names
        if name not in {energy_column, error_column}
        and not _is_auxiliary(name)
    ]
    return trailing or [name for name in column_names if name != energy_column]


def _is_auxiliary(column_name: str) -> bool:
    name = column_name.lower()
    return any(hint in name for hint in AUX_HINTS)


def _detect_regular_grid(frame: pd.DataFrame, cv_columns: list[str]) -> tuple[bool, tuple[int, ...] | None]:
    if not cv_columns:
        return False, None
    unique_counts = tuple(int(frame[column].nunique()) for column in cv_columns)
    duplicated = frame.duplicated(subset=cv_columns).any()
    regular = int(np.prod(unique_counts)) == len(frame) and not duplicated
    return regular, unique_counts if regular else None
