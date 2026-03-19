from __future__ import annotations

from pathlib import Path

import numpy as np


def ensure_demo_files(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root)
    examples = root / "examples"
    examples.mkdir(parents=True, exist_ok=True)

    one_d = examples / "demo_1d_fes.dat"
    two_d = examples / "demo_2d_fes.dat"
    if not one_d.exists():
        _write_demo_1d(one_d)
    if not two_d.exists():
        _write_demo_2d(two_d)
    return {"1d": one_d, "2d": two_d}


def _write_demo_1d(path: Path) -> None:
    x = np.linspace(-3.5, 3.5, 280)
    energy = 0.24 * (x**4) - 1.75 * (x**2) + 0.32 * x + 0.35 * np.sin(6.3 * x) + 4.4
    error = 0.08 + 0.02 * np.cos(1.8 * x + 0.4)
    energy -= energy.min()

    lines = ["#! FIELDS s file.free file.error"]
    lines.extend(f"{value: .8f} {free: .8f} {err: .8f}" for value, free, err in zip(x, energy, error))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_demo_2d(path: Path) -> None:
    x = np.linspace(-3.2, 3.2, 150)
    y = np.linspace(-3.0, 3.0, 150)
    xx, yy = np.meshgrid(x, y)
    surface = (
        7.6
        + 0.055 * (xx**4 + yy**4)
        + 0.18 * xx * yy
        - 4.9 * np.exp(-(((xx + 1.55) ** 2) / 0.50 + ((yy - 1.05) ** 2) / 0.42))
        - 5.4 * np.exp(-(((xx - 1.10) ** 2) / 0.48 + ((yy + 1.30) ** 2) / 0.38))
        - 3.8 * np.exp(-(((xx - 0.20) ** 2) / 0.36 + ((yy - 0.10) ** 2) / 0.54))
        + 0.25 * np.sin(1.8 * xx - 0.7 * yy)
    )
    surface -= surface.min()

    lines = ["#! FIELDS cv1 cv2 file.free"]
    for row_x, row_y, row_z in zip(xx.ravel(), yy.ravel(), surface.ravel()):
        lines.append(f"{row_x: .8f} {row_y: .8f} {row_z: .8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    paths = ensure_demo_files(project_root)
    for name, path in paths.items():
        print(f"{name}: {path}")
