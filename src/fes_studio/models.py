from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class FESDataset:
    source_name: str
    frame: pd.DataFrame
    comments: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    column_names: list[str] = field(default_factory=list)
    cv_columns: list[str] = field(default_factory=list)
    energy_column: str = ""
    error_column: str | None = None
    dimension: int = 0
    regular_grid: bool = False
    grid_shape: tuple[int, ...] | None = None

    @property
    def path(self) -> Path | None:
        raw_path = self.metadata.get("path")
        return Path(raw_path) if raw_path else None

    @property
    def available_columns(self) -> list[str]:
        return list(self.frame.columns)


@dataclass(slots=True)
class AnalysisConfig:
    temperature: float = 300.0
    energy_unit: str = "kJ/mol"
    top_minima: int = 6
    interpolation_points: int = 240
    smoothing_sigma: float = 1.0
    minima_neighborhood: int = 3
    prominence_fraction: float = 0.03
    mfep_images: int = 41
    mfep_iterations: int = 180
    mfep_step_size: float = 0.04
    mfep_two_stage: bool = True
    mfep_coarse_images: int | None = None
    mfep_coarse_iterations: int | None = None
    mfep_coarse_step_size: float | None = None
    mfep_spring_constant: float = 0.18
    mfep_climbing_image: bool = True
    primary_mfep_pair: str | None = None


@dataclass(slots=True)
class PlotConfig:
    dpi: int = 450
    figure_width: float = 7.2
    contour_levels: int = 24
    palette: str = "lagoon"
    language: str = "zh"
    report_language: str = "zh"
    mfep_publication_mode: bool = False
    annotate_barrier_table: bool = False
    export_formats: tuple[str, ...] = ("png", "pdf", "svg")
