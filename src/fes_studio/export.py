from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .i18n import figure_display_name, summary_display_name, table_display_name, tr
from .models import FESDataset, PlotConfig
from .plotting import build_figures, save_matplotlib_figure


def make_output_dir(base_dir: str | Path, source_name: str) -> Path:
    base_path = Path(base_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(source_name).stem
    output_dir = base_path / f"{stem}_analysis_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def export_bundle(
    dataset: FESDataset,
    analysis: dict[str, object],
    plot_config: PlotConfig,
    output_dir: str | Path,
) -> dict[str, list[Path] | Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tables_dir = output_path / "tables"
    figures_dir = output_path / "figures"
    tables_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    saved_files: list[Path] = []
    summary_path = output_path / "summary.json"
    summary_path.write_text(json.dumps(analysis["summary"], indent=2, ensure_ascii=False), encoding="utf-8")
    saved_files.append(summary_path)

    for name, table in analysis["tables"].items():
        if getattr(table, "empty", False) and name != "barrier_matrix":
            continue
        table_path = tables_dir / f"{name}.csv"
        table.to_csv(table_path, index=(name == "barrier_matrix"))
        saved_files.append(table_path)

    workbook_path = output_path / "analysis_tables.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        pd.DataFrame(analysis["summary"].items(), columns=["item", "value"]).to_excel(writer, sheet_name="summary", index=False)
        for name, table in analysis["tables"].items():
            sheet = name[:31]
            table.to_excel(writer, sheet_name=sheet, index=(name == "barrier_matrix"))
    saved_files.append(workbook_path)

    figures = build_figures(analysis, plot_config)
    figure_paths: list[Path] = []
    for name, figure in figures.items():
        if hasattr(figure, "write_html"):
            html_path = figures_dir / f"{name}.html"
            figure.write_html(str(html_path), include_plotlyjs="cdn")
            figure_paths.append(html_path)
            continue
        figure_paths.extend(save_matplotlib_figure(figure, figures_dir / name, plot_config))

    report_path = output_path / "report.md"
    report_path.write_text(build_report(dataset, analysis, figure_paths, plot_config.report_language), encoding="utf-8")
    saved_files.append(report_path)
    saved_files.extend(figure_paths)

    return {"output_dir": output_path, "files": saved_files}


def build_report(dataset: FESDataset, analysis: dict[str, object], figure_paths: list[Path], language: str) -> str:
    lines = [
        f"# {tr(language, 'report_title')}",
        "",
        f"- {tr(language, 'report_source')}: `{dataset.source_name}`",
        f"- {tr(language, 'report_dimension')}: `{analysis['summary']['dimension']}`",
        f"- {tr(language, 'report_cvs')}: `{', '.join(analysis['summary']['cv_columns'])}`",
        f"- {tr(language, 'report_energy_unit')}: `{analysis['summary']['energy_unit']}`",
        "",
        f"## {tr(language, 'report_summary')}",
        "",
    ]
    for key, value in analysis["summary"].items():
        lines.append(f"- {summary_display_name(key, language)}: {value}")

    lines.extend(["", f"## {tr(language, 'report_tables')}", ""])
    for name, table in analysis["tables"].items():
        lines.append(f"### {table_display_name(name, language)}")
        lines.append("")
        if getattr(table, "empty", False):
            lines.append(f"_{tr(language, 'report_no_rows')}_")
        else:
            preview = table.head(10).to_string(index=False)
            lines.append("```text")
            lines.append(preview)
            lines.append("```")
        lines.append("")

    lines.extend([f"## {tr(language, 'report_figures')}", ""])
    for path in figure_paths:
        stem = path.stem
        lines.append(f"- `{figure_display_name(stem, language)}`: `{path.name}`")
    lines.append("")
    return "\n".join(lines)
