from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.colors import Colormap, LinearSegmentedColormap

from .i18n import tr
from .models import PlotConfig

PALETTE_OPTIONS = [
    "lagoon",
    "atlas",
    "ember",
    "viridis",
    "cividis",
    "magma",
    "inferno",
    "plasma",
    "turbo",
    "cubehelix",
    "YlGnBu",
    "PuBuGn",
]

PALETTE_LABELS = {
    "lagoon": "Lagoon",
    "atlas": "Atlas",
    "ember": "Ember",
    "viridis": "Viridis",
    "cividis": "Cividis",
    "magma": "Magma",
    "inferno": "Inferno",
    "plasma": "Plasma",
    "turbo": "Turbo",
    "cubehelix": "Cubehelix",
    "YlGnBu": "YlGnBu",
    "PuBuGn": "PuBuGn",
}


def apply_publication_style(plot_config: PlotConfig) -> Colormap:
    mpl.rcParams.update(
        {
            "figure.dpi": plot_config.dpi,
            "savefig.dpi": plot_config.dpi,
            "font.family": "sans-serif",
            "font.sans-serif": ["Avenir Next", "Helvetica Neue", "PingFang SC", "Arial", "DejaVu Sans"],
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "axes.titleweight": 600,
            "axes.labelweight": 500,
            "axes.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.2,
            "grid.linewidth": 0.7,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "legend.frameon": False,
            "mathtext.fontset": "stix",
            "figure.facecolor": "#fcfbf7",
            "axes.facecolor": "#fcfbf7",
            "savefig.facecolor": "#fcfbf7",
        }
    )
    sns.set_theme(style="ticks", context="paper")
    return build_colormap(plot_config.palette)


def build_colormap(palette: str) -> Colormap:
    palettes = {
        "lagoon": ["#f8f5ec", "#d9ebe2", "#91c2b7", "#3c7d7f", "#17304f"],
        "ember": ["#fff6e8", "#f8d4a0", "#de8454", "#a13031", "#390d17"],
        "atlas": ["#f4efe7", "#b9d6df", "#5f96b2", "#1b4f72", "#0d1d35"],
    }
    if palette in palettes:
        colors = palettes[palette]
        return LinearSegmentedColormap.from_list(f"fes_{palette}", colors, N=256)
    if palette in mpl.colormaps:
        return mpl.colormaps[palette]
    return LinearSegmentedColormap.from_list("fes_lagoon", palettes["lagoon"], N=256)


def build_figures(analysis: dict[str, object], plot_config: PlotConfig) -> dict[str, object]:
    cmap = apply_publication_style(plot_config)
    mode = analysis["mode"]
    figures: dict[str, object] = {}
    if mode == "1d":
        figures["publication_profile"] = plot_1d_profile(analysis, plot_config, cmap)
    elif mode == "2d":
        mfep_figure = plot_mfep_showcase(analysis, plot_config, cmap, publication=False)
        if mfep_figure is not None:
            figures["mfep_showcase"] = mfep_figure
        if plot_config.mfep_publication_mode:
            mfep_publication = plot_mfep_showcase(analysis, plot_config, cmap, publication=True)
            if mfep_publication is not None:
                figures["mfep_publication"] = mfep_publication
        figures["publication_landscape"] = plot_2d_landscape(analysis, plot_config, cmap)
        figures["basin_map"] = plot_2d_basin_map(analysis, plot_config, cmap)
        barrier_fig = plot_barrier_heatmap(analysis, plot_config, cmap)
        if barrier_fig is not None:
            figures["barrier_heatmap"] = barrier_fig
        figures["cross_sections"] = plot_2d_sections(analysis, plot_config)
        figures["interactive_surface"] = plotly_surface(analysis, cmap, plot_config.language)
    return figures


def save_matplotlib_figure(fig: plt.Figure, output_base: Path, plot_config: PlotConfig) -> list[Path]:
    saved_paths: list[Path] = []
    for suffix in plot_config.export_formats:
        output_path = output_base.with_suffix(f".{suffix}")
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.09, dpi=plot_config.dpi)
        saved_paths.append(output_path)
    return saved_paths


def plot_1d_profile(
    analysis: dict[str, object],
    plot_config: PlotConfig,
    cmap: Colormap,
) -> plt.Figure:
    artifacts = analysis["artifacts"]
    minima = analysis["tables"]["minima"]
    barriers = analysis["tables"]["barriers"]
    x = artifacts["x"]
    energy = artifacts["energy"]
    smooth = artifacts["smooth"]
    density = artifacts["density"]
    error = artifacts["error"]
    x_name = artifacts["x_name"]
    energy_unit = analysis["summary"]["energy_unit"]
    language = plot_config.language

    fig = plt.figure(figsize=(plot_config.figure_width, plot_config.figure_width * 0.82))
    grid = fig.add_gridspec(2, 1, height_ratios=[3.1, 1.0], hspace=0.08)
    ax_main = fig.add_subplot(grid[0, 0])
    ax_bottom = fig.add_subplot(grid[1, 0], sharex=ax_main)

    accent = "#b64926"
    line_color = "#18344a"
    fill_color = cmap(0.48)

    if error is not None:
        ax_main.fill_between(x, energy - error, energy + error, color=fill_color, alpha=0.16, linewidth=0)
    ax_main.fill_between(x, 0, energy, color=fill_color, alpha=0.22)
    ax_main.plot(x, energy, color=line_color, linewidth=2.2, label=tr(language, "raw_profile"))
    ax_main.plot(x, smooth, color=accent, linewidth=1.6, linestyle="--", label=tr(language, "smoothed_profile"))

    for _, row in minima.iterrows():
        ax_main.scatter(row[x_name], row["free_energy"], s=56, color=accent, edgecolor="white", linewidth=0.9, zorder=4)
        ax_main.annotate(
            f"M{int(row['rank'])}",
            xy=(row[x_name], row["free_energy"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=line_color,
        )

    if not barriers.empty:
        for _, row in barriers.iterrows():
            ax_main.axvline(row["transition_coordinate"], color="#7f878d", linestyle=":", linewidth=1.0, alpha=0.8)
            ax_main.scatter(row["transition_coordinate"], row["saddle_energy"], s=28, color="#7f878d", zorder=5)

    scaled_density = density / np.nanmax(density)
    ax_bottom.fill_between(x, 0, scaled_density, color=cmap(0.72), alpha=0.30)
    ax_bottom.plot(x, scaled_density, color="#2d5c6b", linewidth=1.9)

    ax_main.set_ylabel(tr(language, "free_energy_label", energy_unit=energy_unit))
    ax_bottom.set_xlabel(x_name)
    ax_bottom.set_ylabel(tr(language, "probability_scaled"))
    ax_main.set_title(tr(language, "publication_profile"))
    ax_main.grid(True, axis="y")
    ax_bottom.grid(True, axis="y")
    ax_main.legend(loc="upper right", ncol=2)

    return fig


def plot_2d_landscape(
    analysis: dict[str, object],
    plot_config: PlotConfig,
    cmap: Colormap,
) -> plt.Figure:
    artifacts = analysis["artifacts"]
    minima = analysis["tables"]["minima"]
    x_marginal = analysis["tables"]["x_marginal"]
    y_marginal = analysis["tables"]["y_marginal"]
    x = artifacts["x"]
    y = artifacts["y"]
    grid = artifacts["grid"]
    x_name = artifacts["x_name"]
    y_name = artifacts["y_name"]
    energy_unit = analysis["summary"]["energy_unit"]
    language = plot_config.language
    primary_path_key = artifacts.get("primary_path_key")
    mfep_paths = artifacts.get("mfep_paths", {})

    fig = plt.figure(figsize=(plot_config.figure_width * 1.22, plot_config.figure_width * 0.98))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[5.2, 1.4],
        height_ratios=[1.4, 5.2],
        hspace=0.08,
        wspace=0.08,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    levels = np.linspace(np.nanmin(grid), np.nanmax(grid), plot_config.contour_levels)
    contour = ax_main.contourf(x, y, grid, levels=levels, cmap=cmap, antialiased=True)
    ax_main.contour(x, y, grid, levels=levels[::2], colors="white", linewidths=0.5, alpha=0.65)

    for _, row in minima.iterrows():
        ax_main.scatter(row[x_name], row[y_name], s=58, color="#b64926", edgecolor="white", linewidth=1.0, zorder=5)
        ax_main.annotate(
            f"M{int(row['rank'])}",
            xy=(row[x_name], row[y_name]),
            xytext=(6, 6),
            textcoords="offset points",
            color="#17304f",
            fontsize=9,
            weight=600,
        )

    for (left, right), points in artifacts["paths"].items():
        if len(points) < 2:
            continue
        points_array = np.asarray(points)
        is_primary = (left, right) == primary_path_key
        line_width = 3.2 if is_primary else 1.8
        line_alpha = 0.98 if is_primary else 0.72
        line_color = "#0f2940" if is_primary else "#2f6070"
        ax_main.plot(points_array[:, 0], points_array[:, 1], color="white", linewidth=line_width + 2.2, alpha=0.88, zorder=4)
        ax_main.plot(points_array[:, 0], points_array[:, 1], color=line_color, linewidth=line_width, alpha=line_alpha, zorder=5)
        bead_step = max(3, len(points_array) // 8)
        ax_main.scatter(
            points_array[:, 0],
            points_array[:, 1],
            s=10 if is_primary else 7,
            color="#fffaf1",
            edgecolor=line_color,
            linewidth=0.45,
            alpha=0.95 if is_primary else 0.7,
            zorder=6,
        )
        ax_main.scatter(
            points_array[::bead_step, 0],
            points_array[::bead_step, 1],
            s=24 if is_primary else 14,
            color="#f7e3c4",
            edgecolor=line_color,
            linewidth=0.75,
            alpha=0.95,
            zorder=7,
        )
        mid = points_array[len(points_array) // 2]
        ax_main.text(mid[0], mid[1], f"{left}-{right}", fontsize=8, color="#102f44", ha="center", va="center")
        path_info = mfep_paths.get((left, right))
        if is_primary and path_info is not None:
            saddle_x, saddle_y = path_info["saddle_point"]
            ax_main.scatter(saddle_x, saddle_y, marker="*", s=150, color="#f7d154", edgecolor="#6f4b00", linewidth=0.8, zorder=7)

    ax_top.plot(x_marginal[x_name], x_marginal["boltzmann_profile"], color="#17304f", linewidth=2.0, label=tr(language, "boltzmann"))
    ax_top.plot(x_marginal[x_name], x_marginal["min_projection"], color="#b64926", linewidth=1.5, linestyle="--", label=tr(language, "min_projection"))
    ax_right.plot(y_marginal["boltzmann_profile"], y_marginal[y_name], color="#17304f", linewidth=2.0)
    ax_right.plot(y_marginal["min_projection"], y_marginal[y_name], color="#b64926", linewidth=1.5, linestyle="--")

    ax_main.set_xlabel(x_name)
    ax_main.set_ylabel(y_name)
    ax_top.set_ylabel(tr(language, "delta_g_label", energy_unit=energy_unit))
    ax_right.set_xlabel(tr(language, "delta_g_label", energy_unit=energy_unit))
    ax_top.legend(loc="upper right", fontsize=8, ncol=2)
    ax_top.set_title(tr(language, "joint_landscape"))
    ax_main.grid(False)
    ax_top.grid(True, axis="y")
    ax_right.grid(True, axis="x")
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    cbar = fig.colorbar(contour, ax=ax_main, pad=0.02, fraction=0.06)
    cbar.set_label(tr(language, "delta_g_label", energy_unit=energy_unit))

    return fig


def plot_mfep_showcase(
    analysis: dict[str, object],
    plot_config: PlotConfig,
    cmap: Colormap,
    publication: bool = False,
) -> plt.Figure | None:
    artifacts = analysis["artifacts"]
    primary_path_key = artifacts.get("primary_path_key")
    mfep_paths = artifacts.get("mfep_paths", {})
    if primary_path_key is None or primary_path_key not in mfep_paths:
        return None

    path = mfep_paths[primary_path_key]
    minima = analysis["tables"]["minima"]
    x = artifacts["x"]
    y = artifacts["y"]
    grid = artifacts["grid"]
    x_name = artifacts["x_name"]
    y_name = artifacts["y_name"]
    energy_unit = analysis["summary"]["energy_unit"]
    language = plot_config.language

    points = np.asarray(path["points"])
    energies = np.asarray(path["energies"])
    progress = np.asarray(path["progress"])
    saddle_idx = int(path["saddle_index"])
    bead_step = max(3, len(points) // 8)
    fig_scale = 1.22 if publication else 1.10
    fig_height = 1.28 if publication else 1.16
    path_linewidth = 3.6 if publication else 3.0
    outline_width = 6.6 if publication else 6.0
    contour_count = plot_config.contour_levels + (14 if publication else 8)
    fig = plt.figure(figsize=(plot_config.figure_width * fig_scale, plot_config.figure_width * fig_height))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 0.038],
        height_ratios=[3.55, 1.42],
        hspace=0.26 if publication else 0.22,
        wspace=0.03,
    )
    ax_map = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax_profile = fig.add_subplot(gs[1, 0])
    ax_empty = fig.add_subplot(gs[1, 1])
    ax_empty.axis("off")

    levels = np.linspace(np.nanmin(grid), np.nanmax(grid), contour_count)
    filled = ax_map.contourf(x, y, grid, levels=levels, cmap=cmap, antialiased=True)
    ax_map.contour(x, y, grid, levels=levels[::2], colors="white", linewidths=0.55, alpha=0.72)

    ax_map.plot(points[:, 0], points[:, 1], color="white", linewidth=outline_width, alpha=0.88, zorder=4)
    ax_map.plot(points[:, 0], points[:, 1], color="#112a42", linewidth=path_linewidth, zorder=5)
    ax_map.scatter(points[:, 0], points[:, 1], s=12 if publication else 11, color="#fffaf1", edgecolor="#112a42", linewidth=0.45, zorder=6)
    ax_map.scatter(points[::bead_step, 0], points[::bead_step, 1], s=28 if publication else 26, color="#f7e3c4", edgecolor="#112a42", linewidth=0.85, zorder=7)
    ax_map.scatter(points[0, 0], points[0, 1], s=84, color="#b64926", edgecolor="white", linewidth=1.0, zorder=7)
    ax_map.scatter(points[-1, 0], points[-1, 1], s=84, color="#17304f", edgecolor="white", linewidth=1.0, zorder=7)
    ax_map.scatter(points[saddle_idx, 0], points[saddle_idx, 1], marker="*", s=180, color="#f6d04d", edgecolor="#7b5200", linewidth=0.9, zorder=8)
    ax_map.annotate(f"S{primary_path_key[0]}", xy=points[0], xytext=(8, 8), textcoords="offset points", color="#7a2f14", fontsize=10, weight=700)
    ax_map.annotate(f"S{primary_path_key[1]}", xy=points[-1], xytext=(8, 8), textcoords="offset points", color="#102f44", fontsize=10, weight=700)
    ax_map.annotate("TS", xy=points[saddle_idx], xytext=(8, -10), textcoords="offset points", color="#6f4b00", fontsize=10, weight=700)

    for _, row in minima.iterrows():
        ax_map.scatter(row[x_name], row[y_name], s=28, color="#ffffff", edgecolor="#17304f", linewidth=0.7, zorder=6, alpha=0.85)

    ax_map.set_xlabel(x_name, labelpad=4)
    ax_map.set_ylabel(y_name, labelpad=4)
    ax_map.set_title(
        tr(language, "publication_mfep" if publication else "mfep_showcase", path_id=path["path_id"]),
        pad=8,
    )
    ax_map.grid(False)
    schedule_text = path.get("schedule_compact")
    if schedule_text:
        stats_text = (
            f"{tr(language, 'beads')}: {schedule_text} | {tr(language, 'spacing_cv_label')}: {path['spacing_cv']:.3f} | "
            f"{tr(language, 'saddle_short')}: {path['saddle_energy']:.2f}"
        )
    else:
        stats_text = (
            f"{tr(language, 'images')}: {path['images']} | {tr(language, 'spacing_cv_label')}: {path['spacing_cv']:.3f} | "
            f"{tr(language, 'saddle_short')}: {path['saddle_energy']:.2f}"
        )
    ax_map.text(
        0.02,
        0.03,
        stats_text,
        transform=ax_map.transAxes,
        fontsize=9,
        color="#102f44",
        bbox={"boxstyle": "round,pad=0.32", "facecolor": (1, 1, 1, 0.72), "edgecolor": (0.06, 0.16, 0.26, 0.18)},
    )
    if publication:
        ax_map.text(0.01, 0.99, "A", transform=ax_map.transAxes, ha="left", va="top", fontsize=16, weight=800, color="#102f44")
        ax_profile.text(0.01, 0.97, "B", transform=ax_profile.transAxes, ha="left", va="top", fontsize=16, weight=800, color="#102f44")
    if plot_config.annotate_barrier_table:
        _draw_barrier_inset_table(ax_map, analysis, primary_path_key, language)
    cbar = fig.colorbar(filled, cax=cax)
    cbar.set_label(tr(language, "delta_g_label", energy_unit=energy_unit))
    cbar.ax.tick_params(length=4, width=0.9, pad=3)

    ax_profile.fill_between(progress, np.min(energies), energies, color="#9ec5bf", alpha=0.28)
    ax_profile.plot(progress, energies, color="#102f44", linewidth=2.4)
    ax_profile.scatter(progress, energies, s=10, color="#fffaf1", edgecolor="#102f44", linewidth=0.35, zorder=4)
    ax_profile.scatter(progress[::bead_step], energies[::bead_step], s=22, color="#f7e3c4", edgecolor="#102f44", linewidth=0.7, zorder=5)
    ax_profile.scatter(progress[0], energies[0], s=52, color="#b64926", edgecolor="white", linewidth=0.9, zorder=5)
    ax_profile.scatter(progress[-1], energies[-1], s=52, color="#17304f", edgecolor="white", linewidth=0.9, zorder=5)
    ax_profile.scatter(progress[saddle_idx], energies[saddle_idx], marker="*", s=155, color="#f6d04d", edgecolor="#7b5200", linewidth=0.9, zorder=6)
    ax_profile.axvline(progress[saddle_idx], color="#7b5200", linestyle="--", linewidth=1.1, alpha=0.8)
    ax_profile.set_xlim(float(np.min(progress)) - 0.01, float(np.max(progress)) + 0.01)
    ax_profile.set_xlabel(tr(language, "path_coordinate"), labelpad=5)
    ax_profile.set_ylabel(tr(language, "delta_g_label", energy_unit=energy_unit), labelpad=4)
    ax_profile.set_title(tr(language, "along_path_profile"), pad=6)
    ax_profile.grid(True, axis="y")
    fig.align_ylabels([ax_map, ax_profile])

    return fig


def _draw_barrier_inset_table(
    ax: plt.Axes,
    analysis: dict[str, object],
    primary_path_key: tuple[int, int],
    language: str,
) -> None:
    mfep_table = analysis["tables"].get("mfep_paths")
    if mfep_table is None or getattr(mfep_table, "empty", True):
        return

    display = mfep_table.copy()
    display["priority"] = np.where(
        (display["state_i"] == primary_path_key[0]) & (display["state_j"] == primary_path_key[1]),
        0,
        1,
    )
    display = display.sort_values(["priority", "saddle_energy", "state_i", "state_j"]).head(6)
    rows = [
        [
            row["path_id"],
            f"{row['barrier_i_to_j']:.2f}",
            f"{row['barrier_j_to_i']:.2f}",
            f"{row['saddle_energy']:.2f}",
        ]
        for _, row in display.iterrows()
    ]

    inset = ax.inset_axes([0.60, 0.58, 0.36, 0.30])
    inset.axis("off")
    inset.set_facecolor((1, 1, 1, 0.0))
    inset.text(0.0, 1.06, tr(language, "barrier_table"), fontsize=9, weight=700, color="#102f44", transform=inset.transAxes)
    table = inset.table(
        cellText=rows,
        colLabels=[
            tr(language, "barrier_path"),
            tr(language, "barrier_forward"),
            tr(language, "barrier_reverse"),
            tr(language, "barrier_saddle"),
        ],
        loc="upper left",
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.3)
    table.scale(1.0, 1.12)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor((0.06, 0.16, 0.26, 0.18))
        if row == 0:
            cell.set_facecolor((0.07, 0.19, 0.28, 0.92))
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor((1.0, 0.985, 0.95, 0.78) if row == 1 else (1.0, 1.0, 1.0, 0.72))


def plot_2d_basin_map(
    analysis: dict[str, object],
    plot_config: PlotConfig,
    cmap: Colormap,
) -> plt.Figure:
    artifacts = analysis["artifacts"]
    minima = analysis["tables"]["minima"]
    x = artifacts["x"]
    y = artifacts["y"]
    grid = artifacts["grid"]
    basins = artifacts["basins"]
    x_name = artifacts["x_name"]
    y_name = artifacts["y_name"]
    language = plot_config.language

    fig, ax = plt.subplots(figsize=(plot_config.figure_width, plot_config.figure_width * 0.88))
    levels = np.linspace(np.nanmin(grid), np.nanmax(grid), plot_config.contour_levels)
    ax.contourf(x, y, grid, levels=levels, cmap=cmap, alpha=0.90)
    ax.imshow(
        basins,
        origin="lower",
        extent=(x.min(), x.max(), y.min(), y.max()),
        cmap="tab20",
        alpha=0.28,
        aspect="auto",
        interpolation="nearest",
    )
    ax.contour(x, y, basins, levels=np.unique(basins)[1:] - 0.5, colors="white", linewidths=0.9, alpha=0.75)

    for _, row in minima.iterrows():
        ax.scatter(row[x_name], row[y_name], s=62, color="#102f44", edgecolor="white", linewidth=1.0, zorder=4)
        ax.annotate(
            f"B{int(row['basin_id'])}",
            xy=(row[x_name], row[y_name]),
            xytext=(6, -10),
            textcoords="offset points",
            color="#102f44",
            fontsize=8,
        )

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(tr(language, "basin_decomposition"))
    ax.grid(False)
    return fig


def plot_barrier_heatmap(
    analysis: dict[str, object],
    plot_config: PlotConfig,
    cmap: Colormap,
) -> plt.Figure | None:
    barrier_matrix = analysis["tables"].get("barrier_matrix")
    if barrier_matrix is None or getattr(barrier_matrix, "empty", True):
        return None
    language = plot_config.language
    fig, ax = plt.subplots(figsize=(plot_config.figure_width * 0.72, plot_config.figure_width * 0.62))
    sns.heatmap(
        barrier_matrix,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"label": tr(language, "saddle_energy_label", energy_unit=analysis['summary']['energy_unit'])},
        ax=ax,
    )
    ax.set_title(tr(language, "barrier_matrix"))
    return fig


def plot_2d_sections(analysis: dict[str, object], plot_config: PlotConfig) -> plt.Figure:
    artifacts = analysis["artifacts"]
    x = artifacts["x"]
    y = artifacts["y"]
    x_name = artifacts["x_name"]
    y_name = artifacts["y_name"]
    section_x = artifacts["global_section_x"]
    section_y = artifacts["global_section_y"]
    energy_unit = analysis["summary"]["energy_unit"]
    language = plot_config.language

    fig, axes = plt.subplots(1, 2, figsize=(plot_config.figure_width * 1.18, plot_config.figure_width * 0.44))
    axes[0].plot(x, section_x, color="#17304f", linewidth=2.0)
    axes[0].fill_between(x, 0, section_x, color="#91c2b7", alpha=0.28)
    axes[0].set_xlabel(x_name)
    axes[0].set_ylabel(tr(language, "delta_g_label", energy_unit=energy_unit))
    axes[0].set_title(tr(language, "section_global_minimum"))

    axes[1].plot(y, section_y, color="#b64926", linewidth=2.0)
    axes[1].fill_between(y, 0, section_y, color="#f0b785", alpha=0.28)
    axes[1].set_xlabel(y_name)
    axes[1].set_ylabel(tr(language, "delta_g_label", energy_unit=energy_unit))
    axes[1].set_title(tr(language, "orthogonal_section"))
    for axis in axes:
        axis.grid(True, axis="y")
    return fig


def plotly_surface(analysis: dict[str, object], cmap: Colormap, language: str) -> go.Figure:
    artifacts = analysis["artifacts"]
    x = artifacts["x"]
    y = artifacts["y"]
    grid = artifacts["grid"]
    minima = analysis["tables"]["minima"]
    x_name = artifacts["x_name"]
    y_name = artifacts["y_name"]
    energy_unit = analysis["summary"]["energy_unit"]
    colorscale = _plotly_colorscale_from_cmap(cmap)
    surface = go.Surface(
        x=x,
        y=y,
        z=grid,
        colorscale=colorscale,
        opacity=0.94,
        colorbar={"title": tr(language, "delta_g_label", energy_unit=energy_unit)},
    )
    traces = [surface]
    scatter = go.Scatter3d(
        x=minima[x_name],
        y=minima[y_name],
        z=minima["free_energy"] + 0.1,
        mode="markers+text",
        marker={"size": 5, "color": "#b64926"},
        text=[f"M{rank}" for rank in minima["rank"]],
        textposition="top center",
    )
    traces.append(scatter)
    primary_path_key = artifacts.get("primary_path_key")
    mfep_paths = artifacts.get("mfep_paths", {})
    if primary_path_key in mfep_paths:
        path = mfep_paths[primary_path_key]
        points = np.asarray(path["points"])
        traces.append(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=np.asarray(path["energies"]) + 0.08,
                mode="lines+markers",
                line={"width": 7, "color": "#102f44"},
                marker={"size": 2.7, "color": "#f8f5ec", "line": {"width": 1, "color": "#102f44"}},
                name=path["path_id"],
            )
        )
    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_white",
        scene={
            "xaxis_title": x_name,
            "yaxis_title": y_name,
            "zaxis_title": tr(language, "delta_g_label", energy_unit=energy_unit),
            "camera": {"eye": {"x": 1.55, "y": -1.6, "z": 0.95}},
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 10},
    )
    return fig


def _plotly_colorscale_from_cmap(cmap: Colormap, n: int = 9) -> list[list[float | str]]:
    positions = np.linspace(0.0, 1.0, n)
    colorscale: list[list[float | str]] = []
    for position in positions:
        red, green, blue, _ = cmap(float(position))
        color = "#{:02x}{:02x}{:02x}".format(
            int(round(red * 255)),
            int(round(green * 255)),
            int(round(blue * 255)),
        )
        colorscale.append([float(position), color])
    return colorscale
