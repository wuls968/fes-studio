from __future__ import annotations

from heapq import heappop, heappush
from math import inf

import networkx as nx
import numpy as np
import pandas as pd
from scipy import ndimage, signal
from scipy.interpolate import RegularGridInterpolator, griddata
from skimage.segmentation import watershed

from .models import AnalysisConfig, FESDataset

K_BOLTZ_KJ = 0.00831446261815324
K_BOLTZ_KCAL = 0.00198720425864083


def run_analysis(dataset: FESDataset, config: AnalysisConfig) -> dict[str, object]:
    if dataset.dimension == 1:
        return analyze_1d(dataset, config)
    if dataset.dimension == 2:
        return analyze_2d(dataset, config)
    return analyze_nd(dataset, config)


def thermal_energy(temperature: float, energy_unit: str) -> float:
    if energy_unit == "kJ/mol":
        return K_BOLTZ_KJ * temperature
    if energy_unit == "kcal/mol":
        return K_BOLTZ_KCAL * temperature
    if energy_unit == "kBT":
        return 1.0
    raise ValueError(f"Unsupported energy unit: {energy_unit}")


def analyze_1d(dataset: FESDataset, config: AnalysisConfig) -> dict[str, object]:
    x_name = dataset.cv_columns[0]
    frame = (
        dataset.frame[[x_name, dataset.energy_column] + ([dataset.error_column] if dataset.error_column else [])]
        .dropna()
        .groupby(x_name, as_index=False)
        .mean(numeric_only=True)
        .sort_values(x_name)
    )
    x = frame[x_name].to_numpy(float)
    energy = frame[dataset.energy_column].to_numpy(float)
    error = frame[dataset.error_column].to_numpy(float) if dataset.error_column else None

    energy = energy - np.nanmin(energy)
    smooth = ndimage.gaussian_filter1d(energy, sigma=config.smoothing_sigma) if len(energy) > 5 else energy.copy()
    energy_span = float(np.nanmax(energy) - np.nanmin(energy))
    prominence = max(energy_span * config.prominence_fraction, 1e-8)
    min_distance = max(1, len(x) // 25)

    minima_idx, _ = signal.find_peaks(-smooth, prominence=prominence, distance=min_distance)
    maxima_idx, _ = signal.find_peaks(smooth, prominence=prominence, distance=min_distance)
    minima_idx = _include_edge_minima(smooth, minima_idx)
    minima_idx = _ensure_global_minimum(energy, minima_idx)
    minima_idx = np.unique(minima_idx)
    maxima_idx = np.unique(maxima_idx)

    kbt = thermal_energy(config.temperature, config.energy_unit)
    weights = np.exp(-(energy - np.nanmin(energy)) / max(kbt, 1e-12))
    density = weights / np.trapezoid(weights, x)

    boundaries = _find_1d_boundaries(smooth, minima_idx)
    basin_rows = []
    minima_rows = []

    total_weight = np.trapezoid(weights, x)
    for rank, idx in enumerate(sorted(minima_idx, key=lambda item: energy[item])[: config.top_minima], start=1):
        basin_id = _basin_id_for_index(idx, boundaries)
        minima_rows.append(
            {
                "rank": rank,
                "index": int(idx),
                x_name: float(x[idx]),
                "free_energy": float(energy[idx]),
                "point_probability": float(weights[idx] / np.sum(weights)),
                "basin_id": basin_id,
            }
        )

    for basin_id, (left, right) in enumerate(boundaries, start=1):
        section = slice(left, right + 1)
        basin_weight = np.trapezoid(weights[section], x[section]) if right > left else weights[left]
        basin_rows.append(
            {
                "basin_id": basin_id,
                "x_left": float(x[left]),
                "x_right": float(x[right]),
                "population": float(basin_weight / total_weight if total_weight else np.nan),
                "minimum_energy": float(np.nanmin(energy[section])),
            }
        )

    barrier_rows = []
    sorted_minima = sorted(minima_idx.tolist())
    for left_idx, right_idx in zip(sorted_minima[:-1], sorted_minima[1:]):
        barrier_idx = left_idx + int(np.argmax(smooth[left_idx : right_idx + 1]))
        barrier_rows.append(
            {
                "from_index": int(left_idx),
                "to_index": int(right_idx),
                "transition_coordinate": float(x[barrier_idx]),
                "saddle_energy": float(energy[barrier_idx]),
                "forward_barrier": float(energy[barrier_idx] - energy[left_idx]),
                "reverse_barrier": float(energy[barrier_idx] - energy[right_idx]),
            }
        )

    minima_df = pd.DataFrame(minima_rows)
    basins_df = pd.DataFrame(basin_rows)
    barriers_df = pd.DataFrame(barrier_rows)
    summary = {
        "dimension": 1,
        "cv_columns": dataset.cv_columns,
        "energy_column": dataset.energy_column,
        "points": int(len(frame)),
        "regular_grid": bool(dataset.regular_grid),
        "global_minimum_coordinate": float(x[int(np.argmin(energy))]),
        "global_minimum_energy": float(np.nanmin(energy)),
        "energy_max": float(np.nanmax(energy)),
        "temperature": float(config.temperature),
        "energy_unit": config.energy_unit,
        "detected_minima": int(len(minima_idx)),
        "detected_barriers": int(len(barriers_df)),
    }
    profiles = pd.DataFrame(
        {
            x_name: x,
            "free_energy": energy,
            "smoothed_free_energy": smooth,
            "probability_density": density,
            **({dataset.error_column: error} if error is not None else {}),
        }
    )

    return {
        "mode": "1d",
        "summary": summary,
        "tables": {"minima": minima_df, "barriers": barriers_df, "basins": basins_df, "profiles": profiles},
        "artifacts": {
            "x": x,
            "energy": energy,
            "error": error,
            "smooth": smooth,
            "density": density,
            "x_name": x_name,
        },
    }


def analyze_2d(dataset: FESDataset, config: AnalysisConfig) -> dict[str, object]:
    x_name, y_name = dataset.cv_columns[:2]
    x, y, grid, source_kind = _prepare_2d_grid(dataset, config)
    grid = grid - np.nanmin(grid)
    finite_mask = np.isfinite(grid)
    smooth = ndimage.gaussian_filter(np.where(finite_mask, grid, np.nanmax(grid)), sigma=config.smoothing_sigma)

    minima_coords = _find_2d_minima(smooth, grid, config)
    minima_coords = minima_coords[: config.top_minima]
    markers = np.zeros_like(grid, dtype=int)
    for idx, (row, col) in enumerate(minima_coords, start=1):
        markers[row, col] = idx
    basins = watershed(smooth, markers, mask=finite_mask)

    kbt = thermal_energy(config.temperature, config.energy_unit)
    weights = np.exp(-(grid - np.nanmin(grid)) / max(kbt, 1e-12))
    dx = float(np.nanmedian(np.diff(x))) if len(x) > 1 else 1.0
    dy = float(np.nanmedian(np.diff(y))) if len(y) > 1 else 1.0
    cell_area = abs(dx * dy) if dx and dy else 1.0
    total_population = float(np.nansum(weights) * cell_area)

    minima_rows = []
    basin_rows = []
    for rank, (row, col) in enumerate(sorted(minima_coords, key=lambda item: grid[item]), start=1):
        label = basins[row, col]
        basin_mask = basins == label
        basin_population = float(np.nansum(weights[basin_mask]) * cell_area / total_population) if total_population else np.nan
        minima_rows.append(
            {
                "rank": rank,
                x_name: float(x[col]),
                y_name: float(y[row]),
                "row_index": int(row),
                "col_index": int(col),
                "free_energy": float(grid[row, col]),
                "basin_id": int(label),
                "basin_population": basin_population,
            }
        )

    max_population = 0.0
    for label in np.unique(basins[finite_mask]):
        if label == 0:
            continue
        basin_mask = basins == label
        basin_population = float(np.nansum(weights[basin_mask]) * cell_area / total_population) if total_population else np.nan
        max_population = max(max_population, basin_population)
        row, col = np.argwhere((markers == label)).tolist()[0]
        basin_rows.append(
            {
                "basin_id": int(label),
                "minimum_x": float(x[col]),
                "minimum_y": float(y[row]),
                "minimum_energy": float(grid[row, col]),
                "population": basin_population,
                "area": float(np.sum(basin_mask) * cell_area),
            }
        )
    basin_df = pd.DataFrame(basin_rows)
    if not basin_df.empty and max_population > 0:
        basin_df["basin_delta_g"] = -kbt * np.log(basin_df["population"] / max_population)

    barrier_rows = []
    mfep_rows = []
    mfep_profile_rows = []
    path_store: dict[tuple[int, int], list[tuple[float, float]]] = {}
    mfep_paths: dict[tuple[int, int], dict[str, object]] = {}
    cost_maps: dict[int, np.ndarray] = {}
    parent_maps: dict[int, np.ndarray] = {}
    graph = nx.Graph()
    surface = _build_surface_model(x, y, grid)
    for start_idx, (row, col) in enumerate(minima_coords):
        costs, parents = _minimax_dijkstra(grid, (row, col), finite_mask)
        cost_maps[start_idx] = costs
        parent_maps[start_idx] = parents

    for i, (row_i, col_i) in enumerate(minima_coords):
        graph.add_node(i)
        for j in range(i + 1, len(minima_coords)):
            row_j, col_j = minima_coords[j]
            coarse_path = _reconstruct_path(parent_maps[i], minima_coords[i], minima_coords[j])
            mfep_result = _refine_mfep_path(
                coarse_path=coarse_path,
                x=x,
                y=y,
                surface=surface,
                config=config,
                state_i=i + 1,
                state_j=j + 1,
            )
            mfep_paths[(i + 1, j + 1)] = mfep_result
            graph.add_edge(i, j, weight=mfep_result["saddle_energy"])
            barrier_rows.append(
                {
                    "state_i": int(i + 1),
                    "state_j": int(j + 1),
                    "saddle_energy": float(mfep_result["saddle_energy"]),
                    "barrier_i_to_j": float(mfep_result["barrier_i_to_j"]),
                    "barrier_j_to_i": float(mfep_result["barrier_j_to_i"]),
                }
            )
            mfep_rows.append(
                {
                    "path_id": mfep_result["path_id"],
                    "state_i": int(i + 1),
                    "state_j": int(j + 1),
                    "schedule_mode": mfep_result["schedule_mode"],
                    "stage_count": int(mfep_result["stage_count"]),
                    "schedule_compact": mfep_result["schedule_compact"],
                    "coarse_images": int(mfep_result["coarse_images"]),
                    "coarse_iterations": int(mfep_result["coarse_iterations"]),
                    "coarse_step_size": float(mfep_result["coarse_step_size"]),
                    "refine_images": int(mfep_result["refine_images"]),
                    "refine_iterations": int(mfep_result["refine_iterations"]),
                    "refine_step_size": float(mfep_result["refine_step_size"]),
                    "images": int(mfep_result["images"]),
                    "arc_length": float(mfep_result["arc_length"]),
                    "segment_min": float(mfep_result["segment_min"]),
                    "segment_max": float(mfep_result["segment_max"]),
                    "segment_mean": float(mfep_result["segment_mean"]),
                    "spacing_cv": float(mfep_result["spacing_cv"]),
                    x_name: float(mfep_result["start_point"][0]),
                    y_name: float(mfep_result["start_point"][1]),
                    f"{x_name}_end": float(mfep_result["end_point"][0]),
                    f"{y_name}_end": float(mfep_result["end_point"][1]),
                    "saddle_progress": float(mfep_result["progress"][mfep_result["saddle_index"]]),
                    "saddle_x": float(mfep_result["saddle_point"][0]),
                    "saddle_y": float(mfep_result["saddle_point"][1]),
                    "saddle_energy": float(mfep_result["saddle_energy"]),
                    "barrier_i_to_j": float(mfep_result["barrier_i_to_j"]),
                    "barrier_j_to_i": float(mfep_result["barrier_j_to_i"]),
                    "path_method": mfep_result["method"],
                }
            )
            for image_index, (progress, point, energy_value, grad_norm) in enumerate(
                zip(
                    mfep_result["progress"],
                    mfep_result["points"],
                    mfep_result["energies"],
                    mfep_result["gradient_norm"],
                ),
                start=1,
            ):
                mfep_profile_rows.append(
                    {
                        "path_id": mfep_result["path_id"],
                        "state_i": int(i + 1),
                        "state_j": int(j + 1),
                        "image_index": int(image_index),
                        "path_coordinate": float(progress),
                        x_name: float(point[0]),
                        y_name: float(point[1]),
                        "free_energy": float(energy_value),
                        "gradient_norm": float(grad_norm),
                        "is_saddle": bool(image_index - 1 == mfep_result["saddle_index"]),
                    }
                )

    if graph.number_of_edges() > 0:
        mst = nx.minimum_spanning_tree(graph, weight="weight")
        for left, right in mst.edges():
            key = (left + 1, right + 1)
            path_store[key] = [tuple(map(float, point)) for point in mfep_paths[key]["points"]]

    barrier_df = pd.DataFrame(
        barrier_rows,
        columns=["state_i", "state_j", "saddle_energy", "barrier_i_to_j", "barrier_j_to_i"],
    )
    if not barrier_df.empty:
        barrier_df = barrier_df.sort_values(["saddle_energy", "state_i", "state_j"], ignore_index=True)

    mfep_paths_df = pd.DataFrame(
        mfep_rows,
        columns=[
            "path_id",
            "state_i",
            "state_j",
            "schedule_mode",
            "stage_count",
            "schedule_compact",
            "coarse_images",
            "coarse_iterations",
            "coarse_step_size",
            "refine_images",
            "refine_iterations",
            "refine_step_size",
            "images",
            "arc_length",
            "segment_min",
            "segment_max",
            "segment_mean",
            "spacing_cv",
            x_name,
            y_name,
            f"{x_name}_end",
            f"{y_name}_end",
            "saddle_progress",
            "saddle_x",
            "saddle_y",
            "saddle_energy",
            "barrier_i_to_j",
            "barrier_j_to_i",
            "path_method",
        ],
    )
    if not mfep_paths_df.empty:
        mfep_paths_df = mfep_paths_df.sort_values(["saddle_energy", "state_i", "state_j"], ignore_index=True)
    mfep_profiles_df = pd.DataFrame(mfep_profile_rows)
    barrier_matrix = _build_barrier_matrix(barrier_df, len(minima_coords))
    requested_primary_pair = _parse_primary_pair(config.primary_mfep_pair)
    primary_path_key, primary_selection_status = _select_primary_path_key(mfep_paths, requested_primary_pair)
    primary_profile_df = (
        mfep_profiles_df[
            (mfep_profiles_df["state_i"] == primary_path_key[0]) & (mfep_profiles_df["state_j"] == primary_path_key[1])
        ].reset_index(drop=True)
        if primary_path_key is not None and not mfep_profiles_df.empty
        else pd.DataFrame(columns=["path_id", "path_coordinate", x_name, y_name, "free_energy", "gradient_norm", "is_saddle"])
    )
    primary_path = mfep_paths.get(primary_path_key) if primary_path_key is not None else None

    minima_df = pd.DataFrame(minima_rows).sort_values("free_energy", ignore_index=True)
    boltz_x = _boltzmann_marginal(grid, axis=0, kbt=kbt)
    boltz_y = _boltzmann_marginal(grid, axis=1, kbt=kbt)
    minproj_x = np.nanmin(grid, axis=0)
    minproj_y = np.nanmin(grid, axis=1)
    minproj_x -= np.nanmin(minproj_x)
    minproj_y -= np.nanmin(minproj_y)

    global_row, global_col = minima_coords[int(np.argmin([grid[idx] for idx in minima_coords]))]
    summary = {
        "dimension": 2,
        "cv_columns": dataset.cv_columns[:2],
        "energy_column": dataset.energy_column,
        "points": int(dataset.frame.shape[0]),
        "grid_shape": tuple(int(value) for value in grid.shape),
        "regular_grid": bool(dataset.regular_grid),
        "grid_source": source_kind,
        "global_minimum_x": float(x[global_col]),
        "global_minimum_y": float(y[global_row]),
        "global_minimum_energy": float(np.nanmin(grid)),
        "energy_max": float(np.nanmax(grid)),
        "temperature": float(config.temperature),
        "energy_unit": config.energy_unit,
        "detected_minima": int(len(minima_df)),
        "detected_basins": int(len(basin_df)),
        "mfep_method": "Two-stage NEB-inspired elastic string" if config.mfep_two_stage else "NEB-inspired elastic string",
        "mfep_schedule_mode": "two-stage" if config.mfep_two_stage else "single-stage",
        "mfep_refine_images": int(config.mfep_images),
        "mfep_refine_iterations": int(config.mfep_iterations),
        "mfep_refine_step_size": float(config.mfep_step_size),
        "mfep_path_count": int(len(mfep_paths_df)),
    }
    if config.mfep_two_stage:
        summary["mfep_coarse_images"] = int(_resolve_mfep_coarse_images(config))
        summary["mfep_coarse_iterations"] = int(_resolve_mfep_coarse_iterations(config))
        summary["mfep_coarse_step_size"] = float(_resolve_mfep_coarse_step_size(config))
    if primary_path is not None:
        summary["primary_mfep"] = primary_path["path_id"]
        summary["primary_mfep_saddle_energy"] = float(primary_path["saddle_energy"])
        summary["primary_mfep_schedule"] = primary_path["schedule_compact"]
    if requested_primary_pair is not None:
        summary["primary_mfep_requested"] = f"S{requested_primary_pair[0]}-S{requested_primary_pair[1]}"
        summary["primary_mfep_request_status"] = primary_selection_status
    else:
        summary["primary_mfep_request_status"] = "automatic"

    return {
        "mode": "2d",
        "summary": summary,
        "tables": {
            "minima": minima_df,
            "barriers": barrier_df,
            "basins": basin_df,
            "barrier_matrix": barrier_matrix,
            "mfep_paths": mfep_paths_df,
            "mfep_profiles": mfep_profiles_df,
            "mfep_profile_primary": primary_profile_df,
            "x_marginal": pd.DataFrame({x_name: x, "boltzmann_profile": boltz_x, "min_projection": minproj_x}),
            "y_marginal": pd.DataFrame({y_name: y, "boltzmann_profile": boltz_y, "min_projection": minproj_y}),
        },
        "artifacts": {
            "x": x,
            "y": y,
            "grid": grid,
            "smooth": smooth,
            "basins": basins,
            "paths": path_store,
            "mfep_paths": mfep_paths,
            "primary_path_key": primary_path_key,
            "primary_selection_status": primary_selection_status,
            "minima_coords": minima_coords,
            "x_name": x_name,
            "y_name": y_name,
            "global_section_x": grid[global_row, :] - np.nanmin(grid[global_row, :]),
            "global_section_y": grid[:, global_col] - np.nanmin(grid[:, global_col]),
        },
    }


def analyze_nd(dataset: FESDataset, config: AnalysisConfig) -> dict[str, object]:
    frame = dataset.frame[dataset.cv_columns + [dataset.energy_column]].dropna().copy()
    frame["free_energy"] = frame[dataset.energy_column] - frame[dataset.energy_column].min()
    kbt = thermal_energy(config.temperature, config.energy_unit)
    frame["weight"] = np.exp(-frame["free_energy"] / max(kbt, 1e-12))
    frame["probability"] = frame["weight"] / frame["weight"].sum()

    minima_df = frame.nsmallest(config.top_minima, "free_energy").reset_index(drop=True)
    minima_df.insert(0, "rank", np.arange(1, len(minima_df) + 1))
    summary = {
        "dimension": dataset.dimension,
        "cv_columns": dataset.cv_columns,
        "energy_column": dataset.energy_column,
        "points": int(len(frame)),
        "regular_grid": bool(dataset.regular_grid),
        "global_minimum_energy": float(frame["free_energy"].min()),
        "temperature": float(config.temperature),
        "energy_unit": config.energy_unit,
    }
    return {
        "mode": "nd",
        "summary": summary,
        "tables": {"minima": minima_df, "annotated_points": frame},
        "artifacts": {},
    }


def _prepare_2d_grid(dataset: FESDataset, config: AnalysisConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    x_name, y_name = dataset.cv_columns[:2]
    subset = dataset.frame[[x_name, y_name, dataset.energy_column]].dropna()
    if dataset.regular_grid:
        pivot = subset.pivot_table(index=y_name, columns=x_name, values=dataset.energy_column, aggfunc="mean")
        x = pivot.columns.to_numpy(float)
        y = pivot.index.to_numpy(float)
        return x, y, pivot.to_numpy(float), "native-grid"

    x_raw = subset[x_name].to_numpy(float)
    y_raw = subset[y_name].to_numpy(float)
    z_raw = subset[dataset.energy_column].to_numpy(float)
    x = np.linspace(np.min(x_raw), np.max(x_raw), config.interpolation_points)
    y = np.linspace(np.min(y_raw), np.max(y_raw), config.interpolation_points)
    grid_x, grid_y = np.meshgrid(x, y)
    cubic = griddata((x_raw, y_raw), z_raw, (grid_x, grid_y), method="cubic")
    nearest = griddata((x_raw, y_raw), z_raw, (grid_x, grid_y), method="nearest")
    grid = np.where(np.isfinite(cubic), cubic, nearest)
    return x, y, grid, "interpolated"


def _find_2d_minima(smooth: np.ndarray, grid: np.ndarray, config: AnalysisConfig) -> list[tuple[int, int]]:
    filtered = ndimage.minimum_filter(smooth, size=config.minima_neighborhood, mode="nearest")
    mask = smooth == filtered
    coords = [tuple(coord) for coord in np.argwhere(mask)]
    coords.sort(key=lambda item: grid[item])

    min_sep = max(3, min(grid.shape) // 18)
    selected: list[tuple[int, int]] = []
    for coord in coords:
        if not np.isfinite(grid[coord]):
            continue
        if any(np.hypot(coord[0] - other[0], coord[1] - other[1]) < min_sep for other in selected):
            continue
        selected.append(coord)
        if len(selected) >= config.top_minima * 2:
            break

    if not selected:
        selected = [tuple(np.unravel_index(int(np.nanargmin(grid)), grid.shape))]
    return selected


def _minimax_dijkstra(
    grid: np.ndarray,
    start: tuple[int, int],
    finite_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = grid.shape
    costs = np.full((rows, cols), np.inf, dtype=float)
    parents = np.full((rows, cols, 2), -1, dtype=int)
    queue: list[tuple[float, int, int]] = []

    costs[start] = float(grid[start])
    heappush(queue, (costs[start], start[0], start[1]))

    while queue:
        current_cost, row, col = heappop(queue)
        if current_cost > costs[row, col]:
            continue
        for next_row, next_col in _neighbors(row, col, rows, cols):
            if not finite_mask[next_row, next_col]:
                continue
            next_cost = max(current_cost, float(grid[next_row, next_col]))
            if next_cost < costs[next_row, next_col]:
                costs[next_row, next_col] = next_cost
                parents[next_row, next_col] = (row, col)
                heappush(queue, (next_cost, next_row, next_col))

    return costs, parents


def _reconstruct_path(
    parents: np.ndarray,
    start: tuple[int, int],
    target: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [target]
    current = target
    guard = 0
    while current != start and guard < parents.shape[0] * parents.shape[1]:
        prev_row, prev_col = parents[current]
        if prev_row < 0:
            break
        current = (int(prev_row), int(prev_col))
        path.append(current)
        guard += 1
    path.reverse()
    return path


def _neighbors(row: int, col: int, rows: int, cols: int) -> list[tuple[int, int]]:
    neighbor_list: list[tuple[int, int]] = []
    for d_row in (-1, 0, 1):
        for d_col in (-1, 0, 1):
            if d_row == 0 and d_col == 0:
                continue
            next_row = row + d_row
            next_col = col + d_col
            if 0 <= next_row < rows and 0 <= next_col < cols:
                neighbor_list.append((next_row, next_col))
    return neighbor_list


def _build_barrier_matrix(barrier_df: pd.DataFrame, state_count: int) -> pd.DataFrame:
    if state_count == 0:
        return pd.DataFrame()
    matrix = np.zeros((state_count, state_count))
    matrix[:] = np.nan
    np.fill_diagonal(matrix, 0.0)
    for _, row in barrier_df.iterrows():
        left = int(row["state_i"]) - 1
        right = int(row["state_j"]) - 1
        matrix[left, right] = row["saddle_energy"]
        matrix[right, left] = row["saddle_energy"]
    labels = [f"S{idx + 1}" for idx in range(state_count)]
    return pd.DataFrame(matrix, index=labels, columns=labels)


def _build_surface_model(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> dict[str, object]:
    grad_y, grad_x = np.gradient(grid, y, x)
    origin = np.array([float(np.min(x)), float(np.min(y))], dtype=float)
    scale = np.array(
        [
            max(float(np.max(x) - np.min(x)), 1e-12),
            max(float(np.max(y) - np.min(y)), 1e-12),
        ],
        dtype=float,
    )
    return {
        "origin": origin,
        "scale": scale,
        "energy_interp": RegularGridInterpolator((y, x), grid, bounds_error=False, fill_value=np.nan),
        "grad_x_interp": RegularGridInterpolator((y, x), grad_x, bounds_error=False, fill_value=np.nan),
        "grad_y_interp": RegularGridInterpolator((y, x), grad_y, bounds_error=False, fill_value=np.nan),
    }


def _refine_mfep_path(
    coarse_path: list[tuple[int, int]],
    x: np.ndarray,
    y: np.ndarray,
    surface: dict[str, object],
    config: AnalysisConfig,
    state_i: int,
    state_j: int,
) -> dict[str, object]:
    coarse_points = np.array([(float(x[col]), float(y[row])) for row, col in coarse_path], dtype=float)
    schedule = _build_mfep_schedule(config)
    path = _resample_polyline(coarse_points, int(schedule[0]["images"]))
    stage_records: list[dict[str, object]] = []
    for stage in schedule:
        target_images = int(stage["images"])
        if len(path) != target_images:
            path = _resample_polyline(path, target_images)
        path, stage_meta = _optimize_mfep_stage(
            path=path,
            anchor_start=coarse_points[0],
            anchor_end=coarse_points[-1],
            x=x,
            y=y,
            surface=surface,
            iterations=int(stage["iterations"]),
            step_size=float(stage["step_size"]),
            spring_constant=float(config.mfep_spring_constant),
            climbing=bool(stage["climbing"]),
        )
        stage_records.append(
            {
                "name": stage["name"],
                "images": int(stage["images"]),
                "iterations": int(stage["iterations"]),
                "iterations_used": int(stage_meta["iterations_used"]),
                "step_size": float(stage["step_size"]),
                "climbing": bool(stage["climbing"]),
            }
        )

    points = _redistribute_beads(path)
    energies, gradients = _sample_surface(surface, points)
    gradient_norm = np.linalg.norm(gradients, axis=1)
    progress = _path_coordinate(points)
    saddle_idx = _find_path_saddle_index(energies)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    start_energy = float(energies[0])
    end_energy = float(energies[-1])
    saddle_energy = float(energies[saddle_idx])
    coarse_stage = stage_records[0]
    refine_stage = stage_records[-1]
    schedule_mode = "two-stage" if len(stage_records) > 1 else "single-stage"

    return {
        "path_id": f"S{state_i}-S{state_j}",
        "method": "elastic-string-ci-two-stage" if len(stage_records) > 1 else "elastic-string-ci",
        "schedule_mode": schedule_mode,
        "stage_count": len(stage_records),
        "stage_records": stage_records,
        "schedule_compact": _format_mfep_schedule(stage_records),
        "coarse_images": int(coarse_stage["images"]),
        "coarse_iterations": int(coarse_stage["iterations"]),
        "coarse_step_size": float(coarse_stage["step_size"]),
        "refine_images": int(refine_stage["images"]),
        "refine_iterations": int(refine_stage["iterations"]),
        "refine_step_size": float(refine_stage["step_size"]),
        "images": len(points),
        "points": points,
        "energies": energies,
        "gradient_norm": gradient_norm,
        "progress": progress,
        "arc_length": float(_path_coordinate(points, normalize=False)[-1]),
        "segment_min": float(np.min(segment_lengths)) if len(segment_lengths) else 0.0,
        "segment_max": float(np.max(segment_lengths)) if len(segment_lengths) else 0.0,
        "segment_mean": float(np.mean(segment_lengths)) if len(segment_lengths) else 0.0,
        "spacing_cv": float(np.std(segment_lengths) / np.mean(segment_lengths)) if len(segment_lengths) and np.mean(segment_lengths) > 0 else 0.0,
        "saddle_index": saddle_idx,
        "saddle_point": points[saddle_idx],
        "saddle_energy": saddle_energy,
        "barrier_i_to_j": float(saddle_energy - start_energy),
        "barrier_j_to_i": float(saddle_energy - end_energy),
        "start_point": points[0],
        "end_point": points[-1],
    }


def _build_mfep_schedule(config: AnalysisConfig) -> list[dict[str, object]]:
    fine_images = max(_make_odd(int(config.mfep_images)), 5)
    fine_iterations = max(int(config.mfep_iterations), 1)
    fine_step_size = max(float(config.mfep_step_size), 1e-4)
    if not config.mfep_two_stage:
        return [
            {
                "name": "refine",
                "images": fine_images,
                "iterations": fine_iterations,
                "step_size": fine_step_size,
                "climbing": bool(config.mfep_climbing_image),
            }
        ]

    coarse_images = _resolve_mfep_coarse_images(config)
    coarse_iterations = _resolve_mfep_coarse_iterations(config)
    coarse_step_size = _resolve_mfep_coarse_step_size(config)
    return [
        {
            "name": "coarse",
            "images": coarse_images,
            "iterations": coarse_iterations,
            "step_size": coarse_step_size,
            "climbing": False,
        },
        {
            "name": "refine",
            "images": fine_images,
            "iterations": fine_iterations,
            "step_size": fine_step_size,
            "climbing": bool(config.mfep_climbing_image),
        },
    ]


def _resolve_mfep_coarse_images(config: AnalysisConfig) -> int:
    fine_images = max(_make_odd(int(config.mfep_images)), 5)
    default_value = fine_images if fine_images <= 7 else max(5, _make_odd(int(round(fine_images * 0.55))) - 2)
    coarse_images = int(config.mfep_coarse_images) if config.mfep_coarse_images is not None else default_value
    coarse_images = max(5, _make_odd(coarse_images))
    if fine_images > 7:
        coarse_images = min(coarse_images, fine_images - 2)
        coarse_images = max(5, _make_odd(coarse_images))
    return min(coarse_images, fine_images)


def _resolve_mfep_coarse_iterations(config: AnalysisConfig) -> int:
    fine_iterations = max(int(config.mfep_iterations), 1)
    if config.mfep_coarse_iterations is not None:
        return max(int(config.mfep_coarse_iterations), 1)
    estimated = int(round(max(fine_iterations * 0.45, 40.0) / 20.0) * 20)
    return max(40, estimated)


def _resolve_mfep_coarse_step_size(config: AnalysisConfig) -> float:
    fine_step_size = max(float(config.mfep_step_size), 1e-4)
    if config.mfep_coarse_step_size is not None:
        return max(float(config.mfep_coarse_step_size), fine_step_size)
    return min(0.18, max(fine_step_size * 1.8, fine_step_size + 0.03))


def _format_mfep_schedule(stage_records: list[dict[str, object]]) -> str:
    if not stage_records:
        return ""
    if len(stage_records) == 1:
        stage = stage_records[0]
        return f"{stage['images']} beads @ {stage['step_size']:.2f}"
    return f"{stage_records[0]['images']}→{stage_records[-1]['images']} beads | {stage_records[0]['step_size']:.2f}→{stage_records[-1]['step_size']:.2f}"


def _make_odd(value: int) -> int:
    return value if value % 2 == 1 else value - 1


def _optimize_mfep_stage(
    path: np.ndarray,
    anchor_start: np.ndarray,
    anchor_end: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    surface: dict[str, object],
    iterations: int,
    step_size: float,
    spring_constant: float,
    climbing: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    dx = float(np.nanmedian(np.diff(x))) if len(x) > 1 else 1.0
    dy = float(np.nanmedian(np.diff(y))) if len(y) > 1 else 1.0
    mean_step = max(0.5 * (abs(dx) + abs(dy)), 1e-6)
    max_disp = max((0.75 + 4.0 * step_size) * mean_step, 1e-6)
    laplacian_weight = 0.20 * spring_constant
    used_iterations = max(int(iterations), 1)

    for iteration in range(max(int(iterations), 1)):
        energies, gradients = _sample_surface(surface, path)
        tangents = _compute_path_tangents(path)
        updated = path.copy()

        interior_energies = energies[1:-1]
        highest_idx = None
        if interior_energies.size > 0 and np.any(np.isfinite(interior_energies)):
            highest_idx = 1 + int(np.nanargmax(interior_energies))

        for idx in range(1, len(path) - 1):
            current_point = path[idx]
            gradient = gradients[idx]
            if not np.all(np.isfinite(gradient)):
                continue
            tangent = tangents[idx]
            grad_perp = gradient - np.dot(gradient, tangent) * tangent
            force = np.zeros(2, dtype=float)
            grad_perp_norm = np.linalg.norm(grad_perp)
            if grad_perp_norm > 1e-12:
                force += -(step_size / grad_perp_norm) * grad_perp
            force += laplacian_weight * (0.5 * (path[idx - 1] + path[idx + 1]) - current_point)

            if climbing and highest_idx is not None and idx == highest_idx and iteration >= max(int(iterations) // 3, 1):
                tangent_grad = np.dot(gradient, tangent)
                if abs(tangent_grad) > 1e-12:
                    force += -0.35 * step_size * np.sign(tangent_grad) * tangent

            displacement_norm = np.linalg.norm(force)
            if displacement_norm > max_disp:
                force *= max_disp / displacement_norm
            updated[idx] = current_point + force

        updated[1:-1, 0] = np.clip(updated[1:-1, 0], x_min, x_max)
        updated[1:-1, 1] = np.clip(updated[1:-1, 1], y_min, y_max)
        updated = _redistribute_beads(updated)
        updated = _smooth_string_geometry(updated, weight=0.09 if len(path) <= 25 else 0.07)
        updated[0] = anchor_start
        updated[-1] = anchor_end

        if np.nanmax(np.linalg.norm(updated - path, axis=1)) < 0.02 * mean_step:
            path = updated
            used_iterations = iteration + 1
            break
        path = updated

    path = _redistribute_beads(path)
    path = _smooth_string_geometry(path, weight=0.05)
    path[0] = anchor_start
    path[-1] = anchor_end
    return path, {"iterations_used": used_iterations}


def _sample_surface(surface: dict[str, object], points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sample_points = np.column_stack([points[:, 1], points[:, 0]])
    energies = np.asarray(surface["energy_interp"](sample_points), dtype=float)
    grad_x = np.asarray(surface["grad_x_interp"](sample_points), dtype=float)
    grad_y = np.asarray(surface["grad_y_interp"](sample_points), dtype=float)
    return energies, np.column_stack([grad_x, grad_y])


def _resample_polyline(points: np.ndarray, n_points: int) -> np.ndarray:
    if len(points) == 1:
        return np.repeat(points, n_points, axis=0)

    progress = _path_coordinate(points)
    if np.allclose(progress, 0.0):
        return np.repeat(points[:1], n_points, axis=0)

    target = np.linspace(0.0, 1.0, n_points)
    resampled = np.column_stack(
        [
            np.interp(target, progress, points[:, 0]),
            np.interp(target, progress, points[:, 1]),
        ]
    )
    resampled[0] = points[0]
    resampled[-1] = points[-1]
    return resampled


def _redistribute_beads(points: np.ndarray) -> np.ndarray:
    redistributed = _resample_polyline(points, len(points))
    redistributed[0] = points[0]
    redistributed[-1] = points[-1]
    return redistributed


def _smooth_string_geometry(points: np.ndarray, weight: float) -> np.ndarray:
    if len(points) <= 2 or weight <= 0:
        return points
    smoothed = points.copy()
    smoothed[1:-1] = (1.0 - weight) * points[1:-1] + 0.5 * weight * (points[:-2] + points[2:])
    smoothed[0] = points[0]
    smoothed[-1] = points[-1]
    return _redistribute_beads(smoothed)


def _compute_path_tangents(points: np.ndarray) -> np.ndarray:
    tangents = np.zeros_like(points)
    if len(points) == 1:
        tangents[0] = np.array([1.0, 0.0])
        return tangents

    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    if len(points) > 2:
        tangents[1:-1] = points[2:] - points[:-2]

    norms = np.linalg.norm(tangents, axis=1)
    valid = norms > 1e-12
    tangents[valid] = tangents[valid] / norms[valid][:, None]
    tangents[~valid] = np.array([1.0, 0.0])
    return tangents


def _path_coordinate(points: np.ndarray, normalize: bool = True) -> np.ndarray:
    if len(points) <= 1:
        return np.zeros(len(points), dtype=float)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    progress = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    if normalize and progress[-1] > 0:
        progress = progress / progress[-1]
    return progress


def _find_path_saddle_index(energies: np.ndarray) -> int:
    if len(energies) <= 2:
        return int(np.nanargmax(energies))
    interior = energies[1:-1]
    if interior.size == 0 or not np.any(np.isfinite(interior)):
        return int(np.nanargmax(energies))
    return int(1 + np.nanargmax(interior))


def _parse_primary_pair(raw: str | None) -> tuple[int, int] | None:
    if raw is None:
        return None
    cleaned = raw.strip().upper().replace("S", "").replace(" ", "")
    if not cleaned:
        return None
    for sep in ("-", ",", ":", "/", "_"):
        if sep in cleaned:
            left, right = cleaned.split(sep, maxsplit=1)
            try:
                state_i = int(left)
                state_j = int(right)
            except ValueError:
                return None
            if state_i <= 0 or state_j <= 0 or state_i == state_j:
                return None
            return tuple(sorted((state_i, state_j)))
    return None


def _select_primary_path_key(
    mfep_paths: dict[tuple[int, int], dict[str, object]],
    preferred_pair: tuple[int, int] | None = None,
) -> tuple[tuple[int, int] | None, str]:
    if not mfep_paths:
        return None, "automatic"
    if preferred_pair is not None and preferred_pair in mfep_paths:
        return preferred_pair, "requested"
    if (1, 2) in mfep_paths and preferred_pair is None:
        return (1, 2), "automatic"
    auto_selected = min(
        mfep_paths,
        key=lambda key: (
            mfep_paths[key]["saddle_energy"],
            mfep_paths[key]["barrier_i_to_j"] + mfep_paths[key]["barrier_j_to_i"],
            key,
        ),
    )
    if preferred_pair is not None:
        return auto_selected, "fallback-auto"
    return auto_selected, "automatic"


def _include_edge_minima(smooth: np.ndarray, minima_idx: np.ndarray) -> np.ndarray:
    collected = minima_idx.tolist()
    if len(smooth) >= 2:
        if smooth[0] <= smooth[1]:
            collected.append(0)
        if smooth[-1] <= smooth[-2]:
            collected.append(len(smooth) - 1)
    return np.array(sorted(set(collected)), dtype=int)


def _ensure_global_minimum(energy: np.ndarray, minima_idx: np.ndarray) -> np.ndarray:
    if len(minima_idx) == 0:
        return np.array([int(np.nanargmin(energy))], dtype=int)
    return np.unique(np.append(minima_idx, int(np.nanargmin(energy))))


def _find_1d_boundaries(smooth: np.ndarray, minima_idx: np.ndarray) -> list[tuple[int, int]]:
    ordered = sorted(minima_idx.tolist())
    if not ordered:
        return [(0, len(smooth) - 1)]

    cut_points = [0]
    for left, right in zip(ordered[:-1], ordered[1:]):
        segment = smooth[left : right + 1]
        cut_points.append(left + int(np.argmax(segment)))
    cut_points.append(len(smooth) - 1)

    boundaries = []
    for left, right in zip(cut_points[:-1], cut_points[1:]):
        boundaries.append((left, right))
    return boundaries


def _basin_id_for_index(index: int, boundaries: list[tuple[int, int]]) -> int:
    for basin_id, (left, right) in enumerate(boundaries, start=1):
        if left <= index <= right:
            return basin_id
    return len(boundaries)


def _boltzmann_marginal(grid: np.ndarray, axis: int, kbt: float) -> np.ndarray:
    shifted = grid - np.nanmin(grid)
    marginal = -kbt * np.log(np.nansum(np.exp(-shifted / max(kbt, 1e-12)), axis=axis))
    marginal -= np.nanmin(marginal)
    return marginal
