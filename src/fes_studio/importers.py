from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import thermal_energy
from .i18n import import_method_label
from .paths import GENERATED_FES_ROOT, default_postprocess_root, resolve_tools_root

DEFAULT_POSTPROCESS_ROOT = default_postprocess_root()


def detect_run_directory(
    run_dir: str | Path,
    tools_root: str | Path | None = None,
    language: str = "zh",
) -> dict[str, object]:
    run_text = str(run_dir).strip()
    run_path = Path(run_text).expanduser() if run_text else None
    tools_path = resolve_tools_root(tools_root)
    info: dict[str, object] = {
        "run_dir": run_path,
        "tools_root": tools_path,
        "exists": bool(run_path and run_path.exists()),
        "methods": [],
        "warnings": [],
    }
    if run_path is None or not run_path.exists():
        return info

    files = {
        "COLVAR": _maybe_path(run_path / "COLVAR"),
        "HILLS": _maybe_path(run_path / "HILLS"),
        "BIAS": _maybe_path(run_path / "BIAS"),
        "STATE": _maybe_path(run_path / "STATE"),
        "KERNELS": _maybe_path(run_path / "KERNELS"),
    }
    fes_files = sorted(
        [
            path
            for path in run_path.iterdir()
            if path.is_file() and path.suffix == ".dat" and "fes" in path.name.lower()
        ]
    )
    info["files"] = files
    info["fes_files"] = fes_files

    colvar_fields = _read_fields(files["COLVAR"]) if files["COLVAR"] else []
    hills_fields = _read_fields(files["HILLS"]) if files["HILLS"] else []
    bias_fields = _read_fields(files["BIAS"]) if files["BIAS"] else []
    kernels_fields = _read_fields(files["KERNELS"]) if files["KERNELS"] else []
    state_fields = _read_fields(files["STATE"]) if files["STATE"] else []
    info["colvar_fields"] = colvar_fields
    info["hills_fields"] = hills_fields
    info["bias_fields"] = bias_fields
    info["kernels_fields"] = kernels_fields
    info["state_fields"] = state_fields

    cv_fields = _infer_cv_fields(colvar_fields) or _infer_cv_fields(hills_fields) or _infer_cv_fields(bias_fields)
    info["cv_fields"] = cv_fields
    info["dimension"] = min(len(cv_fields), 2) if cv_fields else 0
    method_default_cvs = {
        "metad-hills": _infer_cv_fields(hills_fields) or cv_fields,
        "metad-bias": _infer_cv_fields(bias_fields) or cv_fields,
        "metad-reweight": _infer_cv_fields(hills_fields) or _infer_cv_fields(bias_fields) or cv_fields,
        "opes-state": _infer_cv_fields(state_fields) or _infer_cv_fields(kernels_fields) or cv_fields,
        "opes-reweight": _infer_cv_fields(kernels_fields) or _infer_cv_fields(state_fields) or cv_fields,
        "opes-kernels": _infer_cv_fields(kernels_fields) or cv_fields,
    }
    info["method_default_cvs"] = {key: (value[:2] if value else []) for key, value in method_default_cvs.items()}

    has_opes = bool(files["STATE"] or files["KERNELS"] or any("opes." in field for field in colvar_fields))
    has_metad = bool(files["HILLS"] or files["BIAS"] or any("metad." in field for field in colvar_fields))
    info["engine"] = "opes" if has_opes and not has_metad else "metad" if has_metad and not has_opes else "mixed"

    sigma_hints = {
        "metad": _infer_sigma_from_hills(files["HILLS"]) if files["HILLS"] else None,
        "opes": _infer_sigma_from_kernels(files["KERNELS"]) if files["KERNELS"] else None,
    }
    info["sigma_hints"] = sigma_hints
    info["biasfactor"] = _infer_biasfactor(files["HILLS"]) if files["HILLS"] else None
    plumed_available = _which("plumed") is not None
    info["plumed_available"] = plumed_available

    methods: list[dict[str, str]] = []
    for path in fes_files:
        method_id = f"existing-fes::{path.name}"
        methods.append({"id": method_id, "label": import_method_label(method_id, language)})
    if files["HILLS"]:
        methods.append({"id": "metad-hills", "label": import_method_label("metad-hills", language)})
    if files["BIAS"]:
        methods.append({"id": "metad-bias", "label": import_method_label("metad-bias", language)})
    if files["COLVAR"] and has_metad:
        methods.append({"id": "metad-reweight", "label": import_method_label("metad-reweight", language)})
    if files["COLVAR"] and has_opes:
        methods.append({"id": "opes-reweight", "label": import_method_label("opes-reweight", language)})
    if files["STATE"]:
        methods.append({"id": "opes-state", "label": import_method_label("opes-state", language)})
    if files["KERNELS"] and not files["STATE"]:
        if plumed_available:
            methods.append({"id": "opes-kernels", "label": import_method_label("opes-kernels", language)})
        else:
            warning = (
                "KERNELS was detected but plumed is unavailable on this machine, so KERNELS -> STATE conversion cannot be performed."
                if language == "en"
                else "检测到 KERNELS 但本机没有 plumed，无法做 KERNELS -> STATE 转换。"
            )
            info["warnings"].append(warning)

    info["methods"] = methods
    info["suggested_method"] = _suggest_method(methods)
    info["signature"] = _build_signature([path for path in [*files.values(), *fes_files] if path is not None])
    return info


def prepare_fes_from_run_directory(
    run_dir: str | Path,
    method_id: str,
    temperature: float,
    energy_unit: str,
    cv_fields: list[str] | None = None,
    sigma_text: str | None = None,
    bins: int = 161,
    tools_root: str | Path | None = None,
    python_executable: str | None = None,
    language: str = "zh",
) -> dict[str, object]:
    run_info = detect_run_directory(run_dir, tools_root, language=language)
    if not run_info.get("exists"):
        raise ValueError(f"Run directory does not exist: {run_dir}")

    run_path = Path(run_info["run_dir"])
    tools_path = Path(run_info["tools_root"])
    python_exec = python_executable or sys.executable
    selected_cv_fields = list(cv_fields or run_info.get("method_default_cvs", {}).get(method_id) or run_info.get("cv_fields") or [])
    selected_cv_fields = selected_cv_fields[:2]
    dim = len(selected_cv_fields)
    bins_str = str(bins) if dim <= 1 else f"{bins},{bins}"
    kbt = thermal_energy(float(temperature), energy_unit)

    payload = {
        "run_dir": str(run_path),
        "method_id": method_id,
        "temperature": float(temperature),
        "energy_unit": energy_unit,
        "cv_fields": selected_cv_fields,
        "sigma_text": sigma_text or "",
        "bins": bins,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    out_dir = GENERATED_FES_ROOT / run_path.name
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{method_id.replace(':', '_').replace('/', '_')}_{digest}.dat"

    if method_id.startswith("existing-fes::"):
        file_name = method_id.split("::", maxsplit=1)[1]
        fes_path = run_path / file_name
        return {
            "fes_path": fes_path,
            "method_id": method_id,
            "method_label": _method_label(run_info, method_id),
            "log": "Loaded existing FES file from run directory." if language == "en" else "已从工作目录读取现有 FES 文件。",
            "cv_fields": selected_cv_fields,
            "sigma_text": sigma_text,
        }

    if method_id == "metad-hills":
        if not run_info["files"]["HILLS"]:
            raise ValueError("HILLS file not found in run directory.")
        _sum_hills_to_fes(
            hills_path=run_info["files"]["HILLS"],
            output_path=output_path,
            bins=bins,
        )
        log = (
            "Generated FES from HILLS using the built-in sum_hills implementation."
            if language == "en"
            else "已使用内置 sum_hills 从 HILLS 生成 FES。"
        )
    elif method_id == "metad-bias":
        if not run_info["files"]["BIAS"]:
            raise ValueError("BIAS file not found in run directory.")
        _bias_snapshot_to_fes(
            bias_path=run_info["files"]["BIAS"],
            output_path=output_path,
            biasfactor=run_info.get("biasfactor"),
        )
        log = "Generated FES from the BIAS snapshot." if language == "en" else "已从 BIAS 快照生成 FES。"
    elif method_id in {"metad-reweight", "opes-reweight"}:
        if not run_info["files"]["COLVAR"]:
            raise ValueError("COLVAR file not found in run directory.")
        if not selected_cv_fields:
            raise ValueError("No CV columns were detected for COLVAR reweighting.")
        sigma_value = sigma_text or (
            run_info["sigma_hints"]["metad"] if method_id == "metad-reweight" else run_info["sigma_hints"]["opes"]
        )
        if not sigma_value:
            raise ValueError("Sigma could not be inferred automatically. Please provide it manually.")
        sigma_value = _normalize_sigma_text(sigma_value, dim)
        bias_arg = ".bias" if method_id == "metad-reweight" else "opes.bias"
        cmd = [
            python_exec,
            str(tools_path / "FES_from_Reweighting.py"),
            "--colvar",
            str(run_info["files"]["COLVAR"]),
            "--outfile",
            str(output_path),
            "--kt",
            f"{kbt:.10g}",
            "--cv",
            ",".join(selected_cv_fields),
            "--bias",
            bias_arg,
            "--sigma",
            sigma_value,
            "--bin",
            bins_str,
        ]
        log = _run_subprocess(cmd, cwd=run_path)
    elif method_id == "opes-state":
        if not run_info["files"]["STATE"]:
            raise ValueError("STATE file not found in run directory.")
        cmd = [
            python_exec,
            str(tools_path / "FES_from_State.py"),
            "--state",
            str(run_info["files"]["STATE"]),
            "--outfile",
            str(output_path),
            "--kt",
            f"{kbt:.10g}",
            "--bin",
            bins_str,
        ]
        log = _run_subprocess(cmd, cwd=run_path)
    elif method_id == "opes-kernels":
        if not run_info["files"]["KERNELS"]:
            raise ValueError("KERNELS file not found in run directory.")
        if _which("plumed") is None:
            raise ValueError("plumed is not available on this machine, so KERNELS cannot be converted into STATE.")
        tmp_state = out_dir / f"tmp_state_{digest}.dat"
        cmd_state = [
            python_exec,
            str(tools_path / "State_from_Kernels.py"),
            "--kernels",
            str(run_info["files"]["KERNELS"]),
            "--outfile",
            str(tmp_state),
            "--tmpname",
            str(out_dir / f"tmp_plumed_{digest}.dat"),
        ]
        state_log = _run_subprocess(cmd_state, cwd=run_path)
        cmd_fes = [
            python_exec,
            str(tools_path / "FES_from_State.py"),
            "--state",
            str(tmp_state),
            "--outfile",
            str(output_path),
            "--kt",
            f"{kbt:.10g}",
            "--bin",
            bins_str,
        ]
        log = state_log + "\n" + _run_subprocess(cmd_fes, cwd=run_path)
    else:
        raise ValueError(f"Unsupported import method: {method_id}")

    if not output_path.exists():
        raise RuntimeError(f"Expected generated FES file was not created: {output_path}")

    return {
        "fes_path": output_path,
        "method_id": method_id,
        "method_label": _method_label(run_info, method_id),
        "log": log,
        "cv_fields": selected_cv_fields,
        "sigma_text": sigma_text,
    }


def _read_fields(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#! FIELDS"):
                return line.strip().split()[2:]
    return []


def _infer_cv_fields(fields: list[str]) -> list[str]:
    if not fields:
        return []
    excluded = {"time", "height", "biasf", "logweight"}
    result = []
    for field in fields:
        lowered = field.lower()
        if field in excluded:
            continue
        if lowered.startswith("sigma_") or lowered.startswith("der_"):
            continue
        if ".bias" in lowered or ".rbias" in lowered or lowered.endswith(".rct"):
            continue
        result.append(field)
    return result[:2]


def _infer_sigma_from_hills(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    fields = _read_fields(path)
    sigma_fields = [field for field in fields if field.startswith("sigma_")]
    if not sigma_fields:
        return None
    last_row = _read_last_numeric_row(path)
    if last_row is None:
        return None
    indices = [fields.index(field) for field in sigma_fields]
    return ",".join(f"{last_row[index]:.8g}" for index in indices)


def _infer_sigma_from_kernels(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    fields = _read_fields(path)
    sigma_fields = [field for field in fields if field.startswith("sigma_")]
    if not sigma_fields:
        return None
    last_row = _read_last_numeric_row(path)
    if last_row is None:
        return None
    indices = [fields.index(field) for field in sigma_fields]
    return ",".join(f"{last_row[index]:.8g}" for index in indices)


def _normalize_sigma_text(sigma_text: str, dim: int) -> str:
    values = [value.strip() for value in sigma_text.split(",") if value.strip()]
    if dim <= 1:
        return values[0] if values else sigma_text
    if len(values) == 1:
        return ",".join([values[0]] * dim)
    if len(values) >= dim:
        return ",".join(values[:dim])
    raise ValueError(f"Sigma dimension mismatch: expected {dim} values, got {len(values)}.")


def _infer_biasfactor(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    fields = _read_fields(path)
    if "biasf" in fields:
        last_row = _read_last_numeric_row(path)
        if last_row is not None:
            return float(last_row[fields.index("biasf")])
    return None


def _sum_hills_to_fes(hills_path: Path, output_path: Path, bins: int) -> None:
    fields = _read_fields(hills_path)
    cv_names = _infer_cv_fields(fields)
    sigma_fields = [field for field in fields if field.startswith("sigma_")]
    if not cv_names or not sigma_fields:
        raise ValueError("HILLS file does not have the expected CV/sigma fields.")

    table = pd.read_table(hills_path, sep=r"\s+", comment="#", header=None)
    values = table.to_numpy(float)
    dim = min(len(cv_names), len(sigma_fields))
    cv_columns = list(range(1, 1 + dim))
    sigma_columns = list(range(1 + dim, 1 + dim + dim))
    height_idx = fields.index("height")
    biasf_idx = fields.index("biasf") if "biasf" in fields else None

    cvs = [values[:, column] for column in cv_columns]
    sigmas = [values[:, column] for column in sigma_columns]
    heights = values[:, height_idx]
    biasfactor = float(np.nanmedian(values[:, biasf_idx])) if biasf_idx is not None else None
    scale_factor = biasfactor / (biasfactor - 1.0) if biasfactor and biasfactor > 1.0 else 1.0

    ranges = []
    for cv, sigma in zip(cvs, sigmas):
        lower = float(np.nanmin(cv - 3.0 * sigma))
        upper = float(np.nanmax(cv + 3.0 * sigma))
        ranges.append((lower, upper))

    if dim == 1:
        x = np.linspace(ranges[0][0], ranges[0][1], bins)
        bias = np.zeros_like(x)
        for center, sigma, height in zip(cvs[0], sigmas[0], heights):
            bias += height * np.exp(-0.5 * ((x - center) / sigma) ** 2)
        fes = -scale_factor * bias
        fes -= np.nanmin(fes)
        lines = [f"#! FIELDS {cv_names[0]} file.free"]
        lines.extend(f"{xv: .8f} {fv: .8f}" for xv, fv in zip(x, fes))
    elif dim == 2:
        x = np.linspace(ranges[0][0], ranges[0][1], bins)
        y = np.linspace(ranges[1][0], ranges[1][1], bins)
        xx, yy = np.meshgrid(x, y)
        bias = np.zeros_like(xx)
        for cx, cy, sx, sy, height in zip(cvs[0], cvs[1], sigmas[0], sigmas[1], heights):
            bias += height * np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
        fes = -scale_factor * bias
        fes -= np.nanmin(fes)
        lines = [f"#! FIELDS {cv_names[0]} {cv_names[1]} file.free"]
        for row_x, row_y, row_f in zip(xx.ravel(), yy.ravel(), fes.ravel()):
            lines.append(f"{row_x: .8f} {row_y: .8f} {row_f: .8f}")
    else:
        raise ValueError("Only 1D and 2D HILLS files are supported.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bias_snapshot_to_fes(bias_path: Path, output_path: Path, biasfactor: float | None) -> None:
    fields = _read_fields(bias_path)
    cv_names = _infer_cv_fields(fields)
    bias_column = next((field for field in fields if field.endswith(".bias")), None)
    if not cv_names or bias_column is None:
        raise ValueError("BIAS file does not look like a PLUMED bias grid.")

    table = pd.read_table(bias_path, sep=r"\s+", comment="#", header=None)
    bias_idx = fields.index(bias_column)
    scale_factor = biasfactor / (biasfactor - 1.0) if biasfactor and biasfactor > 1.0 else 1.0

    if len(cv_names) == 1:
        cv = table.iloc[:, 0].to_numpy(float)
        bias = table.iloc[:, bias_idx].to_numpy(float)
        fes = -scale_factor * bias
        fes -= np.nanmin(fes)
        lines = [f"#! FIELDS {cv_names[0]} file.free"]
        lines.extend(f"{xv: .8f} {fv: .8f}" for xv, fv in zip(cv, fes))
    elif len(cv_names) == 2:
        cv1 = table.iloc[:, 0].to_numpy(float)
        cv2 = table.iloc[:, 1].to_numpy(float)
        bias = table.iloc[:, bias_idx].to_numpy(float)
        fes = -scale_factor * bias
        fes -= np.nanmin(fes)
        lines = [f"#! FIELDS {cv_names[0]} {cv_names[1]} file.free"]
        lines.extend(f"{xv: .8f} {yv: .8f} {fv: .8f}" for xv, yv, fv in zip(cv1, cv2, fes))
    else:
        raise ValueError("Only 1D and 2D BIAS files are supported.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_subprocess(cmd: list[str], cwd: Path) -> str:
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    log = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    if result.returncode != 0:
        joined = "\n".join(part for part in [log, err] if part)
        raise RuntimeError(joined or f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
    return "\n".join(part for part in [log, err] if part)


def _read_last_numeric_row(path: Path) -> list[float] | None:
    last_tokens: list[float] | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                last_tokens = [float(token) for token in stripped.split()]
            except ValueError:
                continue
    return last_tokens


def _build_signature(paths: list[Path]) -> str:
    parts: list[str] = []
    for path in sorted(paths):
        stat = path.stat()
        parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(parts)


def _suggest_method(methods: list[dict[str, str]]) -> str | None:
    if not methods:
        return None
    priority = [
        "opes-state",
        "metad-hills",
        "opes-reweight",
        "metad-reweight",
        "metad-bias",
    ]
    ids = {item["id"] for item in methods}
    for preferred in priority:
        if preferred in ids:
            return preferred
    return methods[0]["id"]


def _method_label(info: dict[str, object], method_id: str) -> str:
    for item in info.get("methods", []):
        if item["id"] == method_id:
            return item["label"]
    return method_id


def _maybe_path(path: Path) -> Path | None:
    return path if path.exists() else None


def _which(name: str) -> str | None:
    return shutil.which(name, path=os.environ.get("PATH"))
