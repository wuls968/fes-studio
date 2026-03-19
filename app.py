from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from fes_studio.analysis import run_analysis
from fes_studio.demo import ensure_demo_files
from fes_studio.export import export_bundle, make_output_dir
from fes_studio.i18n import figure_display_name, summary_display_name, table_display_name, tr
from fes_studio.importers import detect_run_directory, prepare_fes_from_run_directory
from fes_studio.models import AnalysisConfig, PlotConfig
from fes_studio.parser import load_fes_file, load_fes_text, reconfigure_dataset
from fes_studio.paths import EXPORT_ROOT, PROJECT_ROOT, default_postprocess_root, default_run_directory
from fes_studio.plotting import PALETTE_LABELS, PALETTE_OPTIONS, build_figures

DEFAULT_EXPORT_ROOT = EXPORT_ROOT


def _default_mfep_coarse_images(fine_images: int) -> int:
    if fine_images <= 7:
        return fine_images
    candidate = max(5, int(round(fine_images * 0.55)))
    if candidate % 2 == 0:
        candidate -= 1
    return min(candidate, fine_images - 2)


def _default_mfep_coarse_iterations(fine_iterations: int) -> int:
    estimated = int(round(max(fine_iterations * 0.45, 40.0) / 20.0) * 20)
    return max(40, estimated)


def _default_mfep_coarse_step_size(fine_step_size: float) -> float:
    return min(0.18, max(fine_step_size * 1.8, fine_step_size + 0.03))


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2rem;
            max-width: 1450px;
        }
        .hero {
            background:
                radial-gradient(circle at 10% 20%, rgba(255,255,255,0.58), transparent 32%),
                linear-gradient(135deg, #f1e7d8 0%, #d7ebe3 48%, #98bbb0 100%);
            border: 1px solid rgba(23, 48, 79, 0.12);
            border-radius: 28px;
            padding: 1.4rem 1.55rem;
            margin-bottom: 1.15rem;
            color: #17304f;
            box-shadow: 0 18px 40px rgba(16, 47, 68, 0.08);
        }
        .hero h1 {
            margin: 0;
            font-size: 2.25rem;
            letter-spacing: -0.03em;
        }
        .hero p {
            margin: 0.45rem 0 0 0;
            font-size: 1rem;
            max-width: 980px;
        }
        .source-card {
            background: rgba(255, 255, 255, 0.68);
            border: 1px solid rgba(23, 48, 79, 0.08);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.8rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.58);
            border: 1px solid rgba(23, 48, 79, 0.08);
            padding: 0.8rem 0.9rem;
            border-radius: 18px;
        }
        div[data-testid="stSidebar"] {
            background:
                radial-gradient(circle at top left, rgba(255,255,255,0.82), transparent 24%),
                linear-gradient(180deg, rgba(241,231,216,0.68), rgba(215,235,227,0.78));
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.55rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.5);
            border: 1px solid rgba(23, 48, 79, 0.08);
            padding-left: 0.95rem;
            padding-right: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def cached_prepare_run_dataset(
    run_dir: str,
    method_id: str,
    temperature: float,
    energy_unit: str,
    cv_fields: tuple[str, ...],
    sigma_text: str,
    bins: int,
    tools_root: str | None,
    source_signature: str,
    language: str,
):
    _ = source_signature
    return prepare_fes_from_run_directory(
        run_dir=run_dir,
        method_id=method_id,
        temperature=temperature,
        energy_unit=energy_unit,
        cv_fields=list(cv_fields),
        sigma_text=sigma_text or None,
        bins=bins,
        tools_root=tools_root,
        language=language,
    )


def load_source_dataset(demo_files: dict[str, Path], language: str):
    st.sidebar.subheader(tr(language, "data_source"))
    source = st.sidebar.radio(
        tr(language, "import_mode"),
        ["upload", "run_dir", "demo_1d", "demo_2d"],
        format_func=lambda value: {
            "upload": tr(language, "upload_fes_file"),
            "run_dir": tr(language, "run_dir_import"),
            "demo_1d": tr(language, "builtin_1d"),
            "demo_2d": tr(language, "builtin_2d"),
        }[value],
    )

    st.sidebar.subheader(tr(language, "general_physics"))
    temperature = st.sidebar.number_input(tr(language, "temperature"), min_value=1.0, max_value=2000.0, value=300.0, step=5.0)
    energy_unit = st.sidebar.selectbox(tr(language, "energy_unit"), ["kJ/mol", "kcal/mol", "kBT"], index=0)

    source_meta: dict[str, object] = {
        "source_mode": source,
        "temperature": float(temperature),
        "energy_unit": energy_unit,
    }

    if source == "upload":
        upload = st.sidebar.file_uploader(tr(language, "upload_prompt"), type=["dat", "txt", "colvar", "fes"])
        if upload is None:
            return None, source_meta
        dataset = load_fes_text(upload.getvalue(), source_name=upload.name)
        source_meta.update({"source_label": tr(language, "upload_fes_file"), "resolved_path": upload.name})
        return dataset, source_meta

    if source == "demo_1d":
        dataset = load_fes_file(demo_files["1d"])
        source_meta.update({"source_label": tr(language, "builtin_1d"), "resolved_path": str(demo_files["1d"])})
        return dataset, source_meta

    if source == "demo_2d":
        dataset = load_fes_file(demo_files["2d"])
        source_meta.update({"source_label": tr(language, "builtin_2d"), "resolved_path": str(demo_files["2d"])})
        return dataset, source_meta

    st.sidebar.subheader(tr(language, "directory_processing"))
    run_dir = st.sidebar.text_input(tr(language, "run_directory"), value=default_run_directory())
    tools_root = st.sidebar.text_input(tr(language, "tools_root"), value=str(default_postprocess_root()))
    run_info = detect_run_directory(run_dir, tools_root or None, language=language)
    source_meta.update(
        {
            "source_label": tr(language, "run_dir_import"),
            "resolved_path": run_dir,
            "run_info": run_info,
            "tools_root": tools_root,
        }
    )

    if not run_info.get("exists"):
        st.sidebar.warning(tr(language, "run_dir_missing"))
        return None, source_meta

    methods = run_info.get("methods", [])
    if not methods:
        st.sidebar.error(tr(language, "no_supported_outputs"))
        return None, source_meta

    method_ids = [item["id"] for item in methods]
    suggested = run_info.get("suggested_method")
    default_index = method_ids.index(suggested) if suggested in method_ids else 0
    method_id = st.sidebar.selectbox(
        tr(language, "processing_method"),
        options=method_ids,
        index=default_index,
        format_func=lambda value: next(item["label"] for item in methods if item["id"] == value),
    )

    st.sidebar.caption(tr(language, "engine_detected", engine=run_info.get("engine", "unknown")))
    visible_files = [name for name, path in run_info.get("files", {}).items() if path]
    if visible_files:
        st.sidebar.caption(tr(language, "files_detected", files=", ".join(visible_files)))
    if run_info.get("fes_files"):
        st.sidebar.caption(tr(language, "existing_fes", files=", ".join(path.name for path in run_info["fes_files"])))
    for warning in run_info.get("warnings", []):
        st.sidebar.warning(warning)

    method_default_cvs = run_info.get("method_default_cvs", {})
    cv_options = run_info.get("cv_fields") or []
    selected_cv = tuple((method_default_cvs.get(method_id) or cv_options or [])[:2])
    sigma_text = ""
    bins_default = 201 if len(selected_cv) <= 1 else 161

    with st.sidebar.expander(tr(language, "directory_diagnostics"), expanded=False):
        st.write(tr(language, "default_method", method=run_info.get("suggested_method") or "n/a"))
        st.write(tr(language, "plumed_available", value=tr(language, "yes") if run_info.get("plumed_available") else tr(language, "no")))
        if cv_options:
            st.write(tr(language, "detected_cvs", values="`, `".join(cv_options)))
        recommended = method_default_cvs.get(method_id) or []
        if recommended:
            st.write(tr(language, "current_method_default_cvs", values="`, `".join(recommended)))
        if run_info.get("colvar_fields"):
            st.write(tr(language, "colvar_fields", values="`, `".join(run_info["colvar_fields"])))
        if run_info.get("hills_fields"):
            st.write(tr(language, "hills_fields", values="`, `".join(run_info["hills_fields"])))
        if run_info.get("state_fields"):
            st.write(tr(language, "state_fields", values="`, `".join(run_info["state_fields"])))

    if method_id in {"metad-reweight", "opes-reweight"}:
        selected_cv = tuple(
            st.sidebar.multiselect(
                tr(language, "cvs_for_fes"),
                options=cv_options,
                default=list(selected_cv),
                max_selections=2,
            )
        )
        sigma_default = run_info["sigma_hints"]["metad"] if method_id == "metad-reweight" else run_info["sigma_hints"]["opes"]
        sigma_text = st.sidebar.text_input(tr(language, "sigma_bandwidth"), value=sigma_default or "")

    bins = st.sidebar.slider(tr(language, "fes_grid"), 81, 401, bins_default, step=20)

    try:
        with st.sidebar:
            with st.spinner(tr(language, "generating_fes")):
                result = cached_prepare_run_dataset(
                    run_dir=run_dir,
                    method_id=method_id,
                    temperature=float(temperature),
                    energy_unit=energy_unit,
                    cv_fields=selected_cv,
                    sigma_text=sigma_text,
                    bins=int(bins),
                    tools_root=tools_root or None,
                    source_signature=str(run_info.get("signature", "")),
                    language=language,
                )
        dataset = load_fes_file(result["fes_path"])
        source_meta.update(
            {
                "method_id": method_id,
                "method_label": result["method_label"],
                "generated_fes_path": str(result["fes_path"]),
                "generation_log": result["log"],
                "selected_cv": list(selected_cv),
                "sigma_text": sigma_text,
                "bins": bins,
            }
        )
        return dataset, source_meta
    except Exception as exc:
        st.sidebar.error(tr(language, "directory_processing_failed", error=exc))
        return None, source_meta


def sidebar_controls(dataset, temperature: float, energy_unit: str, language: str):
    st.sidebar.subheader(tr(language, "analysis_settings"))
    columns = dataset.available_columns

    default_cv = dataset.cv_columns or columns[:-1]
    cv_columns = st.sidebar.multiselect(tr(language, "cv_columns"), columns, default=default_cv)
    default_energy_index = columns.index(dataset.energy_column) if dataset.energy_column in columns else len(columns) - 1
    energy_column = st.sidebar.selectbox(tr(language, "free_energy_column"), columns, index=default_energy_index)

    error_choices = ["__none__"] + columns
    default_error = dataset.error_column if dataset.error_column in columns else "__none__"
    error_column = st.sidebar.selectbox(
        tr(language, "error_column_optional"),
        error_choices,
        index=error_choices.index(default_error),
        format_func=lambda value: tr(language, "none") if value == "__none__" else value,
    )

    top_minima = st.sidebar.slider(tr(language, "max_minima"), 1, 12, 6)
    interpolation_points = st.sidebar.slider(tr(language, "interp_grid_2d"), 100, 400, 240, step=20)
    smoothing_sigma = st.sidebar.slider(tr(language, "smoothing_strength"), 0.0, 3.0, 1.0, step=0.1)

    st.sidebar.subheader(tr(language, "mfep_settings"))
    mfep_two_stage = st.sidebar.checkbox(tr(language, "two_stage_mfep"), value=True)
    refine_label = tr(language, "refine") if mfep_two_stage else tr(language, "mfep")
    mfep_images = st.sidebar.slider(tr(language, "node_count", label=refine_label), 15, 91, 41, step=2)
    mfep_iterations = st.sidebar.slider(tr(language, "iterations", label=refine_label), 40, 400, 180, step=20)
    mfep_step_size = st.sidebar.slider(tr(language, "step_size", label=refine_label), 0.01, 0.12, 0.04, step=0.01)
    mfep_coarse_images = None
    mfep_coarse_iterations = None
    mfep_coarse_step_size = None
    if mfep_two_stage:
        coarse_images_max = max(5, mfep_images - 2)
        coarse_default_images = min(_default_mfep_coarse_images(mfep_images), coarse_images_max)
        mfep_coarse_images = st.sidebar.slider(tr(language, "coarse_node_count"), 5, coarse_images_max, coarse_default_images, step=2)
        mfep_coarse_iterations = st.sidebar.slider(
            tr(language, "coarse_iterations"),
            40,
            280,
            _default_mfep_coarse_iterations(mfep_iterations),
            step=20,
        )
        mfep_coarse_step_size = st.sidebar.slider(
            tr(language, "coarse_step_size"),
            min(max(mfep_step_size, 0.02), 0.12),
            0.20,
            _default_mfep_coarse_step_size(mfep_step_size),
            step=0.01,
        )
        st.sidebar.caption(tr(language, "coarse_caption"))
    mfep_spring_constant = st.sidebar.slider(tr(language, "mfep_spring_constant"), 0.00, 0.60, 0.18, step=0.02)
    primary_mfep_pair = st.sidebar.text_input(
        tr(language, "primary_mfep_pair"),
        value="",
        placeholder=tr(language, "primary_mfep_placeholder"),
    )

    st.sidebar.subheader(tr(language, "plot_settings"))
    dpi = st.sidebar.slider(tr(language, "export_dpi"), 200, 900, 450, step=50)
    contour_levels = st.sidebar.slider(tr(language, "contour_levels"), 12, 40, 24)
    palette = st.sidebar.selectbox(
        tr(language, "palette_theme"),
        PALETTE_OPTIONS,
        index=PALETTE_OPTIONS.index("lagoon"),
        format_func=lambda value: PALETTE_LABELS.get(value, value),
    )
    mfep_publication_mode = st.sidebar.checkbox(tr(language, "mfep_publication_mode"), value=True)
    annotate_barrier_table = st.sidebar.checkbox(tr(language, "annotate_barrier_table"), value=False)

    export_root = st.sidebar.text_input(tr(language, "export_root"), value=str(DEFAULT_EXPORT_ROOT))

    dataset = reconfigure_dataset(
        dataset,
        cv_columns=cv_columns,
        energy_column=energy_column,
        error_column=None if error_column == "__none__" else error_column,
    )
    analysis_config = AnalysisConfig(
        temperature=temperature,
        energy_unit=energy_unit,
        top_minima=top_minima,
        interpolation_points=interpolation_points,
        smoothing_sigma=smoothing_sigma,
        mfep_images=mfep_images,
        mfep_iterations=mfep_iterations,
        mfep_step_size=mfep_step_size,
        mfep_two_stage=mfep_two_stage,
        mfep_coarse_images=mfep_coarse_images,
        mfep_coarse_iterations=mfep_coarse_iterations,
        mfep_coarse_step_size=mfep_coarse_step_size,
        mfep_spring_constant=mfep_spring_constant,
        primary_mfep_pair=primary_mfep_pair.strip() or None,
    )
    plot_config = PlotConfig(
        dpi=dpi,
        contour_levels=contour_levels,
        palette=palette,
        language="en",
        report_language=language,
        mfep_publication_mode=mfep_publication_mode,
        annotate_barrier_table=annotate_barrier_table,
    )
    return dataset, analysis_config, plot_config, export_root


def show_summary(summary: dict[str, object], language: str) -> None:
    metric_cols = st.columns(4)
    metric_cols[0].metric(tr(language, "summary_dimension"), summary["dimension"])
    metric_cols[1].metric(tr(language, "summary_points"), summary["points"])
    metric_cols[2].metric(tr(language, "summary_minima"), summary.get("detected_minima", "n/a"))
    metric_cols[3].metric(tr(language, "summary_energy_unit"), summary["energy_unit"])

    summary_df = pd.DataFrame(
        [(summary_display_name(key, language), value) for key, value in summary.items()],
        columns=[tr(language, "item"), tr(language, "value")],
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


def show_source_summary(source_meta: dict[str, object], language: str) -> None:
    if not source_meta:
        return
    st.markdown(f"#### {tr(language, 'source_info')}")
    metric_cols = st.columns(4)
    metric_cols[0].metric(tr(language, "metric_source"), source_meta.get("source_label", "n/a"))
    metric_cols[1].metric(tr(language, "metric_temperature"), f"{source_meta.get('temperature', 'n/a')} K")
    metric_cols[2].metric(tr(language, "metric_unit"), source_meta.get("energy_unit", "n/a"))
    metric_cols[3].metric(tr(language, "metric_method"), source_meta.get("method_label", tr(language, "direct_read")))

    details = [
        (tr(language, "path"), source_meta.get("resolved_path", "n/a")),
        (tr(language, "generated_fes"), source_meta.get("generated_fes_path", "n/a")),
    ]
    run_info = source_meta.get("run_info")
    if run_info:
        details.append((tr(language, "engine"), run_info.get("engine", "n/a")))
    if source_meta.get("selected_cv"):
        details.append((tr(language, "cv_short"), ", ".join(source_meta["selected_cv"])))
    if source_meta.get("sigma_text"):
        details.append(("sigma", source_meta["sigma_text"]))
    if source_meta.get("bins"):
        details.append((tr(language, "generated_grid"), source_meta["bins"]))
    st.dataframe(pd.DataFrame(details, columns=[tr(language, "item"), tr(language, "value")]), use_container_width=True, hide_index=True)

    generation_log = source_meta.get("generation_log")
    if generation_log:
        with st.expander(tr(language, "generation_log"), expanded=False):
            st.code(generation_log, language="text")
    if run_info:
        detected_files = [name for name, path in run_info.get("files", {}).items() if path]
        available_methods = [item["label"] for item in run_info.get("methods", [])]
        diagnostics = []
        if detected_files:
            diagnostics.append((tr(language, "recognized_files"), ", ".join(detected_files)))
        if available_methods:
            diagnostics.append((tr(language, "available_methods"), " | ".join(available_methods)))
        if run_info.get("cv_fields"):
            diagnostics.append((tr(language, "directory_cv"), ", ".join(run_info["cv_fields"])))
        if diagnostics:
            with st.expander(tr(language, "directory_diagnostic_summary"), expanded=False):
                st.dataframe(pd.DataFrame(diagnostics, columns=[tr(language, "item"), tr(language, "value")]), use_container_width=True, hide_index=True)


def show_messages(analysis: dict[str, object], language: str) -> None:
    summary = analysis["summary"]
    status = summary.get("primary_mfep_request_status")
    requested = summary.get("primary_mfep_requested")
    if status == "fallback-auto" and requested:
        st.warning(tr(language, "primary_mfep_fallback", requested=requested, selected=summary.get("primary_mfep")))
    elif status == "requested" and requested:
        st.info(tr(language, "primary_mfep_requested", requested=requested))


def show_tables(analysis: dict[str, object], language: str) -> None:
    for name, table in analysis["tables"].items():
        st.markdown(f"#### {table_display_name(name, language)}")
        st.dataframe(table, use_container_width=True, hide_index=True)


def show_figures(figures: dict[str, object], language: str) -> None:
    for name, figure in figures.items():
        st.markdown(f"#### {figure_display_name(name, language)}")
        if hasattr(figure, "write_html"):
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.pyplot(figure, use_container_width=True, clear_figure=False)


def main() -> None:
    st.set_page_config(page_title="FES Studio", layout="wide", initial_sidebar_state="expanded")
    inject_style()
    demo_files = ensure_demo_files(PROJECT_ROOT)
    language = (
        "en"
        if st.sidebar.toggle(
            "English / 中文",
            value=False,
            help="切换界面和报告语言；图片始终保持英文 / Toggle UI and report language; figures always stay in English",
        )
        else "zh"
    )

    st.markdown(
        f"""
        <div class="hero">
          <h1>FES Studio</h1>
          <p>{tr(language, 'hero_body')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title(tr(language, "sidebar_title"))
    dataset, source_meta = load_source_dataset(demo_files, language)
    if dataset is None:
        st.info(tr(language, "dataset_hint"))
        st.markdown(tr(language, "project_location", path=str(PROJECT_ROOT)))
        return

    try:
        dataset, analysis_config, plot_config, export_root = sidebar_controls(
            dataset,
            temperature=float(source_meta.get("temperature", 300.0)),
            energy_unit=str(source_meta.get("energy_unit", "kJ/mol")),
            language=language,
        )
        analysis = run_analysis(dataset, analysis_config)
        figures = build_figures(analysis, plot_config)
    except Exception as exc:
        st.error(tr(language, "analysis_failed", error=exc))
        return

    overview_tab, result_tab, figure_tab, export_tab = st.tabs(
        [
            tr(language, "overview"),
            tr(language, "analysis_tables"),
            tr(language, "publication_plots"),
            tr(language, "export"),
        ]
    )

    with overview_tab:
        show_messages(analysis, language)
        show_source_summary(source_meta, language)
        show_summary(analysis["summary"], language)
        st.markdown(f"#### {tr(language, 'raw_data_preview')}")
        st.dataframe(dataset.frame.head(25), use_container_width=True, hide_index=True)

    with result_tab:
        show_tables(analysis, language)

    with figure_tab:
        show_figures(figures, language)

    with export_tab:
        st.markdown(f"#### {tr(language, 'export_full_bundle')}")
        st.code(
            f"{tr(language, 'source_file')}: {dataset.source_name}\n{tr(language, 'output_root')}: {export_root}",
            language="text",
        )
        if st.button(tr(language, "generate_analysis_bundle"), type="primary", use_container_width=True):
            output_dir = make_output_dir(export_root, dataset.source_name)
            result = export_bundle(dataset, analysis, plot_config, output_dir)
            st.success(tr(language, "analysis_bundle_written", path=result["output_dir"]))
            path_df = pd.DataFrame({tr(language, "output_files"): [str(path) for path in result["files"]]})
            st.dataframe(path_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
