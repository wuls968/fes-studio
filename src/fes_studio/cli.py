from __future__ import annotations

import argparse

from .analysis import run_analysis
from .demo import ensure_demo_files
from .export import export_bundle, make_output_dir
from .i18n import LANGUAGE_OPTIONS
from .importers import detect_run_directory, prepare_fes_from_run_directory
from .launcher import main as launcher_main
from .models import AnalysisConfig, PlotConfig
from .parser import load_fes_file, reconfigure_dataset
from .paths import PROJECT_ROOT, default_postprocess_root
from .plotting import PALETTE_OPTIONS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze enhanced-sampling free-energy surfaces.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze a FES file and export a full result bundle.")
    analyze.add_argument("input", help="Path to a fes.dat-like file.")
    analyze.add_argument("--output-dir", default="exports", help="Directory for analysis bundles.")
    analyze.add_argument("--temperature", type=float, default=300.0)
    analyze.add_argument("--energy-unit", choices=["kJ/mol", "kcal/mol", "kBT"], default="kJ/mol")
    analyze.add_argument("--language", choices=LANGUAGE_OPTIONS, default="zh")
    analyze.add_argument("--top-minima", type=int, default=6)
    analyze.add_argument("--interpolation-points", type=int, default=240)
    analyze.add_argument("--smoothing-sigma", type=float, default=1.0)
    analyze.add_argument("--mfep-images", type=int, default=41)
    analyze.add_argument("--mfep-iterations", type=int, default=180)
    analyze.add_argument("--mfep-step-size", type=float, default=0.04)
    analyze.add_argument("--single-stage-mfep", action="store_true", help="Disable the coarse-to-fine MFEP schedule.")
    analyze.add_argument("--mfep-coarse-images", type=int, default=None)
    analyze.add_argument("--mfep-coarse-iterations", type=int, default=None)
    analyze.add_argument("--mfep-coarse-step-size", type=float, default=None)
    analyze.add_argument("--mfep-spring-constant", type=float, default=0.18)
    analyze.add_argument("--primary-mfep", default=None, help="Preferred primary MFEP pair, e.g. 1-3 or S1-S3.")
    analyze.add_argument("--cv-columns", nargs="*", default=None)
    analyze.add_argument("--energy-column", default=None)
    analyze.add_argument("--error-column", default=None)
    analyze.add_argument("--dpi", type=int, default=450)
    analyze.add_argument("--palette", choices=PALETTE_OPTIONS, default="lagoon")
    analyze.add_argument("--mfep-publication-mode", action="store_true", help="Export an additional publication-style MFEP figure.")
    analyze.add_argument("--annotate-barrier-table", action="store_true", help="Overlay a barrier table on MFEP 2D figures.")

    import_run = subparsers.add_parser("import-run", help="Prepare a FES file directly from a METAD/OPES run directory.")
    import_run.add_argument("run_dir", help="Path to the simulation output directory.")
    import_run.add_argument("--method", default=None, help="Import method id. Defaults to the detected recommended method.")
    import_run.add_argument(
        "--tools-root",
        default=None,
        help=f"Directory containing the opes-metad helper scripts. Defaults to auto-detection (current default: {default_postprocess_root()}).",
    )
    import_run.add_argument("--temperature", type=float, default=300.0)
    import_run.add_argument("--energy-unit", choices=["kJ/mol", "kcal/mol", "kBT"], default="kJ/mol")
    import_run.add_argument("--language", choices=LANGUAGE_OPTIONS, default="zh")
    import_run.add_argument("--cv-fields", nargs="*", default=None, help="Optional CV names used for reweighting.")
    import_run.add_argument("--sigma", default=None, help="Optional sigma/bandwidth override for reweighting.")
    import_run.add_argument("--bins", type=int, default=161, help="Grid size used when generating the FES.")

    subparsers.add_parser("demo", help="Generate bundled demo FES files.")
    launch = subparsers.add_parser("launch", help="Start the Streamlit interface through the cross-platform launcher.")
    launch.add_argument("--port", type=int, default=None)
    launch.add_argument("--no-browser", action="store_true")
    launch.add_argument("--browser-timeout", type=float, default=30.0)
    subparsers.add_parser("preflight", help="Check the local GUI/runtime environment.")
    subparsers.add_parser("repair-env", help="Run platform-specific environment repair checks.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "launch":
        launch_args = ["launch", f"--browser-timeout={args.browser_timeout}"]
        if args.port:
            launch_args.append(f"--port={args.port}")
        if args.no_browser:
            launch_args.append("--no-browser")
        raise SystemExit(launcher_main(launch_args))
    if args.command == "preflight":
        raise SystemExit(launcher_main(["preflight"]))
    if args.command == "repair-env":
        raise SystemExit(launcher_main(["repair"]))
    if args.command == "demo":
        demo_files = ensure_demo_files(PROJECT_ROOT)
        for name, path in demo_files.items():
            print(f"{name}: {path}")
        return
    if args.command == "import-run":
        info = detect_run_directory(args.run_dir, args.tools_root, language=args.language)
        if not info.get("exists"):
            raise SystemExit(f"Run directory does not exist: {args.run_dir}")
        method_id = args.method or info.get("suggested_method")
        if not method_id:
            raise SystemExit("No supported import method was detected in the run directory.")
        result = prepare_fes_from_run_directory(
            run_dir=args.run_dir,
            method_id=method_id,
            temperature=args.temperature,
            energy_unit=args.energy_unit,
            cv_fields=args.cv_fields,
            sigma_text=args.sigma,
            bins=args.bins,
            tools_root=args.tools_root,
            language=args.language,
        )
        print(f"method: {result['method_label']}")
        print(f"fes: {result['fes_path']}")
        if result.get("log"):
            print(result["log"])
        return

    dataset = load_fes_file(args.input)
    if args.cv_columns or args.energy_column or args.error_column:
        dataset = reconfigure_dataset(
            dataset,
            cv_columns=args.cv_columns or dataset.cv_columns,
            energy_column=args.energy_column or dataset.energy_column,
            error_column=args.error_column or dataset.error_column,
        )

    analysis_config = AnalysisConfig(
        temperature=args.temperature,
        energy_unit=args.energy_unit,
        top_minima=args.top_minima,
        interpolation_points=args.interpolation_points,
        smoothing_sigma=args.smoothing_sigma,
        mfep_images=args.mfep_images,
        mfep_iterations=args.mfep_iterations,
        mfep_step_size=args.mfep_step_size,
        mfep_two_stage=not args.single_stage_mfep,
        mfep_coarse_images=args.mfep_coarse_images,
        mfep_coarse_iterations=args.mfep_coarse_iterations,
        mfep_coarse_step_size=args.mfep_coarse_step_size,
        mfep_spring_constant=args.mfep_spring_constant,
        primary_mfep_pair=args.primary_mfep,
    )
    plot_config = PlotConfig(
        dpi=args.dpi,
        palette=args.palette,
        language="en",
        report_language=args.language,
        mfep_publication_mode=args.mfep_publication_mode,
        annotate_barrier_table=args.annotate_barrier_table,
    )
    analysis = run_analysis(dataset, analysis_config)
    output_dir = make_output_dir(args.output_dir, dataset.source_name)
    export_bundle(dataset, analysis, plot_config, output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
