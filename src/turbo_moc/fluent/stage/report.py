"""Report the solved Fluent stage against the MoC design: figures, tables, Markdown, HTML and PDF.

Extraction reads the solved case and data in a headless Fluent session and
stores everything in ``output/report/data``; with ``plots_only`` the report is
rebuilt from those files without Fluent.
"""

import json
from pathlib import Path

import yaml

from . import analysis, document, extraction, figures


def main(config_file, plots_only=False, case_filename=None):
    """Report one solved case; ``case_filename`` (e.g. a variant saved next to the main case) reports into
    ``report_<its stem>`` instead of ``report``."""
    config_file = Path(config_file).resolve()
    workflow = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    output_dir = (config_file.parent / workflow["output_dir"]).resolve()
    case_name = case_filename or workflow["fluent"]["case_filename"]
    report_dir = output_dir / ("report_" + case_name.replace(".cas.h5", "") if case_filename else "report")
    data_dir = report_dir / "data"
    if not plots_only:
        case_file = output_dir / "fluent" / case_name
        if not case_file.with_name(case_file.name.replace(".cas.h5", ".dat.h5")).exists():
            raise FileNotFoundError(f"Solve and save the case first: no data file next to {case_file}")
        setup = analysis.read_json(output_dir / "fluent" / "setup_summary.json")
        radii = extraction.channel_radii(analysis.read_json(output_dir / "mesh" / "mesh_report.json"))
        print("1/3 Extract surfaces and Fluent reports (headless Fluent)")
        extraction.extract(case_file, setup, data_dir, workflow["fluent"], radii)
        extraction.residual_table(output_dir / "fluent", data_dir)
    print("2/3 Figures and tables")
    options = workflow["report"]
    results = analysis.compute(output_dir, report_dir)
    results.starting = None
    if results.rotor_design is not None:
        from turbo_moc.core.rotor_starting import starting_check
        from turbo_moc.core.rotor_starting import kantrowitz_ratio, relative_isentrope
        results.starting = starting_check(results.rotor_design)
        rotor = results.rotor_design
        isentrope = relative_isentrope(rotor["fluid_name"], rotor["P0_rel"], rotor["T0_rel"],
                                       rotor.get("Q0_rel", float("nan")))
        results.starting["kantrowitz_ratio_cfd"] = kantrowitz_ratio(isentrope, results.rotor_in["M_rel"])
    paths, loading = figures.make_all(results, report_dir / "figures", options)
    (report_dir / "report.json").write_text(json.dumps({
        "stations": results.stations,
        "performance": {row[0]: {"cfd": row[2], "moc": row[3]} for row in results.performance},
        "starting": results.starting, "torque_passage": results.torque_passage, "power_W": results.power,
        "fluent_reports": results.reports,
        "blade_loading": loading}, indent=2, default=float), encoding="utf-8")
    pdf_file = document.write_documents(document.report_blocks(results, paths, options), report_dir)
    print(f"3/3 Report written: {report_dir / 'report.md'}" + (f" and {pdf_file.name}" if pdf_file else
                                                               " (no Chrome or Edge found: PDF skipped)"))
    return report_dir
