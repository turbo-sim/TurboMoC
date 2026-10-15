"""Report the solved Fluent stage against the MoC design: figures, tables, Markdown, HTML and PDF.

Extraction reads the solved case and data in a headless Fluent session and
stores everything in ``output/report/data``; with ``plots_only`` the report is
rebuilt from those files without Fluent.
"""

import json
from pathlib import Path

import yaml

from . import analysis, document, extraction, figures


def main(config_file, plots_only=False):
    config_file = Path(config_file).resolve()
    workflow = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    output_dir = (config_file.parent / workflow["output_dir"]).resolve()
    report_dir = output_dir / "report"
    data_dir = report_dir / "data"
    if not plots_only:
        case_file = output_dir / "fluent" / workflow["fluent"]["case_filename"]
        if not case_file.with_name(case_file.name.replace(".cas.h5", ".dat.h5")).exists():
            raise FileNotFoundError(f"Solve and save the case first: no data file next to {case_file}")
        setup = analysis.read_json(output_dir / "fluent" / "setup_summary.json")
        radii = extraction.channel_radii(analysis.read_json(output_dir / "mesh" / "mesh_report.json"))
        print("1/3 Extract surfaces and Fluent reports (headless Fluent)")
        extraction.extract(case_file, setup, data_dir, workflow["fluent"], radii)
        extraction.residual_table(output_dir / "fluent", data_dir)
    print("2/3 Figures and tables")
    options = workflow["report"]
    results = analysis.compute(output_dir)
    paths, loading = figures.make_all(results, report_dir / "figures", options)
    (report_dir / "report.json").write_text(json.dumps({
        "stations": results.stations,
        "performance": {row[0]: {"cfd": row[2], "moc": row[3]} for row in results.performance},
        "torque_passage": results.torque_passage, "power_W": results.power, "fluent_reports": results.reports,
        "blade_loading": loading}, indent=2, default=float), encoding="utf-8")
    pdf_file = document.write_documents(document.report_blocks(results, paths, options), report_dir)
    print(f"3/3 Report written: {report_dir / 'report.md'}" + (f" and {pdf_file.name}" if pdf_file else
                                                               " (no Chrome or Edge found: PDF skipped)"))
    return report_dir
