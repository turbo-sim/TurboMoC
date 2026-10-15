"""Report content as ordered blocks, rendered to Markdown and to a print-ready HTML page and PDF.

A block is a tuple whose first item is its kind: ``title``, ``kpis``, ``h2``,
``bullets``, ``note``, ``table``, ``figure`` or ``columns``. Tables and figures
are numbered in rendering order.
"""

import html
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .extraction import ROWS

TEXT_WIDTH_MM = 182.0


def number(value, spec):
    return "–" if value is None else format(value, spec)


def minus(text):
    return re.sub(r"(?<![\w.])-(?=\d)", "\u2212", text)


def performance_table(performance):
    rows = []
    for label, unit, cfd, moc, kind, spec in performance:
        if cfd is None or moc is None or kind is None:
            delta = "–"
        elif kind == "relative":
            delta = f"{100 * (cfd - moc) / moc:+.1f} %"
        else:
            delta = format(cfd - moc, "+" + spec)
        rows.append([label, unit, number(cfd, spec), number(moc, spec), delta])
    return rows


def station_table(results):
    rows = []
    for name, frame, values, mach, speed, angle in results.station_rows:
        reference = results.design_stations.get(f"{name}|{frame}", {})
        rows.append([name, frame, number(values["p"] / 1e3, ".2f"),
                     number(reference.get("p") and reference["p"] / 1e3, ".2f"),
                     number(values[mach], ".3f"), number(reference.get("M"), ".3f"),
                     number(values[angle], ".1f"), number(reference.get("angle"), ".1f"),
                     number(values[speed], ".1f")])
    return rows


def final_residuals(results):
    if results.residuals is None:
        return None, []
    equations = [column for column in results.residuals.columns if column not in ("iter", "time/iter")]
    return results.residuals.iloc[-1], equations


def highlights(results, lookup):
    design_exit = results.design_stations["Stator exit (1)|absolute"]
    design_rotor = results.design_stations["Rotor exit (2)|relative"]
    exit_, outlet, flows = results.exit_, results.outlet, results.flows
    work, euler = lookup["Specific work (torque)"][2], lookup["Specific work (Euler)"][2]
    flow, flow_moc = lookup["Mass flow rate, full annulus"][2:4]
    eta, eta_moc = lookup["Efficiency, total-to-static"][2:4]
    items = [
        f"Mass flow {flow:.3f} kg/s, {100 * (flow - flow_moc) / flow_moc:+.1f} % against the MoC annulus estimate; "
        f"the stator is choked and sets it.",
        f"Stator exit: M = {exit_['M']:.3f} (design {design_exit['M']:.3f}) and α = {exit_['alpha']:.1f}° "
        f"(design {design_exit['angle']:.1f}°), at {exit_['p'] / 1e3:.1f} kPa against {design_exit['p'] / 1e3:.1f} kPa "
        f"designed ({100 * (exit_['p'] - design_exit['p']) / design_exit['p']:+.1f} %).",
        f"Rotor exit relative angle {outlet['beta']:.1f}° against {design_rotor['angle']:.1f}° designed: "
        f"{abs(outlet['beta'] - design_rotor['angle']):.1f}° less turning, hence the lower work.",
        f"Specific work {work:.3f} kJ/kg from the rotor torque and {euler:.3f} kJ/kg from the Euler equation "
        f"({100 * abs(work - euler) / work:.1f} % apart); total-to-static efficiency {eta:.3f} against {eta_moc:.3f}.",
    ]
    final, equations = final_residuals(results)
    if final is not None:
        worst = max(equations, key=lambda equation: final[equation])
        items.append(f"{int(final['iter'])} iterations; largest final residual {final[worst]:.1e} ({worst}); "
                     f"mass imbalance {100 * (flows['inlet'] - flows['outlet']) / flows['inlet']:+.2f} % inlet to "
                     f"outlet, {100 * (flows['stator_exit'] - flows['rotor_inlet']) / flows['stator_exit']:+.3f} % "
                     f"across the mixing plane.")
    return items


def mesh_table(results):
    rows = []
    for row in ROWS:
        summary = results.mesh_report["rows"][row]["meshes"]["3d"]["summary"]
        cells, quality = summary["cells"], summary["quality"]
        rows.append([row.split("_")[0].capitalize(), f"{cells.get('hexahedra', 0):,}", f"{cells.get('prisms', 0):,}",
                     number(quality["min_determinant_jacobian"], ".2e"), number(quality["min_sicn"], ".3f"),
                     f"{np.mean(results.y_plus[row]):.0f} / {np.max(results.y_plus[row]):.0f}"])
    return rows


def setup_table(results):
    methods = results.reports["methods"]
    schemes = methods["discretization_scheme"]
    flows, point = results.flows, results.point
    rows = [
        ["Solver", f"Pressure-based {methods['p_v_coupling']['flow_scheme'].lower()}, steady (pseudo-transient)"],
        ["Turbulence", "SST k-ω" if results.reports["viscous"] == "k-omega" else results.reports["viscous"]],
        ["Discretization", f"{schemes['mom'].replace('-', ' ')} (momentum, density, k, ω); pressure "
                           f"{schemes['pressure']}; {methods['gradient_scheme'].replace('-', ' ')} gradients"],
        ["Rotor speed", f"{abs(point['rpm']):.0f} rpm ({abs(results.omega):.1f} rad/s)"],
        ["Boundary conditions", f"P₀ = {results.P0 / 1e3:.1f} kPa at the inlet; "
                                f"p = {point['rotor_exit_pressure'] / 1e3:.2f} kPa at the outlet"],
        ["Mass imbalance", f"{100 * (flows['inlet'] - flows['outlet']) / flows['inlet']:+.3f} % (inlet → outlet), "
                           f"{100 * (flows['stator_exit'] - flows['rotor_inlet']) / flows['stator_exit']:+.3f} % "
                           f"(mixing plane)"],
    ]
    final, equations = final_residuals(results)
    if final is not None:
        rows.append(["Iterations", f"{int(final['iter'])}"])
        rows.append(["Final residuals", ", ".join(f"{equation} {final[equation]:.0e}" for equation in equations)])
    return rows


ASSUMPTIONS = ("Assumptions: steady RANS with the rotor in a moving reference frame; circumferentially averaged mixing "
               "plane between the rows; one passage per row with rotational periodicity; barotropic equilibrium "
               "closure along the inlet isentrope (energy equation off, adiabatic walls); no tip clearance; rotor "
               "shroud stationary.")


def report_blocks(results, figures, options):
    """Ordered document content for the stage report."""
    lookup = {row[0]: row for row in results.performance}
    point, state, blades = results.point, results.state, results.blades
    operating = [("Fluid", state["fluid_name"]), ("P₀", f"{results.P0 / 1e3:.1f} kPa"),
                 ("Inlet quality", f"{state['Q0']:.3f}"),
                 ("Exit pressure", f"{point['rotor_exit_pressure'] / 1e3:.2f} kPa"),
                 ("Speed", f"{abs(point['rpm']):.0f} rpm"),
                 ("Blades", f"{blades['stator_1']} stator / {blades['rotor_1']} rotor"),
                 ("Row gap", f"{results.gap:g} mm")]
    kpis = [(label, unit, lookup[key][2], lookup[key][3], spec) for label, unit, key, spec in (
        ("Mass flow", "kg/s", "Mass flow rate, full annulus", ".3f"), ("Power", "kW", "Power", ".2f"),
        ("Specific work", "kJ/kg", "Specific work (torque)", ".3f"),
        ("Total-to-static efficiency", "", "Efficiency, total-to-static", ".3f"))]
    quality = results.reports["mesh_quality"]
    residual_figure = [("figure", [figures["residuals"]], "Residual history.")] if "residuals" in figures else []
    return [
        ("title", "Supersonic axial stage: CFD assessment of the MoC design",
         "Three-dimensional RANS of the stator–rotor stage (Fluent, mixing plane) against the "
         "method-of-characteristics design", operating),
        ("kpis", kpis),
        ("h2", "Highlights"), ("bullets", highlights(results, lookup)),
        ("h2", "Stage performance"),
        ("table", "Stage performance from CFD and from the MoC design; Δ is relative for dimensional quantities and "
                  "absolute otherwise. Isentropic drops use H(p) = ∫dp/ρ of the barotropic closure.",
         ["Quantity", "Unit", "CFD", "MoC", "Δ"], None, performance_table(results.performance), "numeric-3"),
        ("table", "Mass-weighted averages at the row interfaces. Angles from the axial direction, positive along the "
                  "stator turning; the stator exit is evaluated on the stator side of the mixing plane.",
         ["Station", "Frame", "CFD", "MoC", "CFD", "MoC", "CFD", "MoC", "CFD"],
         [("", 2), ("p [kPa]", 2), ("Mach [-]", 2), ("Angle [deg]", 2), ("V [m/s]", 1)], station_table(results),
         "numeric-2"),
        ("h2", "Geometry and mesh"),
        ("figure", [figures["geometry"]], f"Stage geometry. (a) Blading over {options['passages_3d']} passages per "
                                          "row, rendered from the Fluent surfaces; (b) meridional channel with the "
                                          "row inlets, the mixing plane, the outlet (dashed) and the 25/50/75 % "
                                          "spans."),
        ("table", "Volume meshes (app mesher) and blade y⁺ from the solution. Fluent: minimum orthogonal quality "
                  f"{quality['min_orthogonal_quality']:.3f}, maximum aspect ratio {quality['max_aspect_ratio']:.0f}.",
         ["Row", "Hexahedra", "Prisms", "Min. det. Jacobian", "Min. SICN", "Blade y⁺ mean / max"], None,
         mesh_table(results), "numeric-1"),
        ("h2", "Simulation setup and convergence"),
        ("columns", [("table", "Solver settings and convergence.", ["Item", "Setting"], None, setup_table(results),
                      "keyvalue"), ("note", ASSUMPTIONS)], residual_figure, 0.58),
        ("h2", "Flow field"),
        ("figure", [figures["midspan"]], f"Midspan (r = {results.r_mid * 1e3:.1f} mm) on the unwrapped cylinder. "
                                         "White line: M = 1; dashed line: mixing plane."),
        ("figure", figures["schlieren"], "Numerical schlieren |∇ρ| at midspan, logarithmic grey scale between the 5th "
                                         f"and {options['schlieren_percentile']}th percentiles."),
        ("h2", "Blade loading and stator exit"),
        ("figure", [figures["loading"]], "Blade static pressure and isentropic Mach number at 25, 50 and 75 % span. "
                                         "Stator from P₀; rotor relative, from the rotor-inlet rothalpy at the local "
                                         "radius."),
        ("figure", [figures["pitchwise"]], "Stator exit, stator side of the mixing plane: pitchwise distributions at "
                                           "three spans against the mass average (Table 2) and the MoC design."),
    ]


def render_markdown(blocks, counters=None):
    counters = {"table": 0, "figure": 0} if counters is None else counters
    lines = []
    for block in blocks:
        kind = block[0]
        if kind == "title":
            lines += [f"# {block[1]}", "", f"*{block[2]}*", "",
                      " · ".join(f"**{key}** {value}" for key, value in block[3]), ""]
        elif kind == "kpis":
            lines += ["| | CFD | MoC |", "| --- | --- | --- |"]
            lines += [f"| {label} {f'[{unit}]' if unit else ''} | {number(cfd, spec)} | {number(moc, spec)} |"
                      for label, unit, cfd, moc, spec in block[1]] + [""]
        elif kind == "h2":
            lines += [f"## {block[1]}", ""]
        elif kind == "bullets":
            lines += [f"- {item}" for item in block[1]] + [""]
        elif kind == "note":
            lines += [f"> {block[1]}", ""]
        elif kind == "table":
            counters["table"] += 1
            _, caption, headers, groups, rows, _style = block
            if groups:
                expanded = [label for label, span in groups for _ in range(span)]
                headers = [f"{group} {header}".strip() if group else header for group, header in zip(expanded, headers)]
            lines += [f"**Table {counters['table']}.** {caption}", "", "| " + " | ".join(headers) + " |",
                      "|" + " --- |" * len(headers)] + ["| " + " | ".join(row) + " |" for row in rows] + [""]
        elif kind == "figure":
            counters["figure"] += 1
            lines += [f"![Figure {counters['figure']}]({path})" for path in block[1]]
            lines += ["", f"**Figure {counters['figure']}.** {block[2]}", ""]
        elif kind == "columns":
            lines += render_markdown(block[1] + block[2], counters).splitlines() + [""]
    return "\n".join(lines)


STYLE = """
@page { size: A4; margin: 15mm 14mm 16mm;
        @bottom-center { content: counter(page) " / " counter(pages); font: 8pt "Times New Roman", serif;
                         color: #6b7b82; } }
* { box-sizing: border-box; }
body { margin: 0; font: 10pt/1.38 "Times New Roman", Times, serif; color: #1c2a30; }
header { border-bottom: 1.5pt solid #233c46; padding-bottom: 7pt; margin-bottom: 9pt; }
header h1 { font-size: 18pt; margin: 0 0 3pt; color: #1f3741; letter-spacing: -0.2pt; }
header .subtitle { font-style: italic; color: #4a5b62; margin: 0 0 6pt; }
.chips { display: flex; flex-wrap: wrap; gap: 3pt 12pt; font-size: 8.6pt; color: #33454c; }
.chips b { color: #1f3741; font-weight: 600; margin-right: 2pt; }
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 7pt; margin: 2pt 0 4pt; }
.kpi { border: 0.6pt solid #c9d2d6; border-top: 2.2pt solid #233c46; padding: 5pt 8pt 6pt; background: #f7f9f9; }
.kpi .label { font-size: 8.6pt; color: #4a5b62; font-variant: small-caps; letter-spacing: 0.3pt; }
.kpi .value { font-size: 19pt; color: #1f3741; line-height: 1.15; }
.kpi .value small { font-size: 9pt; color: #4a5b62; margin-left: 3pt; }
.kpi .reference { font-size: 8.4pt; color: #4a5b62; }
.kpi .delta { color: #1f3741; font-weight: 600; margin-left: 5pt; }
h2 { font-size: 12.5pt; color: #1f3741; margin: 12pt 0 5pt; padding-bottom: 2pt;
     border-bottom: 0.6pt solid #c9d2d6; break-after: avoid; }
ul { margin: 2pt 0 4pt; padding-left: 14pt; }
li { margin: 1.5pt 0; padding-left: 2pt; }
li::marker { content: "•  "; color: #233c46; }
figure, .table { margin: 6pt 0 9pt; break-inside: avoid; }
figure { text-align: center; }
figure .row { display: flex; justify-content: center; gap: 6pt; align-items: flex-end; }
figure img { display: block; margin: 0 auto; }
figcaption, .table .caption { font-size: 8.8pt; color: #33454c; text-align: left; line-height: 1.3; }
figcaption { margin-top: 4pt; }
.table .caption { margin-bottom: 3pt; }
figcaption b, .table .caption b { color: #1f3741; }
table { border-collapse: separate; border-spacing: 0; width: 100%; font-size: 8.6pt; font-variant-numeric: tabular-nums; }
thead th { border-top: 1pt solid #233c46; border-bottom: 0.6pt solid #233c46; font-weight: 600; color: #1f3741;
           padding: 2.5pt 5pt; text-align: left; }
thead tr.group th { border-bottom: 0.4pt solid #9fb0b6; border-top: 1pt solid #233c46; text-align: center; }
thead tr.group + tr th { border-top: none; }
tbody td { padding: 2.2pt 5pt; border-bottom: 0.3pt solid #e1e7e9; vertical-align: top; }
tbody tr:last-child td { border-bottom: 1pt solid #233c46; }
table.numeric-1 td:nth-child(n+2), table.numeric-1 th:nth-child(n+2),
table.numeric-2 td:nth-child(n+3), table.numeric-2 tr:not(.group) th:nth-child(n+3),
table.numeric-3 td:nth-child(n+3), table.numeric-3 th:nth-child(n+3) { text-align: right; }
table.keyvalue td:first-child { color: #4a5b62; white-space: nowrap; width: 28%; }
.columns { display: flex; gap: 12pt; align-items: flex-start; break-inside: avoid; }
.columns > div > :first-child { margin-top: 0; }
.columns figure { margin-top: 0; }
.note { font-size: 8.8pt; color: #33454c; background: #f3f6f7; border-left: 2pt solid #9fb0b6;
        padding: 4pt 8pt; margin: 4pt 0 6pt; break-inside: avoid; }
"""


def render_html(blocks, report_dir, text_width_mm=TEXT_WIDTH_MM):
    from PIL import Image
    escape = html.escape
    counters = {"table": 0, "figure": 0}

    def image_size(path):
        with Image.open(report_dir / path) as image:
            width, height = image.size
            dpi = image.info.get("dpi", (200, 200))[0] or 200
        return width, height, width / dpi * 25.4

    def render(block, container_mm=text_width_mm):
        kind = block[0]
        if kind == "title":
            chips = "".join(f"<span><b>{escape(key)}</b>{escape(value)}</span>" for key, value in block[3])
            return (f"<header><h1>{escape(block[1])}</h1><p class='subtitle'>{escape(block[2])}</p>"
                    f"<div class='chips'>{chips}</div></header>")
        if kind == "kpis":
            cards = []
            for label, unit, cfd, moc, spec in block[1]:
                delta = f"<span class='delta'>{minus(f'{100 * (cfd - moc) / moc:+.1f}')} %</span>" if moc else ""
                cards.append(f"<div class='kpi'><div class='label'>{escape(label)}</div>"
                             f"<div class='value'>{minus(format(cfd, spec))}<small>{escape(unit)}</small></div>"
                             f"<div class='reference'>MoC design {number(moc, spec)}{delta}</div></div>")
            return f"<div class='kpis'>{''.join(cards)}</div>"
        if kind == "h2":
            return f"<h2>{escape(block[1])}</h2>"
        if kind == "bullets":
            return "<ul>" + "".join(f"<li>{minus(escape(item))}</li>" for item in block[1]) + "</ul>"
        if kind == "note":
            return f"<div class='note'>{escape(block[1])}</div>"
        if kind == "table":
            counters["table"] += 1
            _, caption, headers, groups, rows, style = block
            head = ""
            if groups:
                head += "<tr class='group'>" + "".join(f"<th colspan='{span}'>{escape(label)}</th>"
                                                       for label, span in groups) + "</tr>"
            head += "<tr>" + "".join(f"<th>{escape(header)}</th>" for header in headers) + "</tr>"
            body = "".join("<tr>" + "".join(f"<td>{minus(escape(cell))}</td>" for cell in row) + "</tr>"
                           for row in rows)
            return (f"<div class='table'><div class='caption'><b>Table {counters['table']}.</b> {escape(caption)}"
                    f"</div><table class='{style}'><thead>{head}</thead><tbody>{body}</tbody></table></div>")
        if kind == "figure":
            counters["figure"] += 1
            sizes = [image_size(path) for path in block[1]]
            if len(sizes) == 1:
                width = min(100.0, 100 * sizes[0][2] / container_mm)
                images = f"<img src='{block[1][0]}' style='width:{width:.1f}%'>"
            else:
                ratios = [width / height for width, height, _ in sizes]
                images = "<div class='row'>" + "".join(
                    f"<img src='{path}' style='width:calc({100 * ratio / sum(ratios):.2f}% - 4pt)'>"
                    for path, ratio in zip(block[1], ratios)) + "</div>"
            return (f"<figure>{images}<figcaption><b>Figure {counters['figure']}.</b> {escape(block[2])}"
                    f"</figcaption></figure>")
        if kind == "columns":
            left = "".join(render(item, container_mm * block[3]) for item in block[1])
            right = "".join(render(item, container_mm * (1 - block[3]) - 12 * 0.353) for item in block[2])
            return (f"<div class='columns'><div style='flex:{block[3]}'>{left}</div>"
                    f"<div style='flex:{1 - block[3]}'>{right}</div></div>")
        raise ValueError(f"Unknown report block {kind!r}")

    body = "".join(render(block) for block in blocks)
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{escape(blocks[0][1])}</title>"
            f"<style>{STYLE}</style></head><body>{body}</body></html>")


def browser():
    candidates = [shutil.which(name) for name in ("msedge", "chrome", "google-chrome", "chromium")]
    candidates += [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                   r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
    return next((path for path in candidates if path and Path(path).exists()), None)


def write_pdf(html_file):
    """Print the HTML page to PDF with a headless Chromium browser; None when no browser is installed."""
    executable = browser()
    if executable is None:
        return None
    pdf_file = html_file.with_suffix(".pdf")
    subprocess.run([executable, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_file}", html_file.as_uri()], check=True, capture_output=True, timeout=120)
    return pdf_file


def write_documents(blocks, report_dir):
    (report_dir / "report.md").write_text(render_markdown(blocks), encoding="utf-8")
    html_file = report_dir / "report.html"
    html_file.write_text(render_html(blocks, report_dir), encoding="utf-8")
    return write_pdf(html_file)
