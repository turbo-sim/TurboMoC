"""Meanline design statistics from the stator MoC and rotor vortex solutions."""
from math import atan2, cos, degrees, hypot, isfinite, pi, radians, sin, sqrt

import jaxprop as jxp
import plotly.graph_objects as go
from dash import dcc, html
from scipy.optimize import brentq

from turbo_moc.app import plots
from turbo_moc.app import workflow as flow
from turbo_moc.app.rotor_speed import resolve_speed
from turbo_moc.configuration import project as projects
from turbo_moc.configuration.fluid_inputs import inlet_values
from turbo_moc.configuration.rotor import inlet_radius_mm

ASSUMPTIONS = [
    "Meanline (midspan) estimates: the MoC and vortex solutions are two-dimensional sections.",
    "Isentropic flow in both rows; no profile, endwall, trailing-edge or mixing losses.",
    "Stator exit = rotor inlet: the static state at the end of the MoC wall is transferred without an interspace model.",
    "Angles are measured from the axial (meridional) direction; positive tangential components follow the stator turning.",
    "Rotor exit: the relative stagnation enthalpy follows rothalpy conservation between the mean inlet and outlet radii.",
    "Mass flow: uniform stator exit velocity over the full annulus, without blade blockage (annular machines only).",
]


NOTATION = [
    "V absolute velocity, W relative velocity, U blade speed; V = W + U.",
    "Subscripts m and t: meridional and tangential components. Stations: 0 stator inlet, 1 stator exit / rotor inlet, 2 rotor exit.",
    "Absolute frame: stator and stationary observer. Relative frame: rotating with the rotor at the mean radius.",
    "Efficiency, total-to-static: Euler work / (h₀₀ − h₂ at the inlet entropy); degree of reaction: (h₁ − h₂)/(h₀₀ − h₂).",
]


def state(fluid, kind, a, b):
    result = fluid.get_state(kind, a, b)
    return {"p": float(result.p), "T": float(result.T), "rho": float(result.rho), "h": float(result.h),
            "s": float(result.s), "a": float(result.a)}


def static_from_velocity(fluid, h0, s, mach):
    def residual(h):
        local = fluid.get_state(jxp.HmassSmass_INPUTS, h, s)
        return sqrt(max(2 * (h0 - h), 0.0)) - mach * float(local.a)

    low = h0 - 1.0
    while residual(low) < 0:
        low = h0 - 2 * (h0 - low)
    high = h0 - 1e-6 * max(1.0, abs(h0))
    return state(fluid, jxp.HmassSmass_INPUTS, brentq(residual, low, high, xtol=1e-6), s)


def station(name, frame, values, velocity=None, meridional=None, tangential=None):
    angle = degrees(atan2(tangential, meridional)) if velocity else None
    mach = velocity / values["a"] if velocity is not None else 0.0
    return {"station": name, "frame": frame, **values, "M": mach, "velocity": velocity,
            "meridional": meridional, "tangential": tangential, "angle": angle}


def rotor_radii(config):
    channel = config["meridional_channel"]
    inlet = inlet_radius_mm(config)
    if inlet is None:
        return None, None
    outlet = float(channel["outlet_radius"]) if channel["type"] == "radial" else inlet
    return inlet / 1000, outlet / 1000


def stator_statistics(project, row, config, data):
    inputs = config["method_of_characteristics"]
    fluid = jxp.Fluid(project["machine"]["fluid_name"])
    values = inlet_values(inputs)
    total = state(fluid, jxp.INPUT_PAIRS[values["inlet_property_pair"]],
                  values["inlet_property_1"], values["inlet_property_2"])
    axis = data["axis"]
    throat = state(fluid, jxp.DmassT_INPUTS, float(axis["rho"][0]), float(axis["T"][0]))
    wall = data["wall_final"]
    exit_state = state(fluid, jxp.DmassT_INPUTS, float(wall["rho"][-1]), float(wall["T"][-1]))
    alpha = config["blade_geometry"]["metal_angle_out"]
    speed = float(wall["M"][-1]) * exit_state["a"]
    stations = [
        station("Stator inlet (0)", "absolute, stagnation", total),
        station("Stator throat (sonic line)", "absolute", throat, throat["a"] * float(axis["M"][0]),
                throat["a"] * float(axis["M"][0]), 0.0),
        station("Stator exit (1)", "absolute", exit_state, speed, speed * cos(radians(alpha)),
                speed * sin(radians(alpha))),
    ]
    drop = total["h"] - exit_state["h"]
    performance = [("Isentropic enthalpy drop, total-to-static [kJ/kg]", drop / 1e3),
                   ("Pressure ratio p₀/p₁ [-]", total["p"] / exit_state["p"]),
                   ("Exit Mach number [-]", float(wall["M"][-1])),
                   ("Exit velocity [m/s]", speed), ("Exit flow angle [deg]", alpha),
                   ("MoC exit Mach, design target [-]", data["design_Noz_Mach"])]
    channel = config["meridional_channel"]
    area = None
    if channel["type"] == "axial":
        area = pi * ((channel["hub_radius"] + channel["blade_height"]) ** 2 - channel["hub_radius"] ** 2) * 1e-6
    elif channel["type"] == "radial":
        area = 2 * pi * channel["outlet_radius"] * channel["blade_height"] * 1e-6
    mass_flow = exit_state["rho"] * speed * cos(radians(alpha)) * area if area else None
    if mass_flow:
        performance.append(("Mass flow, annulus estimate [kg/s]", mass_flow))
    return {"total": total, "exit": exit_state, "stations": stations, "performance": performance,
            "mass_flow": mass_flow, "fluid": fluid}


def rotor_statistics(project, row, config, result, stator=None):
    fluid = jxp.Fluid(project["machine"]["fluid_name"])
    source = config.get("rotor_source")
    design = config["rotor_design"]
    r1, r2 = rotor_radii(config)
    speed = None
    if source:
        triangle = source["triangle"]
        inlet = state(fluid, jxp.DmassT_INPUTS, triangle["rho_static"], triangle["T_static"])
        speed = triangle["U"]
        rpm = source["rotational_speed"]["rpm"]
    else:
        quality = float(result.get("Q0_rel", float("nan")))
        relative = (state(fluid, jxp.PQ_INPUTS, float(result["P0_rel"]), quality) if isfinite(quality)
                    else state(fluid, jxp.PT_INPUTS, float(result["P0_rel"]), float(result["T0_rel"])))
        inlet = static_from_velocity(fluid, relative["h"], relative["s"], design["M_inlet"])
        specification = config["inlet_coupling"]["rotational_speed"]
        if r1 and specification["type"] == "actual":
            resolved = resolve_speed(specification, r1 * 1000)
            speed, rpm = resolved["U"], resolved["rpm"]
    beta1, beta2 = radians(result["beta_inlet"]), radians(result["beta_outlet"])
    w1 = result["M_inlet"] * inlet["a"]
    h0_relative = inlet["h"] + 0.5 * w1 ** 2
    u2 = speed * r2 / r1 if speed is not None else None
    if speed is not None:
        h0_relative -= 0.5 * speed ** 2 - 0.5 * u2 ** 2
    outlet = static_from_velocity(fluid, h0_relative, inlet["s"], result["M_outlet"])
    w2 = result["M_outlet"] * outlet["a"]
    wm1, wt1, wm2, wt2 = w1 * cos(beta1), w1 * sin(beta1), w2 * cos(beta2), w2 * sin(beta2)
    stations = [station("Rotor inlet (1)", "relative", inlet, w1, wm1, wt1),
                station("Rotor exit (2)", "relative", outlet, w2, wm2, wt2)]
    performance = [("Relative inlet Mach [-]", result["M_inlet"]), ("Relative exit Mach [-]", result["M_outlet"]),
                   ("Relative inlet angle [deg]", result["beta_inlet"]),
                   ("Relative exit angle [deg]", result["beta_outlet"]),
                   ("Flow turning [deg]", abs(result["beta_inlet"] - result["beta_outlet"])),
                   ("Solidity (chord/pitch) [-]", result.get("solidity", float("nan")))]
    triangles = {"inlet": {"U": None, "W": (wm1, wt1)}, "outlet": {"U": None, "W": (wm2, wt2)}}
    stage = None
    if speed is not None:
        vt1, vt2 = wt1 + speed, wt2 + u2
        stations.insert(1, station("Rotor inlet (1)", "absolute", inlet, hypot(wm1, vt1), wm1, vt1))
        stations.append(station("Rotor exit (2)", "absolute", outlet, hypot(wm2, vt2), wm2, vt2))
        work = speed * vt1 - u2 * vt2
        performance = [("Blade speed, inlet U₁ [m/s]", speed), ("Blade speed, exit U₂ [m/s]", u2),
                       ("Rotational speed [rpm]", rpm), ("Specific work, Euler [kJ/kg]", work / 1e3),
                       *performance]
        triangles = {"inlet": {"U": speed, "W": (wm1, wt1)}, "outlet": {"U": u2, "W": (wm2, wt2)}}
        if stator is not None:
            total = stator["total"]
            isentropic = total["h"] - outlet["h"]
            kinetic = 0.5 * (wm2 ** 2 + vt2 ** 2)
            stage = [("Isentropic drop, total-to-static [kJ/kg]", isentropic / 1e3),
                     ("Specific work [kJ/kg]", work / 1e3),
                     ("Efficiency, total-to-static [-]", work / isentropic),
                     ("Degree of reaction [-]", (inlet["h"] - outlet["h"]) / (total["h"] - outlet["h"])),
                     ("Exit kinetic energy fraction [-]", kinetic / isentropic),
                     ("Velocity ratio U₁/c₀ [-]", speed / sqrt(2 * isentropic))]
            if stator["mass_flow"]:
                stage.append(("Power, annulus estimate [kW]", stator["mass_flow"] * work / 1e3))
    return {"stations": stations, "performance": performance, "stage": stage, "triangles": triangles}


def _number(value, digits=4):
    if value is None or (isinstance(value, float) and not isfinite(value)):
        return "–"
    return f"{value:.{digits}g}"


def badge(status):
    return html.Span(status.title(), className=f"status-badge {status}")


def card(title, subtitle, rows, status=None):
    return html.Section([
        html.Div([html.Div([html.H4(title), html.Small(subtitle)]), badge(status) if status else None],
                    className="stats-card-header"),
        html.Table([html.Tbody([html.Tr([html.Td(name), html.Td(_number(value))]) for name, value in rows])],
                   className="stats-table"),
    ], className="stats-card")


def station_table(stations):
    head = [("Station", ""), ("Frame", ""), ("p", "kPa"), ("T", "K"), ("ρ", "kg/m³"), ("h", "kJ/kg"), ("M", "–"),
            ("|V|, |W|", "m/s"), ("Vm", "m/s"), ("Vt", "m/s"), ("Angle", "deg")]
    body = [html.Tr([html.Td(item["station"]), html.Td(html.Span(item["frame"], className="frame-tag")),
                     *[html.Td(_number(value)) for value in (
                         item["p"] / 1e3, item["T"], item["rho"], item["h"] / 1e3, item["M"], item["velocity"],
                         item["meridional"], item["tangential"], item["angle"])]]) for item in stations]
    return html.Table([html.Thead(html.Tr([html.Th([name, html.Br(), html.Small(unit)]) for name, unit in head])),
                       html.Tbody(body)], className="stats-table stations")


def _arrow(result, start, end, color, label, shift):
    result.add_annotation(x=end[0], y=end[1], ax=start[0], ay=start[1], xref="x", yref="y", axref="x", ayref="y",
                          showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.8, arrowcolor=color, text="")
    result.add_annotation(x=0.5 * (start[0] + end[0]), y=0.5 * (start[1] + end[1]), text=f"<b>{label}</b>",
                          showarrow=False, font=dict(color=color, size=13), xshift=shift)


def triangle_figure(stator_exit, triangles):
    result = plots.figure("Velocity triangles", "Meridional [m/s]", "Tangential [m/s]")
    points = [(0.0, 0.0)]
    vectors = []
    if stator_exit and not triangles:
        vectors.append(((0, 0), stator_exit, plots.EDGE, "V₁", -14))
    offset = 0.0
    span = max([abs(t["W"][0]) for t in (triangles or {}).values() if t] + [1.0])
    for key, index in (("inlet", "₁"), ("outlet", "₂")):
        triangle = (triangles or {}).get(key)
        if not triangle:
            continue
        wm, wt = triangle["W"]
        origin, tip_w = (offset, 0.0), (offset + wm, wt)
        vectors.append((origin, tip_w, "#477db3", "W" + index, 14))
        if triangle["U"] is not None:
            tip_v = (offset + wm, wt + triangle["U"])
            vectors += [(origin, tip_v, plots.EDGE, "V" + index, -14), (tip_w, tip_v, "#c94f4f", "U" + index, 14)]
        offset += abs(wm) + span
    for start, end, color, label, shift in vectors:
        _arrow(result, start, end, color, label, shift)
        points += [start, end]
    xs, ys = zip(*points)
    pad_x = 0.08 * (max(xs) - min(xs) or 1.0)
    pad_y = 0.08 * (max(ys) - min(ys) or 1.0)
    result.update_xaxes(range=[min(xs) - pad_x, max(xs) + 2 * pad_x], constrain="domain")
    result.update_yaxes(range=[min(ys) - pad_y, max(ys) + pad_y], scaleanchor="x", scaleratio=1, constrain="domain")
    result.update_layout(margin=dict(l=55, r=15, t=15, b=45))
    return result


def panel(project, registry):
    cards, stations, notes, statuses = [], [], [], {}
    stator, stator_exit, triangles, stage = None, None, None, None
    for row in projects.active_rows(project):
        references = (registry or {}).get(row["id"], {})
        role = row["role"].title()
        try:
            config = flow.resolved_configuration(project, row["id"], registry)
            result = flow.load_nozzle(references.get("nozzle"), config)
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            notes.append(f"{role}: {exc}")
            continue
        keys = flow.keys(config, references.get("nozzle"))
        status = "current" if flow.current(references.get("nozzle"), keys["nozzle"]) else "outdated"
        statuses[row["role"]] = status
        try:
            if row["role"] == "stator":
                stator = stator_statistics(project, row, config, result)
                exit_station = stator["stations"][-1]
                stator_exit = (exit_station["meridional"], exit_station["tangential"])
                cards.append(card("Stator", "Method of characteristics · absolute frame", stator["performance"],
                                  status))
                stations += stator["stations"]
            else:
                coupled = stator if config.get("rotor_source") else None
                rotor = rotor_statistics(project, row, config, result, coupled)
                absolute = rotor["triangles"]["inlet"]["U"] is not None
                frame = "relative and absolute frames" if absolute else "relative frame"
                cards.append(card("Rotor", f"Vortex method · {frame}", rotor["performance"], status))
                if coupled:
                    stations[-1]["station"] = "Stator exit / rotor inlet (1)"
                    stations += [item for item in rotor["stations"]
                                 if not (item["station"] == "Rotor inlet (1)" and item["frame"] == "absolute")]
                else:
                    stations += rotor["stations"]
                triangles, stage = rotor["triangles"], rotor["stage"]
        except (ValueError, RuntimeError, KeyError, ZeroDivisionError) as exc:
            notes.append(f"{role}: statistics unavailable ({exc}).")
    if stage:
        overall = "current" if all(value == "current" for value in statuses.values()) else "outdated"
        cards.append(card("Stage", "Stator + rotor · ideal, isentropic", stage, overall))
    if not cards:
        return [html.Small(" ".join(notes) or "Compute the blade design to display its statistics.",
                           className="message")]
    assumptions = html.Section([
        html.Div(html.H4("Assumptions"), className="stats-card-header"),
        html.Ul([html.Li(text) for text in ASSUMPTIONS], className="stats-notes"),
        html.Div(html.H4("Notation"), className="stats-card-header stats-subheader"),
        html.Ul([html.Li(text) for text in NOTATION], className="stats-notes"),
        *([html.Ul([html.Li(note) for note in notes], className="stats-notes warning")] if notes else []),
    ], className="stats-card")
    triangles_card = html.Section([
        html.Div(html.H4("Velocity triangles"), className="stats-card-header"),
        dcc.Graph(figure=triangle_figure(stator_exit, triangles),
                  config={"displaylogo": False, "displayModeBar": False}, style={"height": "400px"}),
    ], className="stats-card")
    return [html.Div(cards, className="stats-grid"),
            html.Section([html.Div(html.H4("Flow stations"), className="stats-card-header"),
                          html.Div(station_table(stations), className="stats-scroll")], className="stats-card"),
            html.Div([triangles_card, assumptions], className="stats-columns")]
