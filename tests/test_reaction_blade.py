"""Geometry-only rotor blades: TurboDash reaction section (camberline + modified NACA) and constant passage."""
from copy import deepcopy
from math import pi

import numpy as np
import pytest

pytest.importorskip("turbodash.geometry.blade_parametrization.thickness_models")

from turbo_moc.app import workflow as flow
from turbo_moc.app.rotor_inlet import inlet_warnings
from turbo_moc.configuration import project as projects
from turbo_moc.core.reaction_blade import (REACTION_DEFAULTS, constant_passage_geometric, geometric_rotor,
                                           reaction_blade, reaction_section)
from turbo_moc.geometry.rotor_blade import build_rotor_design

CASE = dict(REACTION_DEFAULTS)  # 0 -> 60 deg, location 0.52, concentration 1.2, t 0.12 at 0.30, r_LE 0.01, t_TE 0.015


def reaction_project(**reaction):
    project = projects.set_design_mode(projects.default_project("axial"), "rotor")
    row = projects.get_row(project, "rotor_1")
    row["geometry"]["meridional_channel"].update(hub_radius=45.0, blade_height=10.0, hub_chord=15.0,
                                                 tip_chord=15.0, num_blades=24)
    design = row["geometry"]["rotor_design"]
    design["blade_type"] = "reaction"
    design["reaction"] = dict(CASE, **reaction)
    return projects.validate_project(projects.normalize_project(project))


@pytest.mark.parametrize("location,concentration", [(0.52, 1.2), (0.65, 2.0), (0.35, 0.8)])
def test_camberline_angle_is_half_turning_at_the_specified_location(location, concentration):
    section = reaction_section(dict(CASE, turning_location=location, turning_concentration=concentration))
    beta = np.degrees(np.interp(location, section["xi"], section["beta"]))
    assert beta == pytest.approx(30.0, abs=1e-9)
    assert np.degrees(section["beta"][[0, -1]]) == pytest.approx([0.0, 60.0])


def test_location_and_concentration_act_independently():
    base = reaction_section(CASE)
    moved = reaction_section(dict(CASE, turning_location=0.65))
    sharper = reaction_section(dict(CASE, turning_concentration=2.0))
    grid = np.linspace(0.05, 0.95, 19)
    angle = lambda s: np.degrees(np.interp(grid, s["xi"], s["beta"]))
    # Moving the location delays the turning everywhere; it stays monotonic
    assert np.all(angle(moved) <= angle(base) + 1e-12) and np.all(np.diff(angle(moved)) > 0)
    # Concentration keeps the half-turning point and steepens the turning around it
    assert np.degrees(np.interp(0.52, sharper["xi"], sharper["beta"])) == pytest.approx(30.0)
    slope = lambda s: np.gradient(np.degrees(s["beta"]), s["xi"])[np.argmin(abs(s["xi"] - 0.52))]
    assert slope(sharper) > 1.3 * slope(base)
    # Thickness is untouched by the camberline parameters
    for other in (moved, sharper):
        assert np.interp(grid, other["xi"], other["thickness"]) == pytest.approx(
            np.interp(grid, base["xi"], base["thickness"]), rel=1e-4)  # Sampling grids differ by one station


def test_naca_thickness_law_endpoints():
    s = reaction_section(CASE)
    assert s["thickness"].max() == pytest.approx(0.12, rel=1e-12)
    assert s["xi"][np.argmax(s["thickness"])] == pytest.approx(0.30)
    assert s["thickness"][-1] == pytest.approx(0.015)
    # Square-root nose: t ~ 2 sqrt(2 r ds/dxi xi) as xi -> 0 (beta_in = 0 so ds/dxi = 1)
    xi = s["xi"][1:4]
    assert s["thickness"][1:4] == pytest.approx(2 * np.sqrt(2 * 0.01 * xi), rel=0.05)
    # Offsets are normal to the camberline and symmetric about it
    half = np.linalg.norm(s["plus"] - s["camber"], axis=1)
    assert half == pytest.approx(s["thickness"] / 2)


def test_section_maps_to_requested_chord_pitch_and_frame():
    config = projects.row_configuration(reaction_project(), "rotor_1")
    rotor = geometric_rotor(config)
    assert rotor["beta_inlet"] == 0.0 and rotor["beta_outlet"] == -60.0  # TurboMoC solver frame: y = -theta
    design = build_rotor_design(rotor, config)
    curve = np.column_stack((design.blade["blade_curve"]["x"], design.blade["blade_curve"]["y"]))
    assert np.allclose(curve[0], curve[-1])
    pitch = 2 * pi * 50.0 / 24
    assert design.blade["pitch"] == pytest.approx(pitch, rel=1e-9)
    assert rotor["pitch"] * 15.0 == pytest.approx(pitch, rel=1e-9)  # natural pitch = row pitch
    # Camberline chord 15 mm: LE at q_y = 0, TE camber point at q_y = -15 mm (TE cut adds t_TE/2 sin(beta))
    assert curve[:, 1].max() == pytest.approx(0.0, abs=1e-5)  # Spline sampling of the nose
    extent = -curve[:, 1].min()
    assert extent == pytest.approx(15.0 * (1 + 0.5 * 0.015 * np.sin(np.radians(60))), rel=1e-3)
    assert set(design.curves) >= {"blade_curve"}


def test_validation_inlet_and_defaults():
    project = reaction_project()
    config = projects.row_configuration(project, "rotor_1")
    config["rotor_design"]["M_inlet"] = 0.4  # Subsonic: allowed for a geometric blade
    assert inlet_warnings(config) == []
    bad = deepcopy(project)
    projects.get_row(bad, "rotor_1")["geometry"]["rotor_design"]["reaction"]["turning_location"] = 1.2
    with pytest.raises(ValueError, match="turning_location"):
        projects.validate_project(bad)
    # Vortex rows are not given geometric parameter groups (their job keys stay unchanged)
    vortex = projects.set_design_mode(projects.default_project("axial"), "rotor")
    design = projects.get_row(vortex, "rotor_1")["geometry"]["rotor_design"]
    assert "reaction" not in design and "constant_passage" not in design
    # Selecting the blade type in the editor adds its parameter group
    edited = flow.edited_project(vortex, ["constant_passage"], [{"row": "rotor_1", "path": "rotor_design.blade_type"}])
    group = projects.get_row(edited, "rotor_1")["geometry"]["rotor_design"]["constant_passage"]
    assert set(group) == {"blade_angle_deg", "pitch_to_chord"}
    projects.validate_project(edited)


def test_geometry_edit_invalidates_blade_and_meshes():
    config = projects.row_configuration(reaction_project(), "rotor_1")
    reference = {"session": "a" * 32, "job": "b" * 32, "row_id": "rotor_1", "key": flow.keys(config)["nozzle"]}
    before = flow.keys(config, reference)
    for key, value in (("turning_location", 0.65), ("maximum_thickness", 0.13), ("trailing_edge_thickness", 0.02)):
        edited = deepcopy(config)
        edited["rotor_design"]["reaction"][key] = value
        after = flow.keys(edited, reference)
        assert after["nozzle"] != before["nozzle"]
        assert after["reference"] != before["reference"] and after["volume"] != before["volume"]


def test_subsonic_reaction_job_starts(monkeypatch, tmp_path):
    project = reaction_project()
    config = flow.resolved_configuration(project, "rotor_1", {})
    config["rotor_design"]["M_inlet"] = 0.5
    projects.get_row(project, "rotor_1")["geometry"]["rotor_design"]["M_inlet"] = 0.5

    class Process:
        pid = 1
    monkeypatch.setattr(flow, "OUTPUT", tmp_path)
    monkeypatch.setattr(flow.subprocess, "Popen", lambda *args, **kwargs: Process())
    handle = flow.start_job("d" * 32, "nozzle", config, {}, project=project, row_id="rotor_1",
                            row_role="rotor", registry={})
    flow.PROCESSES.pop((handle["session"], handle["job"]))
    assert handle["action"] == "nozzle"


def test_constant_passage_from_geometry_only():
    blade = constant_passage_geometric({"blade_angle_deg": 60.0, "pitch_to_chord": 0.6}, le_te_ratio=25.0,
                                       num_points=60)
    assert blade["pitch"] / np.ptp(blade["blade"]["x"]) == pytest.approx(0.6, rel=1e-9)
    assert blade["beta_outlet"] == -60.0
    config = projects.row_configuration(reaction_project(), "rotor_1")
    config["rotor_design"]["blade_type"] = "constant_passage"
    config["rotor_design"]["constant_passage"] = {"blade_angle_deg": 55.0, "pitch_to_chord": 0.6}
    rotor = geometric_rotor(config)  # Annular row: the blade count sets pitch/chord
    assert rotor["pitch"] / np.ptp(rotor["blade"]["x"]) == pytest.approx(2 * pi * 50 / 24 / 15, rel=1e-9)
    design = build_rotor_design(rotor, config)
    assert design.blade["pitch"] == pytest.approx(2 * pi * 50 / 24)


def test_reaction_result_layout():
    rotor = reaction_blade(dict(CASE))
    n = len(rotor["blade_lower"]["x"])
    assert len(rotor["blade"]["x"]) == 2 * n
    assert rotor["blade"]["x"][0] == rotor["blade"]["x"][-1] and rotor["blade"]["y"][0] == rotor["blade"]["y"][-1]


# --- Studio (Dash callback endpoint): select, edit, save and reload --------------------------------------------------
def _parameters(component):
    """All mounted parameter controls of a rendered sidebar: (id, value) pairs."""
    found = []
    if isinstance(component, dict):
        props = component.get("props", {})
        identifier = props.get("id")
        if isinstance(identifier, dict) and identifier.get("type") == "parameter":
            found.append((identifier, props.get("value")))
        for item in component.values():
            found += _parameters(item)
    elif isinstance(component, list):
        for item in component:
            found += _parameters(item)
    return found


def _sidebar(app, project):
    from app_helpers import call
    changed = call(app, "..base-project.data", "loaded-project.data", {
        ("loaded-project", "data"): project, ("design-mode", "value"): "stator",
        ("machine-family", "value"): "axial", ("base-project", "data"): projects.default_project(),
        ("selected-row", "value"): "rotor_1"})
    return changed["base-project"]["data"], changed["settings"]["children"]


def _edit(app, base, controls):
    from app_helpers import call
    key = next(key for key in app.callback_map if key.startswith("..config.data"))
    pattern = next(item["id"] for item in app.callback_map[key]["inputs"] if item["id"].startswith("{"))
    result = call(app, "..config.data", pattern + ".value", {
        ("base-project", "data"): base, ("selected-row", "value"): "rotor_1", ("row-references", "data"): {},
        (pattern, "value"): [value for _, value in controls], (pattern, "id"): [ident for ident, _ in controls]})
    return result["config"]["data"]


def test_studio_select_edit_save_reload(tmp_path):
    pytest.importorskip("dash_vtk")
    import base64
    import yaml
    from app_helpers import call
    from turbo_moc.app import create_app

    app = create_app()
    # A vortex rotor: the selector is mounted, the geometric groups are hidden and ignored on edit
    vortex = projects.set_design_mode(projects.default_project("axial"), "rotor")
    base, sidebar = _sidebar(app, vortex)
    controls = _parameters(sidebar)
    paths = {ident["path"]: value for ident, value in controls if ident["row"] == "rotor_1"}
    assert paths["rotor_design.blade_type"] == "vortex"
    assert "rotor_design.reaction.turning_location" in paths
    state = _edit(app, base, controls)
    assert state["valid"]
    assert "reaction" not in projects.get_row(state["project"], "rotor_1")["geometry"]["rotor_design"]
    # Select the reaction blade and edit its half-turning location in the same submission
    edited = [(ident, "reaction" if ident["path"] == "rotor_design.blade_type" and ident["row"] == "rotor_1"
               else 0.6 if ident["path"] == "rotor_design.reaction.turning_location" else value)
              for ident, value in controls]
    state = _edit(app, base, edited)
    assert state["valid"], state
    design = projects.get_row(state["project"], "rotor_1")["geometry"]["rotor_design"]
    assert design["blade_type"] == "reaction" and design["reaction"]["turning_location"] == 0.6
    assert "constant_passage" not in design
    # Save, reload: identical project, selector and visible group restored
    saved = call(app, "..loaded-project.data", "save-configuration.n_clicks", {
        ("save-configuration", "n_clicks"): 1, ("base-project", "data"): projects.default_project(),
        ("config", "data"): {"project": state["project"], "valid": True}})
    content = saved["configuration-download"]["data"]["content"]
    assert yaml.safe_load(content) == state["project"]
    loaded = call(app, "..loaded-project.data", "configuration-upload.contents", {
        ("configuration-upload", "contents"): "data:text/yaml;base64," + base64.b64encode(content.encode()).decode(),
        ("configuration-upload", "filename"): "reaction.yaml", ("base-project", "data"): projects.default_project()})
    assert loaded["loaded-project"]["data"] == state["project"]
    base, sidebar = _sidebar(app, loaded["loaded-project"]["data"])
    paths = {ident["path"]: value for ident, value in _parameters(sidebar) if ident["row"] == "rotor_1"}
    assert paths["rotor_design.blade_type"] == "reaction"
    assert paths["rotor_design.reaction.turning_location"] == 0.6
    # A geometry edit after reload changes the blade, reference and volume identities
    config = flow.resolved_configuration(base, "rotor_1", {})
    reference = {"session": "a" * 32, "job": "b" * 32, "row_id": "rotor_1", "key": flow.keys(config)["nozzle"]}
    old = flow.keys(config, reference)
    state = _edit(app, base, [(ident, 1.5 if ident["path"] == "rotor_design.reaction.turning_concentration"
                               else value) for ident, value in _parameters(sidebar)])
    new = flow.keys(state["config"], reference)
    assert state["config"]["rotor_design"]["reaction"]["turning_concentration"] == 1.5
    assert new["volume"] != old["volume"] and not flow.current({"key": old["volume"]}, new["volume"])


def test_coupled_inlet_sets_the_inlet_metal_angle(tmp_path, monkeypatch):
    import json as _json
    from turbo_moc.app import rotor_inlet
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    project["machine"]["fluid_name"] = "Air"
    rotor = projects.get_row(project, "rotor_1")["geometry"]
    rotor["inlet_coupling"].update(mode="stator_exit", rotational_speed={"type": "actual", "value": 4000})
    rotor["rotor_design"]["blade_type"] = "reaction"
    project = projects.validate_project(projects.normalize_project(project))
    stator = projects.row_configuration(project, "stator_1")
    reference = {"session": "a" * 32, "job": "b" * 32, "row_id": "stator_1", "key": flow.keys(stator)["nozzle"]}
    monkeypatch.setattr(flow, "OUTPUT", tmp_path)
    folder = flow.directory(reference)
    folder.mkdir(parents=True)
    flow.write_json(folder / "moc_result.json", {"fluid_name": "Air", "converged": True,
        "wall_final": {"p": [100000.], "T": [280.], "rho": [1.2], "M": [2.]}})
    monkeypatch.setattr(rotor_inlet, "_triangle", lambda *args: {
        "P0_rel": 150000., "T0_rel": 330., "M_rel": 0.6, "beta_rel_deg": 45., "U": 40.})
    registry = {"stator_1": {"nozzle": reference}}
    config = flow.resolved_configuration(project, "rotor_1", registry)
    assert config["rotor_design"]["reaction"]["inlet_metal_angle_deg"] == -45.0  # TurboDash convention
    assert geometric_rotor(config)["beta_inlet"] == 45.0  # TurboMoC frame = beta_rel
    # The saved manual value is untouched
    saved = projects.get_row(project, "rotor_1")["geometry"]["rotor_design"]["reaction"]["inlet_metal_angle_deg"]
    assert saved == CASE["inlet_metal_angle_deg"]


def test_geometric_rotor_gets_ideal_triangles_and_reaction():
    from turbo_moc.app import design_statistics as statistics
    from turbo_moc.configuration.fluid_inputs import solver_parameters
    from turbo_moc.configuration.rotor import vortex_parameters
    project = reaction_project()
    projects.get_row(project, "rotor_1")["geometry"]["inlet_coupling"]["mode"] = "manual"
    design = projects.get_row(project, "rotor_1")["geometry"]["rotor_design"]
    design.update(M_inlet=0.6, outlet_condition="pressure")
    design["reaction"]["inlet_metal_angle_deg"] = -35.0  # beta_rel = +35 deg
    project = projects.validate_project(projects.normalize_project(project))
    config = flow.resolved_configuration(project, "rotor_1", {})
    config["rotor_design"]["p_back"] = 0.7 * config["rotor_design"].get("P0_rel", 5e5)
    rotor = geometric_rotor(config)
    states = vortex_parameters(solver_parameters(config["rotor_design"], rotor=True))  # As the worker does
    rotor.update({name: states[name] for name in ("fluid_name", "P0_rel", "T0_rel", "M_inlet", "M_outlet")})
    row = projects.get_row(project, "rotor_1")
    stats = statistics.rotor_statistics(project, row, config, rotor)
    assert stats["triangles"]["inlet"] and stats["triangles"]["outlet"]
    performance = dict(stats["performance"])
    assert performance["Relative inlet Mach [-]"] == pytest.approx(0.6)
    assert 0 < performance["Relative exit Mach [-]"] < 1.5
    # The stage card (degree of reaction, efficiency) needs the coupled stator, as for vortex rotors
    assert stats["stage"] is None


@pytest.mark.parametrize("target", [0.0, 0.3, 0.5])
def test_degree_of_reaction_outlet_sets_the_backpressure(tmp_path, monkeypatch, target):
    import jaxprop as jxp
    from turbo_moc.app import rotor_inlet
    from turbo_moc.configuration.fluid_inputs import solver_parameters
    from turbo_moc.configuration.rotor import vortex_parameters
    project = projects.set_design_mode(projects.default_project(), "stator_rotor")
    project["machine"]["fluid_name"] = "Air"
    rotor = projects.get_row(project, "rotor_1")["geometry"]
    rotor["inlet_coupling"].update(mode="stator_exit", rotational_speed={"type": "actual", "value": 4000})
    rotor["rotor_design"].update(blade_type="reaction", outlet_condition="reaction", degree_of_reaction=target)
    project = projects.validate_project(projects.normalize_project(project))
    stator = projects.row_configuration(project, "stator_1")
    reference = {"session": "a" * 32, "job": "b" * 32, "row_id": "stator_1", "key": flow.keys(stator)["nozzle"]}
    monkeypatch.setattr(flow, "OUTPUT", tmp_path)
    folder = flow.directory(reference)
    folder.mkdir(parents=True)
    flow.write_json(folder / "moc_result.json", {"fluid_name": "Air", "converged": True,
        "wall_final": {"p": [100000.], "T": [280.], "rho": [1.244], "M": [1.2]}})
    static = jxp.Fluid("Air").get_state(jxp.PT_INPUTS, 100000., 280.)
    monkeypatch.setattr(rotor_inlet, "_triangle", lambda *args: {  # Stator exit state, consistent with the EOS
        "p_static": 100000., "T_static": 280., "rho_static": float(static.rho), "V_abs": 1.2 * float(static.a),
        "P0_rel": 150000., "T0_rel": 330., "M_rel": 0.6, "beta_rel_deg": 45., "U": 40.})
    config = flow.resolved_configuration(project, "rotor_1", {"stator_1": {"nozzle": reference}})
    design, triangle = config["rotor_design"], config["rotor_source"]["triangle"]
    assert design["outlet_condition"] == "pressure" and design["p_back"] <= triangle["p_static"]
    fluid = jxp.Fluid("Air")
    inlet = fluid.get_state(jxp.DmassT_INPUTS, triangle["rho_static"], triangle["T_static"])
    exit_state = fluid.get_state(jxp.PSmass_INPUTS, design["p_back"], float(inlet.s))
    h1, h2 = float(inlet.h), float(exit_state.h)
    h00 = h1 + 0.5 * triangle["V_abs"] ** 2
    assert (h1 - h2) / (h00 - h2) == pytest.approx(target, abs=1e-6)
    assert vortex_parameters(solver_parameters(design, rotor=True))["M_outlet"] > 0  # Ideal statistics resolve


def test_degree_of_reaction_outlet_needs_a_coupled_inlet():
    from turbo_moc.configuration.rotor import vortex_parameters
    with pytest.raises(ValueError, match="coupled"):
        vortex_parameters({"beta_inlet": 30.0, "outlet_condition": "reaction", "degree_of_reaction": 0.5,
                           "blade_type": "reaction"})


def test_natural_blade_count_follows_turbodash_solidity():
    import numpy as np
    from turbo_moc.app import blade_counts
    from turbo_moc.core.reaction_blade import natural_pitch_to_chord
    p = dict(CASE, solidity_chord=1.2, inlet_metal_angle_deg=10.0, outlet_metal_angle_deg=60.0)
    # TurboDash (core.turbine): c = c_m / cos(mean metal angle), s = c / solidity
    assert natural_pitch_to_chord(p) == pytest.approx(1 / (1.2 * np.cos(np.radians(35.0))))
    blade = reaction_blade(dict(p, pitch_to_chord=0.5))  # Drawn pitch differs from the natural one
    native = blade_counts.native_passage({"rotor_design": {}}, blade)
    assert native["pitch"] == pytest.approx(natural_pitch_to_chord(p))
