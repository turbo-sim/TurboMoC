"""Three-dimensional Fluent contracts without starting a licensed solver."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import jaxprop as jxp
import numpy as np
import pytest
import yaml

from turbo_moc.fluent import import_checks, workflow_3d as workflow
from turbo_moc.configuration.fluid_inputs import pair_properties
from turbo_moc.geometry.linear_cascade import (
    LinearCascadeDefinition,
    LinearCascadeMapping,
)
from turbo_moc.geometry.meridional_channel import (
    MeridionalChannel,
    geometry_from_machine,
)

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_FILE = ROOT / "examples/fluent_3D/stator/workflow_fluent_stator_3D.yaml"
CASES = ROOT / "examples/apps/cases"


@pytest.fixture
def raw_workflow():
    raw = yaml.safe_load(WORKFLOW_FILE.read_text(encoding="utf-8"))
    for key in ("machine", "workflow", "rows"):
        raw.pop(key)
    raw["design_file"] = "../../gmsh_3D/workflow_gmsh_stator_3D.yaml"
    return raw


def test_example_resolves_project_and_all_three_momentum_equations():
    options, config = workflow.read_workflow(WORKFLOW_FILE)
    assert options["design_file"].is_absolute()
    assert options["output_dir"] == WORKFLOW_FILE.parent / "output"
    assert options["row_id"] == "stator_1"
    assert config["meridional_channel"]["type"] == "axial"
    assert config["mesh"]["cgns"]["length_scale"] == 0.001
    assert set(options["fluent"]["residual_equations"]) == {
        "continuity",
        "x-velocity",
        "y-velocity",
        "z-velocity",
        "k",
        "omega",
    }
    assert set(options["fluent"]["boundaries"]) == {
        "inlet",
        "outlet",
        "blade",
        "hub",
        "shroud",
        "periodic_left",
        "periodic_right",
    }


@pytest.mark.parametrize("family", ["linear", "axial", "radial"])
def test_selects_supported_stationary_stator_rows(raw_workflow, family):
    raw_workflow["design_file"] = str(CASES / f"{family}_stator.yaml")
    _, config = workflow.validate_workflow(raw_workflow, WORKFLOW_FILE)
    assert config["meridional_channel"]["type"] == family


@pytest.mark.parametrize("pair", ["PQ_INPUTS", "QT_INPUTS", "HmassP_INPUTS"])
@pytest.mark.parametrize("expands", [True, False], ids=["expansion", "no-expansion"])
def test_pair_validation_resolves_pressure_in_jaxprop_input_order(
    raw_workflow, tmp_path, pair, expands
):
    project = yaml.safe_load((CASES / "linear_stator.yaml").read_text(encoding="utf-8"))
    inlet = jxp.Fluid(project["machine"]["fluid_name"]).get_state(jxp.PQ_INPUTS, 251300., .025)
    properties = {"P": float(inlet.p), "T": float(inlet.T),
                  "Q": float(inlet.Q), "Hmass": float(inlet.h)}
    first, second = pair_properties(pair)
    inputs = project["rows"][0]["geometry"]["method_of_characteristics"]
    inputs.update(inlet_property_pair=pair, inlet_property_1=properties[first],
                  inlet_property_2=properties[second],
                  p_back=95000. if expands else float(inlet.p) * 1.01)
    design_file = tmp_path / "paired_inlet.yaml"
    design_file.write_text(yaml.safe_dump(project), encoding="utf-8")
    raw_workflow["design_file"] = str(design_file)
    if not expands:
        with pytest.raises(ValueError, match="p_back"):
            workflow.validate_workflow(raw_workflow, WORKFLOW_FILE)
        return
    _, config = workflow.validate_workflow(raw_workflow, WORKFLOW_FILE)
    actual = config["method_of_characteristics"]
    assert actual["inlet_property_pair"] == pair
    assert actual["inlet_property_1"] == inputs["inlet_property_1"]
    assert actual["inlet_property_2"] == inputs["inlet_property_2"]
    assert not {"P0", "T0", "Q0"} & actual.keys()


@pytest.mark.parametrize("missing", ["z-velocity", "hub", "shroud"])
def test_rejects_incomplete_3d_physics_configuration(raw_workflow, missing):
    section = "residual_equations" if missing == "z-velocity" else "boundaries"
    del raw_workflow["fluent"][section][missing]
    with pytest.raises(ValueError, match="residual|boundar|hub|shroud"):
        workflow.validate_workflow(raw_workflow, WORKFLOW_FILE)


@pytest.mark.parametrize("row_id", ["missing_row", "rotor_1"])
def test_rejects_unknown_or_rotating_row(raw_workflow, row_id):
    raw_workflow["design_file"] = str(CASES / "stator_rotor.yaml")
    raw_workflow["row_id"] = row_id
    with pytest.raises(ValueError, match="row|stator|rotor"):
        workflow.validate_workflow(raw_workflow, WORKFLOW_FILE)


def test_rejects_inactive_stator_row(raw_workflow, tmp_path):
    project = yaml.safe_load((CASES / "stator_rotor.yaml").read_text(encoding="utf-8"))
    project["machine"]["design_mode"] = "rotor"
    design_file = tmp_path / "inactive_stator.yaml"
    design_file.write_text(yaml.safe_dump(project), encoding="utf-8")
    raw_workflow["design_file"] = str(design_file)
    raw_workflow["row_id"] = "stator_1"
    with pytest.raises(ValueError, match="not active"):
        workflow.validate_workflow(raw_workflow, WORKFLOW_FILE)


@pytest.mark.parametrize("scale", [1.0, 0.01])
def test_rejects_cgns_coordinates_not_in_metres(raw_workflow, tmp_path, scale):
    source = WORKFLOW_FILE.parent / raw_workflow["design_file"]
    project = yaml.safe_load(source.read_text(encoding="utf-8"))
    project["rows"][0]["mesh"]["cgns"]["length_scale"] = scale
    design_file = tmp_path / "wrong_units.yaml"
    design_file.write_text(yaml.safe_dump(project), encoding="utf-8")
    raw_workflow["design_file"] = str(design_file)
    with pytest.raises(ValueError, match="metre|length_scale|0.001"):
        workflow.validate_workflow(raw_workflow, WORKFLOW_FILE)


def periodic_test_points():
    """Unequally spaced nodes expose nearest-neighbor and transform mistakes."""
    return np.array(
        [
            [0.10, 0.002, -0.010],
            [0.10, 0.004, 0.003],
            [0.11, 0.008, 0.020],
            [0.11, 0.013, 0.011],
        ]
    )


@pytest.mark.parametrize("family", ["linear", "axial", "radial"])
def test_geometry_uses_physical_axes_signed_rotation_and_metres(family):
    config = workflow.read_configuration(CASES / f"{family}_stator.yaml")
    if family == "linear":
        mapping = LinearCascadeMapping(LinearCascadeDefinition(20.0, 10.0, 5.0))
    else:
        mapping = MeridionalChannel(geometry_from_machine(config["meridional_channel"]))
    context = workflow.geometry_context(
        SimpleNamespace(meridional_channel=mapping), config
    )
    transform = np.asarray(context["transform"])
    assert context["kind"] == family
    assert context["axis_direction"] == [0.0, 0.0, 1.0]
    assert context["axis_origin"] == [0.0, 0.0, 0.0]
    if family == "linear":
        assert context["translation_m"] == pytest.approx([0.0, 0.005, 0.0])
        assert transform @ [0.02, 0.0, 0.007, 1.0] == pytest.approx(
            [0.02, 0.005, 0.007, 1.0]
        )
    else:
        expected_angle = -360.0 / config["meridional_channel"]["num_blades"]
        angle = np.deg2rad(expected_angle)
        assert context["angle_deg"] == pytest.approx(expected_angle)
        assert transform @ [0.10, 0.0, 0.007, 1.0] == pytest.approx(
            [0.10 * np.cos(angle), 0.10 * np.sin(angle), 0.007, 1.0]
        )
    if family == "axial":
        dimensions = config["meridional_channel"]
        assert context["section"]["radius_m"] == pytest.approx(
            (dimensions["hub_radius"] + dimensions["blade_height"] / 2) / 1000
        )
    else:
        assert context["section"]["point_m"] == pytest.approx(
            [0, 0, config["meridional_channel"]["blade_height"] / 2000]
        )


@pytest.mark.parametrize("rotational", [False, True], ids=["translation", "rotation"])
def test_periodic_matching_handles_reordered_nodes_and_signed_transform(rotational):
    left = periodic_test_points()
    transform = np.eye(4)
    if rotational:
        angle = np.deg2rad(-360.0 / 31)
        cosine, sine = np.cos(angle), np.sin(angle)
        transform[:2, :2] = [[cosine, -sine], [sine, cosine]]
    else:
        transform[1, 3] = 0.005
    right = left @ transform[:3, :3].T + transform[:3, 3]
    report = workflow.validate_periodic_vertices((left, right[[2, 0, 3, 1]]), transform)
    assert report["node_count"] == len(left)
    assert report["max_mismatch_m"] < 1e-12
    with pytest.raises(RuntimeError, match="periodic|match|transform"):
        workflow.validate_periodic_vertices((left, right), np.linalg.inv(transform))


@pytest.mark.parametrize("defect", ["count", "coordinate", "duplicate"])
def test_periodic_matching_rejects_nonconformal_nodes(defect):
    left = periodic_test_points()
    right = left.copy()
    if defect == "count":
        right = right[:-1]
    elif defect == "coordinate":
        right[0, 2] += 0.0001
    else:
        right[-1] = right[0]
    with pytest.raises(RuntimeError, match="periodic|match|node"):
        workflow.validate_periodic_vertices((left, right), np.eye(4))


def test_periodic_matching_requires_bijection_even_within_coordinate_tolerance():
    left = periodic_test_points()
    left[1] = left[0] + [0, 0, 1e-10]
    right = left.copy()
    right[1, 2] += 1e-9
    # Both close left nodes select the same right node, although every distance
    # lies below the coordinate tolerance. This is not a conformal node pairing.
    with pytest.raises(RuntimeError, match="periodic|match"):
        workflow.validate_periodic_vertices((left, right), np.eye(4))


@pytest.mark.parametrize("family", ["linear", "axial"])
@pytest.mark.parametrize(
    "already_paired", [False, True], ids=["new_pair", "imported_pair"]
)
def test_periodic_setup_passes_physical_translation_or_rotation_axis(
    monkeypatch,
    family,
    already_paired,
):
    options, config = workflow.read_workflow(WORKFLOW_FILE)
    if family == "linear":
        config["meridional_channel"] = {
            "type": "linear",
            "blade_height": 10.0,
            "meridional_chord": 20.0,
        }
        mapping = LinearCascadeMapping(LinearCascadeDefinition(20.0, 10.0, 5.0))
    else:
        mapping = MeridionalChannel(geometry_from_machine(config["meridional_channel"]))
    geometry = workflow.geometry_context(
        SimpleNamespace(meridional_channel=mapping), config
    )
    matrix = np.asarray(geometry["transform"])
    left = periodic_test_points()
    right = left @ matrix[:3, :3].T + matrix[:3, 3]
    expected = options["fluent"]["boundaries"]
    zones = [expected["periodic_left"], expected["periodic_right"]]
    settings = MagicMock()
    monkeypatch.setattr(
        workflow.common, "boundary_types", lambda bc: dict.fromkeys(zones, "wall")
    )
    periodic = settings.setup.boundary_conditions.periodic
    periodic.get_object_names.side_effect = [zones if already_paired else [], zones]

    class ConformalPeriodic:
        # Fluent 24.2 exposes the conformal pair's axes only on its cell zone.
        # Reject accidental writes to inactive boundary axis attributes.
        __slots__ = ("p_jump", "rotationally_periodic")

        def __init__(self):
            self.p_jump = Mock()

    periodic.__getitem__.side_effect = {
        name: SimpleNamespace(periodic=ConformalPeriodic()) for name in zones
    }.__getitem__
    fluid = settings.setup.cell_zone_conditions.fluid
    fluid.get_object_names.return_value = ["fluid_cells"]
    frame = SimpleNamespace(
        frame_motion=True,
        reference_frame_axis_origin=[SimpleNamespace(value=None) for _ in range(3)],
        reference_frame_axis_direction=[SimpleNamespace(value=None) for _ in range(3)],
    )
    fluid.__getitem__.return_value = SimpleNamespace(reference_frame=frame)
    assert (
        workflow.configure_periodicity(settings, expected, geometry, (left, right))
        == zones[0]
    )
    if already_paired:
        settings.mesh.modify_zones.make_periodic.assert_not_called()
    else:
        arguments = settings.mesh.modify_zones.make_periodic.call_args.kwargs
        assert arguments["rotate_periodic"] is (family != "linear")
        assert arguments["zone_name"] == zones[0]
        assert arguments["shadow_zone_name"] == zones[1]
        if family == "linear":
            assert arguments["direction"] == pytest.approx([0, 0.005, 0])
        else:
            assert "direction" not in arguments
    fluid.get_object_names.assert_called_once()
    assert frame.frame_motion is False
    if family != "linear":
        assert [item.value for item in frame.reference_frame_axis_origin] == [0, 0, 0]
        assert [item.value for item in frame.reference_frame_axis_direction] == [
            0,
            0,
            1,
        ]
    assert geometry["periodic_validation"]["node_count"] == 4
    assert geometry["periodic_validation"]["max_mismatch_m"] < 1e-12
    settings.mesh.check.assert_called_once()


def test_3d_setup_configures_all_walls_and_monitors_z_momentum(monkeypatch, tmp_path):
    options, config = workflow.read_workflow(WORKFLOW_FILE)
    options = options["fluent"]
    mapping = MeridionalChannel(geometry_from_machine(config["meridional_channel"]))
    geometry = workflow.geometry_context(
        SimpleNamespace(meridional_channel=mapping), config
    )
    zones = {role: f"import-{name}" for role, name in options["boundaries"].items()}
    types = {name: "wall" for name in zones.values()}
    types[zones["inlet"]] = "pressure_inlet"
    types[zones["outlet"]] = "pressure_outlet"
    monkeypatch.setattr(workflow.common, "boundary_types", lambda bc: types)
    monkeypatch.setitem(
        sys.modules,
        "ansys.fluent.core",
        SimpleNamespace(
            SurfaceFieldDataRequest=SimpleNamespace,
            SurfaceDataType=SimpleNamespace(Vertices="vertices"),
        ),
    )
    solver = MagicMock()
    settings = solver.settings
    settings.setup.cell_zone_conditions.fluid.get_object_names.return_value = [
        "fluid_cells"
    ]
    settings.solution.methods.high_order_term_relaxation.options.select_variables.allowed_values.return_value = [
        "Flow Variables Only",
    ]
    residuals = settings.solution.monitor.residual.equations
    equations = {name: SimpleNamespace() for name in options["residual_equations"]}
    residuals.get_object_names.return_value = list(equations)
    residuals.__getitem__.side_effect = equations.__getitem__
    walls = {zones[role]: MagicMock() for role in ("blade", "hub", "shroud")}
    settings.setup.boundary_conditions.wall.__getitem__.side_effect = walls.__getitem__
    solver.fields.field_data.get_field_data.return_value = {
        zones[role]: SimpleNamespace(vertices=periodic_test_points()) for role in zones
    }
    solver.fields.reduction.count.return_value = 12
    monkeypatch.setattr(
        workflow, "configure_periodicity", Mock(return_value=zones["periodic_left"])
    )
    monkeypatch.setattr(workflow, "prepare_contour_graphics", Mock())
    state = {"P0": 251300.0, "p_back": 95000.0}

    configured = workflow.configure_fluent(
        solver, tmp_path / "mesh.cgns", state, {}, geometry, options
    )

    assert all(configured[role] == zones[role] for role in zones)
    for wall in walls.values():
        assert wall.momentum.wall_motion == "Stationary Wall"
        assert wall.momentum.shear_condition == "No Slip"
    assert equations["z-velocity"].check_convergence is True
    assert (
        equations["z-velocity"].absolute_criteria
        == options["residual_equations"]["z-velocity"]
    )
    assert settings.setup.models.energy.enabled is False
    assert settings.setup.models.viscous.k_omega_model == "sst"
    settings.solution.initialization.initialize.assert_not_called()


@pytest.fixture
def imported_mesh(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "ansys.fluent.core",
        SimpleNamespace(
            SurfaceFieldDataRequest=SimpleNamespace,
            SurfaceDataType=SimpleNamespace(Vertices="vertices"),
        ),
    )
    solver = MagicMock()
    solver.settings.setup.cell_zone_conditions.fluid.get_object_names.return_value = [
        "fluid_cells"
    ]
    fluid_zone = object()
    solver.settings.setup.cell_zone_conditions.fluid.__getitem__.return_value = (
        fluid_zone
    )
    zones = {name: f"prefix-{name}" for name in import_checks.BOUNDARY_ROLES}

    def count(locations):
        if len(locations) == 1 and locations[0] is fluid_zone:
            return 48
        # PyFluent 0.42 accepts names for boundary surfaces, but needs the
        # actual settings object for a fluid cell zone.
        assert len(locations) == 1 and locations[0] in zones.values()
        return 2

    solver.fields.reduction.count.side_effect = count
    vertices = periodic_test_points()
    data = {name: SimpleNamespace(vertices=vertices.copy()) for name in zones.values()}
    solver.fields.field_data.get_field_data.return_value = data
    expected = {
        "cell_count": 48,
        "bounds_m": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
    }
    return solver, zones, expected, data


def test_import_audit_compares_exported_count_bounds_and_all_seven_boundaries(
    imported_mesh,
):
    solver, zones, expected, _ = imported_mesh
    report = import_checks.validate_import(solver, zones, expected)
    assert report["cell_count"] == report["expected_cell_count"] == 48
    assert report["max_bounds_mismatch_m"] == 0
    assert report["bounds_m"] == expected["bounds_m"]
    assert set(report["boundaries"]) == set(import_checks.BOUNDARY_ROLES)
    assert all(item["face_count"] == 2 for item in report["boundaries"].values())
    solver.settings.solution.initialization.initialize.assert_not_called()


@pytest.mark.parametrize(
    "defect",
    [
        "missing_wall",
        "aliased_wall",
        "empty_wall",
        "invalid_count",
        "missing_cells",
        "wrong_units",
    ],
)
def test_import_audit_rejects_incomplete_or_incorrectly_scaled_volume(
    imported_mesh, defect
):
    solver, zones, expected, data = imported_mesh
    if defect == "missing_wall":
        zones.pop("shroud")
    elif defect == "aliased_wall":
        zones["shroud"] = zones["hub"]
    elif defect == "empty_wall":
        data[zones["shroud"]].vertices = np.empty((0, 3))
    elif defect == "invalid_count":
        solver.fields.reduction.count.side_effect = lambda locations: 0
    elif defect == "missing_cells":
        expected["cell_count"] = 49
    else:
        for surface in data.values():
            surface.vertices *= 1000
    with pytest.raises(RuntimeError):
        import_checks.validate_import(solver, zones, expected)


@pytest.mark.parametrize("enabled", [False, True])
def test_yplus_reports_separate_blade_hub_and_shroud_statistics(monkeypatch, enabled):
    options, _ = workflow.read_workflow(WORKFLOW_FILE)
    options = options["fluent"]
    options["wall_yplus"] = enabled
    common_reports = {"relative_mass_imbalance": 1e-6}
    monkeypatch.setattr(
        workflow.common, "solution_diagnostics", Mock(return_value=common_reports)
    )
    solver = MagicMock()
    solver.settings.setup.boundary_conditions.wall.get_object_names.return_value = [
        "prefix-blade",
        "prefix-hub",
        "prefix-shroud",
    ]
    reports = solver.settings.solution.report_definitions
    reports.compute.return_value = [
        {
            f"workflow-{stat}-yplus-{wall}": [value]
            for stat, value in (("mean", 0.8), ("max", 1.6))
            for wall in ("blade", "hub", "shroud")
        }
    ]
    result = workflow.solution_diagnostics(solver, options)
    assert result["relative_mass_imbalance"] == 1e-6
    if enabled:
        for wall in ("blade", "hub", "shroud"):
            assert result[f"mean-yplus-{wall}"] == 0.8
            assert result[f"max-yplus-{wall}"] == 1.6
        definitions = [
            call.args[1] for call in reports.surface.__setitem__.call_args_list
        ]
        assert len(definitions) == 6
        for call in reports.surface.__setitem__.call_args_list:
            report_name, definition = call.args
            expected_type = (
                "surface-areaavg"
                if report_name.startswith("workflow-mean-")
                else "surface-facetmax"
            )
            assert definition["report_type"] == expected_type
        assert {tuple(item["surface_names"]) for item in definitions} == {
            ("prefix-blade",),
            ("prefix-hub",),
            ("prefix-shroud",),
        }
    else:
        reports.compute.assert_not_called()
        assert result == common_reports


@pytest.mark.parametrize("setup_fails", [False, True], ids=["create", "setup_error"])
def test_case_creation_uses_3d_and_closes_without_solving(
    monkeypatch,
    tmp_path,
    setup_fails,
):
    options, _ = workflow.read_workflow(WORKFLOW_FILE)
    options.update(output_dir=tmp_path, run_case=False)
    solver = MagicMock()
    solver.get_fluent_version.return_value = "24.2"
    solver.settings.solution.methods.high_order_term_relaxation.get_state.return_value = (
        {}
    )
    solver.settings.file.write_case.side_effect = lambda file_name: Path(
        file_name
    ).write_bytes(b"configured 3d case")
    configure = Mock(return_value={"inlet": "inlet", "hub": "hub", "shroud": "shroud"})
    if setup_fails:
        configure.side_effect = RuntimeError("invalid periodic pair")
    monkeypatch.setattr(workflow, "configure_fluent", configure)
    pyfluent = SimpleNamespace(launch_fluent=Mock(return_value=solver))
    geometry = {"kind": "axial"}
    if setup_fails:
        with pytest.raises(RuntimeError, match="invalid periodic pair"):
            workflow.save_fluent_case(
                pyfluent,
                options,
                tmp_path / "mesh.cgns",
                {},
                {},
                geometry,
            )
        solver.settings.file.write_case.assert_not_called()
    else:
        case_file = workflow.save_fluent_case(
            pyfluent,
            options,
            tmp_path / "mesh.cgns",
            {},
            {},
            geometry,
        )
        assert case_file.is_file()
        summary = json.loads((case_file.parent / "setup_summary.json").read_text())
        assert summary["dimension"] == 3
        assert summary["initialized"] is False
        assert summary["solution_iterations"] == 0
    assert pyfluent.launch_fluent.call_args.kwargs["dimension"] == 3
    solver.settings.solution.initialization.initialize.assert_not_called()
    solver.settings.solution.run_calculation.iterate.assert_not_called()
    solver.settings.file.write_case_data.assert_not_called()
    solver.exit.assert_called_once()


@pytest.mark.parametrize(
    "prepare_only", [True, False], ids=["prepare_only", "create_only"]
)
def test_main_reuses_one_fresh_nozzle_for_geometry_and_barotropy(
    raw_workflow,
    monkeypatch,
    tmp_path,
    prepare_only,
):
    raw_workflow["design_file"] = str(
        (WORKFLOW_FILE.parent / raw_workflow["design_file"]).resolve()
    )
    raw_workflow.update(
        output_dir="output",
        prepare_only=prepare_only,
        run_case=False,
        show_gmsh=False,
        show_pyvista=False,
        save_mesh_figure=False,
    )
    raw_workflow["geometry_figure"].update(save=False, show=False)
    config_file = tmp_path / "workflow.yaml"
    config_file.write_text(yaml.safe_dump(raw_workflow), encoding="utf-8")
    _, config = workflow.read_workflow(config_file)
    nozzle = {"converged": True, "meta": {"Q0": 0.025}}
    solve = Mock(return_value=nozzle)
    monkeypatch.setattr(workflow, "solve_nozzle", solve)
    mapping = MeridionalChannel(geometry_from_machine(config["meridional_channel"]))
    design = SimpleNamespace(meridional_channel=mapping)
    build = Mock(return_value=design)
    monkeypatch.setattr(workflow.geo, "build_design", build)
    section = object()
    reference = Mock()
    volume = Mock()
    export = Mock()
    monkeypatch.setattr(workflow.mesh3, "reference_mesh", reference)
    monkeypatch.setattr(
        workflow.mesh3, "load_section", Mock(return_value=(section, design))
    )
    monkeypatch.setattr(workflow.mesh3, "volume_mesh", volume)
    monkeypatch.setattr(workflow.mesh3, "export_mesh", export)
    monkeypatch.setattr(
        workflow,
        "mesh_expectations",
        Mock(
            return_value={
                "cell_count": 48,
                "bounds_m": [[0, 0, 0], [0.1, 0.1, 0.02]],
            }
        ),
    )
    barotropy = object()
    pyfluent = object()
    dependencies = Mock(return_value=(barotropy, pyfluent))
    monkeypatch.setattr(workflow.common, "load_dependencies", dependencies)
    expressions, state = {"barotropic_density": "1 [kg/m^3]"}, {"P0": 251300.0}
    generate = Mock(return_value=(expressions, state))
    monkeypatch.setattr(workflow, "generate_barotropic_model", generate)
    save = Mock(return_value=tmp_path / "output/fluent/setup.cas.h5")
    monkeypatch.setattr(workflow, "save_fluent_case", save)

    workflow.main(config_file)

    solve.assert_called_once_with(config["method_of_characteristics"])
    assert build.call_count == 1 and build.call_args.args[0] is nozzle
    assert generate.call_count == 1 and generate.call_args.args[2] is nozzle
    assert generate.call_args.args[0] is barotropy
    assert generate.call_args.args[1]["nozzle"] == config["method_of_characteristics"]
    reference.assert_called_once()
    volume.assert_called_once()
    assert volume.call_args.args[0] is section
    assert volume.call_args.args[1] is design
    export.assert_called_once()
    dependencies.assert_called_once_with(prepare_only)
    if prepare_only:
        save.assert_not_called()
    else:
        save.assert_called_once()
        assert save.call_args.args[0] is pyfluent
        assert save.call_args.args[3] is state
        assert save.call_args.args[4] is expressions


@pytest.mark.parametrize("family", ["axial", "radial", "linear"])
def test_prepare_mesh_saves_reference_figures_from_real_3d_geometry(
    raw_workflow, monkeypatch, tmp_path, family
):
    workflow.plt.switch_backend("Agg")
    raw_workflow["design_file"] = str(CASES / f"{family}_stator.yaml")
    options, config = workflow.validate_workflow(raw_workflow, WORKFLOW_FILE)
    assert options["geometry_figure"]["save"] is True
    options.update(show_gmsh=False, show_pyvista=False, save_mesh_figure=False)
    config["blade_geometry"].update(
        r_trailing=0.1, num_baseline_points=100, num_curve_points=100, n_cp=15
    )
    config["flow_domain"]["inlet"]["extension_length"] = 0.5
    config["flow_domain"]["outlet"]["extension_length"] = 0.5
    config["flow_domain"]["number_of_points"] = 40
    if family != "linear":
        config["meridional_channel"]["num_blades"] = 12
    x = np.linspace(0, 0.01, 80)
    nozzle = {
        "converged": True,
        "wall_final": {"x": x, "y": 0.002 + 0.001 * (x / x[-1]) ** 2},
    }
    monkeypatch.setattr(workflow, "solve_nozzle", Mock(return_value=nozzle))
    # Exercise real geometry construction and real Matplotlib output. Meshing
    # is downstream of the failure and is verified separately.
    reference = Mock()
    monkeypatch.setattr(workflow.mesh3, "reference_mesh", reference)
    monkeypatch.setattr(workflow.mesh3, "load_section", Mock(return_value=(None, None)))
    monkeypatch.setattr(workflow.mesh3, "volume_mesh", Mock())
    monkeypatch.setattr(workflow.mesh3, "export_mesh", Mock())
    monkeypatch.setattr(workflow, "mesh_expectations", Mock(return_value={}))
    original_figures = workflow.plt.get_fignums()

    result, _, _ = workflow.prepare_mesh(options, config, tmp_path)

    assert result is nozzle
    reference.assert_called_once()
    design = reference.call_args.args[0]
    # Plotting must not alter the geometry consumed by the volume mesher.
    assert "suction" not in design.curves
    assert "construction_camberline" not in design.curves
    for extension in options["geometry_figure"]["formats"]:
        image = tmp_path / f"stator_reference_geometry{extension}"
        assert image.is_file() and image.stat().st_size > 1000
    svg = (tmp_path / "stator_reference_geometry.svg").read_text(encoding="utf-8")
    assert "Raw suction surface" in svg
    assert "Raw pressure surface" in svg
    assert workflow.plt.get_fignums() == original_figures
