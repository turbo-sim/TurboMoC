"""Regenerate the README figures (docs/images/readme_*.png).

Run from the repository root (needs the cad and mesh extras: cadquery, vtk,
gmsh):

    poetry run python docs/images/make_readme_figures.py

Everything is computed from the package itself: a two-phase Cyclopentane
nozzle, its stator blade, a supersonic vortex rotor, the 2D/3D meshes and
the axial and radial stages. Takes a few minutes (MoC solve and meshing).
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

import turbo_moc
from turbo_moc.app import _mirrored_rotor_blade, _place_radial_rotor, _rotor_z_offset
from turbo_moc.geometry import (
    consistent_radial_wrap,
    export_annular_blade_stl,
    export_radial_blade_stl,
    parametrize_stator_blade,
    size_rotor_from_pitch,
    size_stator_mode_a,
)
from turbo_moc.meshing.stator_mesh_2d import default_mesh_sizes, mesh_sized_stator
from turbo_moc.meshing.stator_mesh_3d import load_patch_surfaces, mesh_swept_passage
from turbo_moc.rotor import design_rotor_vortex_blade

OUT = Path(__file__).resolve().parent
STATOR_COLOR, ROTOR_COLOR = (0.36, 0.45, 0.58), (0.80, 0.47, 0.32)
BACKGROUND = (1.0, 1.0, 1.0)


# ----------------------------------------------------------------- design
nozzle = turbo_moc.design_nozzle(fluid_name="Cyclopentane", P0=2.513e5, Q0=0.025, p_back=0.95e5,
                                 solver="conventional", use_true_sauer_line=True)
wall = nozzle["wall_final"]
blade = parametrize_stator_blade(wall["x"], wall["y"], metal_angle_in=0.0, metal_angle_out=70.0,
                                 r_trailing=0.5, inlet_opening_ratio=1.5)
stator = size_stator_mode_a(blade, nozzle, mdot=2.0, n_blades=24, r_hub=100.0)
rotor_data = design_rotor_vortex_blade(fluid_name="Novec649", P0_rel=5e5, T0_rel=400.0, M_inlet=1.5,
                                       M_outlet=1.5, M_lower=1.05, M_upper=2.0, beta_inlet=65.0,
                                       backend="HEOS", num_points=60)
rotor = size_rotor_from_pitch(rotor_data, stator["r_hub"], stator["r_shroud"], 40)


# ----------------------------------------------------------------- 2D figures
def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.9, 1]})
turbo_moc.mpl.plot_characteristics_mesh(nozzle, ax=axes[0], mirror=True)
turbo_moc.mpl.plot_ts_diagram(nozzle, ax=axes[1])
axes[0].set_title("Method of characteristics: two-phase Cyclopentane nozzle")
axes[1].set_title("Expansion in the T-s diagram")
save(fig, "readme_nozzle.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
turbo_moc.mpl.plot_blade(blade, ax=axes[0], n_blades=2, show_control_points=False)
turbo_moc.mpl.plot_rotor_vortex_blade(rotor_data, ax=axes[1], show_surfaces=False)
axes[0].set_title("Stator blade from the nozzle wall")
axes[1].set_title("Supersonic vortex rotor blade")
save(fig, "readme_blades.png")

scaled = stator["scaled_blade"]
pitch = scaled["pitch"]
sizes = {**default_mesh_sizes(pitch), "blade_size": pitch / 90, "domain_size": pitch / 25,
         "first_height": pitch / 3000, "num_layers": 12, "growth_ratio": 1.25}
mesh = mesh_sized_stator(scaled, sizes, topology="structured")
xy = np.asarray(mesh["xy"])
segments = np.array([xy[[a - 1, b - 1]] for cell in mesh["cells"] for a, b in zip(cell, cell[1:] + cell[:1])])
from matplotlib.collections import LineCollection  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.15, 1]})
contour = np.column_stack((scaled["blade_curve"]["x"] + scaled["trailing_edge"]["x"],
                           scaled["blade_curve"]["y"] + scaled["trailing_edge"]["y"]))
te = np.mean(np.column_stack((scaled["trailing_edge"]["x"], scaled["trailing_edge"]["y"])), axis=0)
for ax, window in zip(axes, (None, 6 * scaled["r_trailing"])):
    ax.add_collection(LineCollection(segments, lw=0.25 if window is None else 0.5, color="#e08a2c"))
    for shift in (-pitch, 0.0, pitch):
        ax.fill(contour[:, 0] + shift, contour[:, 1], color="#3d4a5c")
    ax.set_aspect("equal")
    if window is None:
        ax.set_xlim(xy[:, 0].min() - 0.6 * pitch, xy[:, 0].max() + 0.2 * pitch)
        ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
        ax.set_title(f"{mesh['topology'].capitalize()} 2D passage mesh")
    else:
        ax.set_xlim(te[0] - window, te[0] + window)
        ax.set_ylim(te[1] - window, te[1] + window)
        ax.set_title("Trailing edge: y+-based inflation layers")
    ax.set_xlabel("Pitchwise (mm)")
    ax.set_ylabel("Axial (mm)")
save(fig, "readme_mesh_2d.png")


# ----------------------------------------------------------------- 3D renders
def read_stl(path):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def rotated(poly, degrees):
    transform = vtk.vtkTransform()
    transform.RotateZ(degrees)
    out = vtk.vtkTransformPolyDataFilter()
    out.SetTransform(transform)
    out.SetInputData(poly)
    out.Update()
    return out.GetOutput()


def patch_poly(surface):
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk.vtkPoints())
    poly.GetPoints().SetData(numpy_to_vtk(np.asarray(surface["points"], dtype=float), deep=True))
    cells = vtk.vtkCellArray()
    cells.ImportLegacyFormat(numpy_to_vtkIdTypeArray(np.asarray(surface["polys"], dtype=np.int64), deep=True))
    poly.SetPolys(cells)
    return poly


def render(actors, name, azimuth=0, elevation=0, zoom=1.0, size=(1000, 800), camera_at=None):
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(*BACKGROUND)
    for poly, color, edges in actors:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(*color)
        prop.SetSpecular(0.35)
        prop.SetSpecularPower(25)
        if edges:
            prop.EdgeVisibilityOn()
            prop.SetEdgeColor(0.15, 0.15, 0.15)
            prop.SetLineWidth(0.6)
        renderer.AddActor(actor)
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(*size)
    window.AddRenderer(renderer)
    renderer.ResetCamera()
    camera = renderer.GetActiveCamera()
    if camera_at is not None:  # (position, focal point, view up)
        camera.SetPosition(*camera_at[0])
        camera.SetFocalPoint(*camera_at[1])
        camera.SetViewUp(*camera_at[2])
    camera.Azimuth(azimuth)
    camera.Elevation(elevation)
    camera.Zoom(zoom)
    renderer.ResetCameraClippingRange()
    window.Render()
    grab = vtk.vtkWindowToImageFilter()
    grab.SetInput(window)
    grab.Update()
    writer = vtk.vtkPNGWriter()
    path = Path(tempfile.gettempdir()) / name
    writer.SetFileName(str(path))
    writer.SetInputConnection(grab.GetOutputPort())
    writer.Write()
    return path


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    # Axial stage: stator row + rotor row on the shared annulus.
    export_annular_blade_stl(scaled, stator["r_hub"], stator["r_shroud"], tmp / "stator.stl",
                             n_blades=stator["n_blades"], flare="pitch_scale", source="stator")
    z_offset = _rotor_z_offset(stator, rotor, 0.25 * pitch)
    export_annular_blade_stl(_mirrored_rotor_blade(rotor["scaled_blade"]), rotor["r_hub"], rotor["r_shroud"],
                             tmp / "rotor.stl", n_blades=rotor["n_blades"], flare="pitch_scale",
                             source="rotor", z_offset=z_offset)
    stator_stl, rotor_stl = read_stl(tmp / "stator.stl"), read_stl(tmp / "rotor.stl")
    actors = [(rotated(stator_stl, k * 360 / stator["n_blades"]), STATOR_COLOR, False)
              for k in range(stator["n_blades"])]
    actors += [(rotated(rotor_stl, k * 360 / rotor["n_blades"]), ROTOR_COLOR, False)
               for k in range(rotor["n_blades"])]
    axial_png = render(actors, "axial_stage.png", azimuth=0, elevation=-55, zoom=1.35)

    # Swept 3D stator mesh: one meshed passage + neighbouring blades.
    height = stator["r_shroud"] - stator["r_hub"]
    span = {"first_height": height / 400, "growth_ratio": 1.3, "num_layers": 6, "transition_ratio": 1.3,
            "core_size": height / 8}
    coarse = mesh_sized_stator(scaled, {**sizes, "blade_size": pitch / 50, "domain_size": pitch / 12,
                                        "num_layers": 6, "first_height": pitch / 800})
    swept = mesh_swept_passage(coarse, stator["r_hub"], stator["r_shroud"], stator["n_blades"], span, tmp / "m3")
    patches = load_patch_surfaces(swept["output_dir"], ["blade", "hub", "periodic_left"])
    actors = [(patch_poly(patches["hub"]), (0.93, 0.62, 0.30), True),
              (patch_poly(patches["periodic_left"]), (0.98, 0.80, 0.55), True),
              (patch_poly(patches["blade"]), (0.30, 0.33, 0.38), True)]
    actors += [(rotated(stator_stl, k * 360 / stator["n_blades"]), STATOR_COLOR, False) for k in (-1, 1)]
    hub_points = np.asarray(patches["hub"]["points"], dtype=float)
    focal = hub_points.mean(axis=0)
    outward = np.r_[focal[:2] / np.linalg.norm(focal[:2]), 0.0]
    extent = np.ptp(hub_points, axis=0).max()
    position = focal + 1.6 * extent * outward + 0.9 * extent * np.array([0.0, 0.0, 1.0])
    mesh_png = render(actors, "mesh_3d.png", azimuth=20, camera_at=(position, focal, (0, 0, 1)), zoom=1.1)

    # Radial stage: consistent N and r2 for both rows (the designed passages).
    s_wrap = consistent_radial_wrap(blade, 150.0, 300.0, source="stator")
    r_wrap = consistent_radial_wrap(rotor_data, 100.0, 200.0, source="rotor")
    stator_radial = {"r1": 150.0, "r2": s_wrap["r2"], "n_blades": s_wrap["n_blades"], "depth": 8.0}
    rotor_radial = _place_radial_rotor(stator_radial, {"r1": 100.0, "r2": r_wrap["r2"],
                                                       "n_blades": r_wrap["n_blades"], "depth": 8.0}, 6.0)
    export_radial_blade_stl(blade, stator_radial["r1"], stator_radial["r2"], 8.0, tmp / "rs.stl", source="stator")
    export_radial_blade_stl(rotor_data, rotor_radial["r1"], rotor_radial["r2"], 8.0, tmp / "rr.stl",
                            source="rotor")
    rs, rr = read_stl(tmp / "rs.stl"), read_stl(tmp / "rr.stl")
    actors = [(rotated(rs, k * 360 / s_wrap["n_blades"]), STATOR_COLOR, False) for k in range(s_wrap["n_blades"])]
    actors += [(rotated(rr, k * 360 / r_wrap["n_blades"]), ROTOR_COLOR, False) for k in range(r_wrap["n_blades"])]
    radial_png = render(actors, "radial_stage.png", azimuth=0, elevation=-50, zoom=1.25)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, path, title in zip(axes, (axial_png, mesh_png, radial_png),
                           ("Axial stage (stator + rotor rows)", "Swept 3D passage mesh (hex/prism)",
                            "Radial stage (log-spiral wrap)")):
    image = plt.imread(path)
    rows, cols = np.where(image[..., :3].min(axis=2) < 0.98)  # crop the white margin
    ax.imshow(image[max(rows.min() - 10, 0):rows.max() + 10, max(cols.min() - 10, 0):cols.max() + 10])
    ax.set_title(title)
    ax.axis("off")
save(fig, "readme_3d.png")
