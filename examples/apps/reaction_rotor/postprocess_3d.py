"""Simple 3D post-processing of a Fluent stage (stator + rotor) with PyVista: only the case and data files are needed.

    python postprocess_3d.py path/to/stage_fluent.cas.h5 [--rpm 5000] [--out figures]

The data file (same name, .dat.h5) must sit next to the case file. Writes into --out:
  - blade_to_blade_<span>.png : Mach number and pressure on unwrapped cylinders at 10, 50 and 90 % span
                                (absolute Mach in the stator, relative Mach in the rotor);
  - axial_planes.csv          : mass-averaged quantities on axial planes through the domain;
  - view_3d.png               : 3D view of the pressure on the cell zones.

Assumptions (check them against your case):
  - axial direction +z, rotation about z, rotor speed from --rpm (sign: the rotor moves towards +theta for rpm > 0);
  - the velocities stored by Fluent are absolute (default absolute-velocity formulation);
  - the operating pressure is 0, so the stored pressure is absolute (the TurboMoC cases use 0);
  - barotropic fluid: p depends on rho only, so the speed of sound is a^2 = dp/drho, taken from the data itself.
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import vtk


def read_case(case_file):
    """Cell zones of the case with p, rho and the absolute velocity, as point data (smooth contours)."""
    blocks = pv.FLUENTCFFReader(str(case_file)).read()
    zones = [b for b in blocks if b is not None and b.n_cells > 0]
    out = []
    for zone in zones:
        keep = ("SV_P", "SV_DENSITY", "SV_U", "SV_V", "SV_W")  # Fluent writes some empty arrays: they crash VTK
        for name in list(zone.cell_data.keys()):
            if name not in keep:
                del zone.cell_data[name]
        zone = zone.extract_cells(np.arange(zone.n_cells))  # Own points only (the reader shares one point list)
        z = zone.cell_data_to_point_data()
        z.point_data["p"] = z.point_data["SV_P"]
        z.point_data["rho"] = z.point_data["SV_DENSITY"]
        z.point_data["velocity"] = np.column_stack([z.point_data["SV_U"], z.point_data["SV_V"], z.point_data["SV_W"]])
        out.append(z)
    out.sort(key=lambda m: -float(np.nanmax(m.point_data["p"])))  # Stator first: it holds the inlet total pressure
    return out


def speed_of_sound_model(zones):
    """a(rho) from a smooth fit of p against rho over all cells (barotropic: p = p(rho))."""
    rho = np.concatenate([z.point_data["rho"] for z in zones])
    p = np.concatenate([z.point_data["p"] for z in zones])
    ok = np.isfinite(rho) & np.isfinite(p) & (rho > 0) & (p > 0)
    rho, p = rho[ok], p[ok]
    pick = np.random.default_rng(0).choice(len(rho), min(len(rho), 200000), replace=False)
    fit = np.polyfit(np.log(rho[pick]), np.log(p[pick]), 4)  # log p against log rho: smooth
    floor = float(rho.min())

    def a(r):
        lr = np.log(np.maximum(r, floor))
        dlogp = np.polyval(np.polyder(fit), lr)
        return np.sqrt(np.maximum(dlogp * np.exp(np.polyval(fit, lr)) / np.exp(lr), 1e-12))
    return a


def add_kinematics(zone, omega, rotating, a_of_rho):
    x, y = zone.points[:, 0], zone.points[:, 1]
    r, theta = np.hypot(x, y), np.arctan2(y, x)
    c = zone.point_data["velocity"]
    c_r = c[:, 0] * np.cos(theta) + c[:, 1] * np.sin(theta)
    c_t = -c[:, 0] * np.sin(theta) + c[:, 1] * np.cos(theta)
    c_z = c[:, 2]
    w_t = c_t - (omega * r if rotating else 0.0)
    a = a_of_rho(zone.point_data["rho"])
    zone.point_data.update({"r": r, "theta": theta, "c_z": c_z, "c_t": c_t, "c_r": c_r, "w_t": w_t, "a": a,
                            "Mach": np.sqrt(c_z ** 2 + c_t ** 2 + c_r ** 2) / a,
                            "Mach_rel": np.sqrt(c_z ** 2 + w_t ** 2 + c_r ** 2) / a})


def cylinder_cut(zone, radius):
    cyl = vtk.vtkCylinder()
    cyl.SetRadius(radius)
    cyl.SetAxis(0, 0, 1)
    return zone.slice_implicit(cyl)


def blade_to_blade(zones, span, out):
    r_hub = min(np.percentile(z.point_data["r"], 0.1) for z in zones)
    r_tip = max(np.percentile(z.point_data["r"], 99.9) for z in zones)
    radius = r_hub + span * (r_tip - r_hub)
    fig, axes = plt.subplots(2, 1, figsize=(7, 9), sharex=True, constrained_layout=True)
    for row, zone in enumerate(zones):
        cut = cylinder_cut(zone, radius)
        if cut.n_points == 0:
            continue
        tri = cut.triangulate()
        faces = tri.faces.reshape(-1, 4)[:, 1:]
        z_mm = tri.points[:, 2] * 1e3
        arc_mm = np.unwrap(np.arctan2(tri.points[:, 1], tri.points[:, 0])) * radius * 1e3
        mach = tri.point_data["Mach_rel" if row == 1 else "Mach"]
        for ax, values, label in ((axes[0], mach, "Mach (stator abs., rotor rel.)"),
                                  (axes[1], tri.point_data["p"] / 1e3, "p [kPa]")):
            tc = ax.tripcolor(z_mm, arc_mm, faces, values, shading="gouraud", cmap="viridis")
            if row == len(zones) - 1:
                fig.colorbar(tc, ax=ax, label=label)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_ylabel(r"$r\theta$ [mm]")
    axes[1].set_xlabel("z [mm]")
    fig.suptitle(f"{int(round(span * 100))} % span (r = {radius * 1e3:.1f} mm)")
    fig.savefig(out / f"blade_to_blade_{int(round(span * 100)):02d}.png", dpi=200)
    plt.close(fig)


def axial_planes(zones, out, count=12):
    rows = []
    for row, zone in enumerate(zones):
        z0, z1 = zone.bounds[4], zone.bounds[5]
        for z in np.linspace(z0, z1, count + 2)[1:-1]:
            cut = zone.slice(normal="z", origin=(0, 0, z)).triangulate().compute_cell_sizes(length=False, volume=False)
            if cut.n_cells == 0:
                continue
            cells = cut.point_data_to_cell_data()
            area = cells.cell_data["Area"]
            flux = cells.cell_data["rho"] * cells.cell_data["c_z"] * area
            m = flux.sum()
            avg = {k: float(np.sum(flux * cells.cell_data[k]) / m) for k in ("p", "Mach", "Mach_rel", "c_z", "c_t")}
            rows.append(dict(zone=("stator", "rotor")[row] if len(zones) == 2 else row, z_mm=1e3 * z,
                             mass_flow_passage=float(m), **avg))
    with open(out / "axial_planes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['zone']:>6} z {r['z_mm']:7.2f} mm  m {r['mass_flow_passage']:.4f} kg/s  p {r['p'] / 1e3:7.2f} kPa  "
              f"M {r['Mach']:.3f}  M_rel {r['Mach_rel']:.3f}")


def view_3d(zones, out):
    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1000))
    for zone in zones:
        plotter.add_mesh(zone.extract_surface(), scalars="p", cmap="viridis", scalar_bar_args={"title": "p [Pa]"})
    plotter.view_isometric()
    plotter.screenshot(str(out / "view_3d.png"))
    plotter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", help="Fluent case file (.cas.h5); the .dat.h5 must be next to it")
    parser.add_argument("--rpm", type=float, default=0.0, help="rotor speed [rpm] (relative Mach in the rotor)")
    parser.add_argument("--out", default="figures_3d", help="output folder")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    zones = read_case(args.case)
    print(f"{len(zones)} cell zones, {sum(z.n_cells for z in zones)} cells")
    a_of_rho = speed_of_sound_model(zones)
    omega = args.rpm * 2 * np.pi / 60
    for row, zone in enumerate(zones):
        add_kinematics(zone, omega, rotating=(row == len(zones) - 1 and len(zones) > 1), a_of_rho=a_of_rho)
    for span in (0.1, 0.5, 0.9):
        blade_to_blade(zones, span, out)
    axial_planes(zones, out)
    view_3d(zones, out)
    print("written to", out.resolve())


if __name__ == "__main__":
    main()
