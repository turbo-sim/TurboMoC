"""Shared 3D display style for the app (vtk.js) and desktop scripts (PyVista).

One palette, lighting, camera convention and display-copy transform. Display
copies and the annular display frame are rendering transforms only; mesh and
geometry coordinates are never modified.
"""
import numpy as np

SURFACE_COLORS = {"blade": "#477db3", "hub": "#a6b3b8", "shroud": "#a6b3b8", "inlet": "#c94f4f",
                  "outlet": "#c94f4f", "periodic_left": "#4d9966", "periodic_right": "#4d9966"}
ROTOR_BLADE_COLOR = "#ef8c26"
CONTEXT_COLOR = "#8f9ca6"
EDGE_COLOR = "#000000"
BACKGROUND = "#fafbfc"
LIGHTING = {"ambient": 0.25, "diffuse": 0.75}
QUALITY_STOPS = ((35, 155, 86), (244, 208, 63), (192, 57, 43))
QUALITY_MISSING = (137, 148, 156)
CAMERA = {"axial": {"direction": (-1.0, 0.12, 0.20), "up": (0.0, 1.0, 0.0)},
          "radial": {"direction": (0.0, 0.0, 1.0), "up": (0.0, 1.0, 0.0)},
          "linear": {"direction": (1.0, 1.0, 1.0), "up": (0.0, 0.0, 1.0)}}


def surface_color(name, role="stator"):
    if name == "blade" and role == "rotor":
        return ROTOR_BLADE_COLOR
    return SURFACE_COLORS.get(name, "#ffffff")


def rgb(color):
    return [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]


def camera(family):
    return CAMERA.get(family, CAMERA["axial"])


def copy_transform(index, *, is_linear, num_blades=None, pitch_mm=None, placement=None):
    """Display transform of passage copy ``index``, identical to the app's actor transform."""
    placement = placement or {}
    result = np.eye(4)
    if is_linear:
        result[1, 3] = index * float(pitch_mm)
        return result
    angle = np.deg2rad(180.0 + float(placement.get("angular_phase_deg", 0.0)) - 360.0 * index / int(num_blades))
    result[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    result[2, 3] = float(placement.get("streamwise_offset_mm", 0.0))
    return result


def quality_rgb(values):
    """Green-to-red badness colours of the app's inspection legend (0 good, 1 bad)."""
    values = np.asarray(values, dtype=float)
    stops = np.asarray(QUALITY_STOPS, dtype=float)
    x = np.clip(np.nan_to_num(values, nan=0.0), 0, 1) * 2
    segment = np.minimum(1, np.floor(x)).astype(int)
    fraction = (x - segment)[:, None]
    colors = np.rint(stops[segment] + fraction * (stops[segment + 1] - stops[segment]))
    colors[~np.isfinite(values)] = QUALITY_MISSING
    return colors.astype(np.uint8)

