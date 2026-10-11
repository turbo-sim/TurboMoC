"""
turbo_moc.geometry.annular -- true 3D annular wrap of an axial blade row,
from a hub radius to a shroud radius.

Unlike turbo_moc.geometry.radial (which maps a blade's CHORDWISE position to
radius, for radial-flow-style visualization), this module keeps the blade's
axial (chordwise) stations fixed across span and instead sweeps a SPAN
parameter r from r_hub to r_shroud, wrapping the blade's pitchwise extent
circumferentially at each radius:

    z_i     = y                          (axial, unchanged across span)
    x_i     = x * (r_i / r_hub)          if flare == "pitch_scale"
            = x                          if flare == "constant"
    theta_i = theta0 + x_i / r_i         (arc length = r_i * theta)
    X_i, Y_i = r_i * cos(theta_i), r_i * sin(theta_i)

"pitch_scale" matches the convention already used by
turbo_moc.geometry.blade.export_flared_blade_step (pitch grows
proportionally with radius at constant blade count); under it theta_i is
constant across span, i.e. a straight radial-fiber blade -- the standard
simplification, not something hard-coded here.

Blade axis convention (see turbo_moc.geometry.radial's module docstring):
a stator's blade_curve/trailing_edge have x = pitchwise, y = chordwise
(after the final profile rotation); a rotor's "blade" has x = chordwise,
y = pitchwise.
"""

import numpy as np

__all__ = ["wrap_blade_annular"]


def _span_wrap(x, y, r_hub, r_shroud, n_span, flare, theta0, chordwise_axis):
    """3D-wrap ONE curve (x, y) across n_span stations from r_hub to
    r_shroud. Returns (r_vals, X, Y, Z), each shape (n_span, len(x))."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    chord, pitch = (y, x) if chordwise_axis == "y" else (x, y)

    r_vals = np.linspace(r_hub, r_shroud, n_span)
    if flare == "pitch_scale":
        pitch_i = pitch[None, :] * (r_vals[:, None] / r_hub)
    elif flare == "constant":
        pitch_i = np.broadcast_to(pitch[None, :], (n_span, pitch.size))
    else:
        raise ValueError(f"flare must be 'pitch_scale' or 'constant', got {flare!r}")

    # Minus sign: increasing pitchwise coordinate sweeps theta in the
    # negative (clockwise, viewed from +Z) direction -- matches the
    # machine's actual circumferential sense; a "+" here wrapped the
    # blade the wrong way round the axis.
    theta_i = theta0 - pitch_i / r_vals[:, None]
    X = r_vals[:, None] * np.cos(theta_i)
    Y = r_vals[:, None] * np.sin(theta_i)
    Z = np.broadcast_to(chord[None, :], (n_span, chord.size)).copy()
    return r_vals, X, Y, Z


def _rotate_copies(X, Y, Z, n_blades, n_draw):
    """Rotate an (n_span, n_pts) (X, Y, Z) triple about the Z axis into
    n_draw copies (n_draw <= n_blades), spaced 2*pi/n_blades apart -- the
    TRUE physical spacing always comes from n_blades, even when only a
    handful of copies are actually drawn (a preview-performance cap must
    never change how tightly packed the blades look)."""
    d_theta = 2.0 * np.pi / n_blades
    copies = []
    for k in range(n_draw):
        c, s = np.cos(k * d_theta), np.sin(k * d_theta)
        copies.append((X * c - Y * s, X * s + Y * c, Z))
    return copies, d_theta


def wrap_blade_annular(blade_data, r_hub, r_shroud, n_blades=1, flare="pitch_scale",
                        n_span=13, theta0=0.0, source="stator", n_preview=None):
    """
    Wrap a Phase-2-style stator blade (parametrize_stator_blade/_semi's
    output) or a Phase-3-style rotor blade (design_rotor_vortex_blade's
    output) onto a true annulus from r_hub to r_shroud: the blade curves
    circumferentially at every span station, not just a flat hub/tip loft.

    Parameters
    ----------
    blade_data : dict
        Stator: needs "blade_curve" and "trailing_edge" ({"x", "y"} each,
        mm). Rotor: needs "blade" ({"x", "y"}).
    r_hub, r_shroud : float
        Hub/shroud radii (same length units as blade_data, mm).
    n_blades : int
        TRUE number of blades around the full annulus -- sets the angular
        spacing (2*pi/n_blades) between copies. Changing this always
        changes how tightly packed the returned copies are, even if only
        a few of them are actually returned (see n_preview).
    flare : str
        "pitch_scale" (pitch grows with radius, matching
        export_flared_blade_step's convention) or "constant" (profile
        unchanged across span).
    n_span : int
        Number of span stations from r_hub to r_shroud (>= 2).
    theta0 : float
        Angular offset [rad] of the blade's own reference (index-0) copy.
    source : str
        "stator" or "rotor" -- which blade_data shape to expect.
    n_preview : int or None
        Cap on how many of the n_blades copies to actually compute/return
        (e.g. for a responsive Mesh3d preview) -- purely a rendering cap,
        NEVER changes d_theta/the physical spacing above. None (default)
        returns all n_blades copies.

    Returns
    -------
    dict
        "curve": {"r", "x", "y", "z"} each shape (n_drawn, n_span, n_pts)
        -- the main blade contour (blade_curve for stator, blade for rotor).
        "trailing_edge": same shape, or None for a rotor (its TE is part of
        the closed "blade" contour already).
        "r_hub"/"r_shroud"/"n_blades"/"n_span"/"flare"/"units" echoed back
        ("n_blades" is always the TRUE count, not n_drawn).
    """
    if n_span < 2:
        raise ValueError(f"n_span must be >= 2, got {n_span}")
    n_draw = n_blades if n_preview is None else min(n_blades, n_preview)

    if source == "stator":
        curve = blade_data["blade_curve"]
        te = blade_data["trailing_edge"]
        chordwise_axis = "y"
    elif source == "rotor":
        curve = blade_data["blade"]
        te = None
        chordwise_axis = "x"
    else:
        raise ValueError(f"source must be 'stator' or 'rotor', got {source!r}")

    def _wrap(cv):
        r_vals, X, Y, Z = _span_wrap(cv["x"], cv["y"], r_hub, r_shroud, n_span,
                                      flare, theta0, chordwise_axis)
        copies, _ = _rotate_copies(X, Y, Z, n_blades, n_draw)
        Xs = np.stack([c[0] for c in copies])
        Ys = np.stack([c[1] for c in copies])
        Zs = np.stack([c[2] for c in copies])
        r_full = np.broadcast_to(r_vals[None, :, None], Xs.shape)
        return {"r": r_full, "x": Xs, "y": Ys, "z": Zs}

    result = {
        "curve": _wrap(curve),
        "trailing_edge": _wrap(te) if te is not None else None,
        "r_hub": float(r_hub), "r_shroud": float(r_shroud),
        "n_blades": int(n_blades), "n_drawn": int(n_draw), "n_span": int(n_span), "flare": flare,
        "units": blade_data.get("units", "mm"),
    }
    return result
