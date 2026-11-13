"""Section-wise rotor blades in the 3D sweep (TurboExpo 3D design (b), 2026-11-12).

The span sweep normally scales one reference section. With ``rotor_design.section_profiles`` = {"hub": json, "mid": json,
"tip": json} (2D ``rotor_result.json`` files of the vortex rotors designed at the local U/c), the blade at span fraction
eta is the point-wise quadratic (Lagrange) blend of the three profiles, each at its own natural pitch-to-chord ratio for
the row's pitch. Profiles are put in the conformal plane exactly as ``rotor_blade.build_rotor_design`` does (LE origin),
scaled so that their native pitch equals the row pitch, stacked with a common LE meridional station (the rotor inlet
interface strip keeps its length) and a common tangential centroid. The midspan profile must reproduce the reference
blade. Wall nodes move to the blended profile at the same arc-length fraction; the protected boundary-layer collar moves
with its nearest wall node, and ``CoreDeformation`` spreads the displacement into the core.
"""
import json

import numpy as np
from scipy.spatial import cKDTree

from turbo_moc.geometry import passage_2d as planar

POINTS = 4000
EDGE_RIGID, EDGE_BLEND = 0.3, 1.5  # mm (conformal): rigid edge region, end of the blend to the point-wise map
SPANS = {"hub": 0.0, "mid": 0.5, "tip": 1.0}


def _closed_resample(q, count=POINTS):
    """Closed contour, counter-clockwise, LE (largest meridional coordinate) at sample 0 and TE (smallest) at sample
    count / 2; each side is resampled by its own arc length so the edges map onto each other between profiles
    (a single arc-length fraction around the whole contour drifted by tenths of a mm at the thin TE and folded it)."""
    q = np.asarray(q, float)
    if np.allclose(q[0], q[-1]):
        q = q[:-1]
    area = 0.5 * np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1])
    if area < 0:
        q = q[::-1]
    q = np.roll(q, -np.argmax(q[:, 1]), axis=0)
    te = int(np.argmin(q[:, 1]))
    sides = (q[:te + 1], np.vstack([q[te:], q[:1]]))
    half = count // 2
    out = []
    for side in sides:
        s = np.r_[0, np.cumsum(np.hypot(*np.diff(side, axis=0).T))]
        t = np.linspace(0, s[-1], half + 1)[:-1]
        out.append(np.column_stack((np.interp(t, s, side[:, 0]), np.interp(t, s, side[:, 1]))))
    return np.vstack(out)


def _conformal(result, pitch):
    native = planar.xy(result["blade"])
    q = np.column_stack((native[:, 1], -native[:, 0]))
    q = (q - q[np.argmax(q[:, 1])]) * (pitch / float(result["pitch"]))
    return _closed_resample(q)


def _stator_conformal(path, config, chord):
    """Stator section from its MoC result (``moc_result.json``, stator design option B of 2026-11-13): the blade
    template of the row's own blade_geometry on the displaced wall, LE at the origin, scaled to the reference
    conformal chord exactly as ``meridional_channel.build_section`` scales the reference stator."""
    from turbo_moc.geometry.stator_blade import parametrize_stator
    from turbo_moc.geometry.stator_boundary_layer import displaced_nozzle
    nozzle = displaced_nozzle(json.loads(open(path, encoding="utf-8").read()), config)
    wall = nozzle["wall_final"]
    contour = planar.blade_contour(parametrize_stator(wall["x"], wall["y"], **config["blade_geometry"]))
    q = (contour - contour[np.argmax(contour[:, 1])]) * (chord / np.ptp(contour[:, 1]))
    return _closed_resample(q)


class SectionMorph:
    def __init__(self, profiles, section, design, config=None):
        pitch = float(design.meridional_channel.pitch)
        reference = _closed_resample(planar.xy(design.blade["blade_curve"]))
        self.stator = str(profiles["mid"]).endswith("moc_result.json")
        if self.stator:  # Stator sections (MoC results)
            chord = float(np.ptp(reference[:, 1]))
            raw = {k: _stator_conformal(profiles[k], config, chord) for k in SPANS}
            # Shape differences only, on the exact reference: the re-fitted midspan template sat 0.09 mm off the
            # reference blade and that offset inverted cells at every span (2026-11-13). Hub/tip = reference +
            # (own section - own midspan), so the midspan is the reference and the stacking is unchanged.
            mid_raw = raw["mid"] - raw["mid"][0] + reference[0]
            loaded = {k: reference + (raw[k] - raw[k][0] + reference[0] - mid_raw) for k in SPANS}
        else:
            loaded = {k: _conformal(json.loads(open(profiles[k], encoding="utf-8").read()), pitch) for k in SPANS}
        # Align the midspan profile on the reference blade (same LE point), then stack the others: LE meridional
        # station and tangential centroid of the midspan profile.
        shift = reference[0] - loaded["mid"][0]
        loaded = {k: v + shift for k, v in loaded.items()}
        self.delta = {k: float(np.max(np.linalg.norm(loaded[k] - loaded["mid"], axis=1))) for k in ("hub", "tip")}
        # Radial stacking through the section centroids (both directions). LE stacking put the whole chord change at
        # the TE (hub chord -10 %: 3.8 mm) and folded the pressure-side core there (2026-11-12); hub and tip sections
        # are shorter than midspan, so their LE moves downstream, away from the short inlet interface strip.
        centre = loaded["mid"].mean(axis=0)
        for k in ("hub", "tip"):
            loaded[k] = loaded[k] - (loaded[k].mean(axis=0) - centre)
            # A section longer than midspan would push its LE upstream into the short inlet interface strip (reacting
            # tip section, chord +4.6 %, folded the strip at 2026-11-13): keep the LE at or behind the midspan LE.
            upstream = loaded[k][0, 1] - loaded["mid"][0, 1]
            if upstream > 0:
                loaded[k] = loaded[k] - [0.0, upstream]
        self.profiles = loaded
        xy = section["xy"]
        self.wall = np.unique(section["boundaries"]["blade"])
        tree = cKDTree(reference)
        gap, self.index = tree.query(xy[self.wall])
        mismatch = float(np.max(np.linalg.norm(loaded["mid"] - reference, axis=1)))
        if gap.max() > 0.02 * pitch or mismatch > 0.02 * pitch:
            raise ValueError(f"Section profiles do not match the reference blade (wall gap {gap.max():.3g}, "
                             f"midspan profile mismatch {mismatch:.3g}, pitch {pitch:.3g}).")
        self.reference = reference
        self.xy = xy
        self.summary = {"midspan_mismatch": mismatch, "wall_gap": float(gap.max()),
                        "chords": {k: float(np.ptp(v[:, 1])) for k, v in loaded.items()}, "max_shape_change": self.delta}

    def blade(self, eta):
        """Piecewise-linear blend hub-mid-tip. The quadratic (Lagrange) blend overshot between the sections: with the
        hub LE 1.3 mm downstream and the tip LE at the midspan station, the LE at eta 0.81 sat 0.15 mm upstream of
        both and folded the inlet strip (reacting design, 2026-11-13)."""
        eta = float(np.clip(eta, 0.0, 1.0))
        if eta <= 0.5:
            w = eta / 0.5
            return (1 - w) * self.profiles["hub"] + w * self.profiles["mid"]
        w = (eta - 0.5) / 0.5
        return (1 - w) * self.profiles["mid"] + w * self.profiles["tip"]

    def displacement(self, eta, scale, offset, collar):
        """Per-node displacement (len(xy) x 2): wall nodes to the blended blade at the same landmark-wise arc-length
        position; collar (boundary-layer stack) nodes rigidly with their nearest wall node. The caller prescribes it
        on the collar and spreads it into the core with the Laplacian (CoreDeformation.deform)."""
        target = self.blade(float(eta)) + np.asarray(offset)  # Edge-stacking offset (stator TE strip; zero for rotors)
        ref = self.reference * scale + np.asarray(offset)
        move = np.zeros_like(self.xy)
        wall = target[self.index] - ref[self.index]
        # Edges move rigidly with their landmark (LE sample 0, TE sample half), blending to the point-wise map over
        # EDGE_BLEND mm: point-wise blending of the tiny elliptical edges folded the TE cell fan (2026-11-12).
        for k in (0, len(ref) // 2):
            dist = np.linalg.norm(self.xy[self.wall] * scale + np.asarray(offset) - ref[k], axis=1)
            w = np.clip((dist - EDGE_RIGID) / (EDGE_BLEND - EDGE_RIGID), 0, 1)[:, None]
            wall = w * wall + (1 - w) * (target[k] - ref[k])
        move[self.wall] = wall
        if len(collar):
            nearest = cKDTree(self.xy[self.wall]).query(self.xy[collar])[1]
            move[collar] = move[self.wall][nearest]
        return move

    def periodic_shift(self, eta, scale, offset, section):
        """Shift (len(xy) x 2) of the periodic nodes, identical on each master/slave pair.

        The passage edge lies between the pressure side of one blade and the suction side of the next: it moves by the
        mean wall displacement of the two sides at the same meridional station (both components), fading linearly to
        zero at the inlet cap (the shared interface plane stays put) and one chord (or the outlet cap) behind the TE.
        Without it the twisted, shorter hub section closed on the fixed periodic lines near both edges (2026-11-12).
        """
        target = self.blade(float(eta)) + np.asarray(offset)  # Edge-stacking offset (stator TE strip; zero for rotors)
        ref = self.reference * scale + np.asarray(offset)
        half = len(ref) // 2
        disp = target - ref
        m_le, m_te = ref[0, 1], ref[half, 1]
        sides = []
        for side in (np.arange(0, half + 1), np.r_[np.arange(half, len(ref)), 0]):
            order = np.argsort(ref[side, 1])
            sides.append((ref[side, 1][order], disp[side][order]))
        xy = self.xy * scale + np.asarray(offset)
        caps = {k: np.unique(section["boundaries"][k]) for k in ("inlet", "outlet")}
        m_in, m_out = xy[caps["inlet"], 1].max(), xy[caps["outlet"], 1].min()
        nodes = np.unique(np.concatenate([np.unique(section["boundaries"][k]) for k in ("periodic_left", "periodic_right")]))
        m = xy[nodes, 1]
        station = np.clip(m, m_te, m_le)
        mean = 0.5 * sum(np.column_stack([np.interp(station, ms, ds[:, j]) for j in (0, 1)]) for ms, ds in sides)
        up = np.clip((m_in - m) / max(m_in - m_le, 1e-9), 0, 1)
        down = np.clip((m - max(m_te - (m_le - m_te), m_out)) / max(m_te - max(m_te - (m_le - m_te), m_out), 1e-9), 0, 1)
        weight = np.where(m > m_le, up, np.where(m < m_te, down, 1.0))
        shift = np.zeros_like(self.xy)
        shift[nodes] = mean * weight[:, None]
        shift[np.concatenate(list(caps.values()))] *= [1.0, 0.0]  # Cap corners stay on their planes
        shift[section["slave"]] = shift[section["master"]]
        return shift
