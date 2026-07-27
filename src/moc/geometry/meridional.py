"""
moc.geometry.meridional -- meridional (axial-radial, x-r) view of a
stator + rotor stage, sharing its hub/tip radii inputs with the flared 3D
blade extrusion (see moc.geometry.blade.export_flared_blade_step).

Convention (matching Anderson et al. 2025's own Table 2 / Fig. 4/5 -- rhub,
rtip at cascade inlet/outlet): each row's hub and tip lines are assumed to
span the SAME axial stations (x_in, x_out) -- only their radii differ, the
standard "conical annulus" flare model (Anderson's own Eq. 4 linear-
combination-of-radii formula assumes this too, via a shared axial
fraction). This means a flared 3D blade extrusion does NOT need to
axially shift the tip profile relative to the hub profile -- flare shows
up purely as radius change (this meridional view) and, in the 3D blade,
as the tip profile's pitchwise (tangential) extent scaling with radius
(pitch = 2*pi*r/Z_blades at constant blade count) -- the two are two
views of the SAME flare inputs, not independently specified.
"""

__all__ = ["build_meridional_view"]


def build_meridional_view(
    stator_axial_chord, stator_r_hub_in, stator_r_hub_out, stator_r_tip_in, stator_r_tip_out,
    rotor_axial_chord, rotor_r_hub_in, rotor_r_hub_out, rotor_r_tip_in, rotor_r_tip_out,
    gap,
):
    """
    Hub/tip line coordinates for a stator row (x in [0, stator_axial_chord])
    followed by a rotor row (x in [stator_axial_chord + gap, ... +
    rotor_axial_chord]) -- for moc.plotly/mpl's plot_meridional_view.

    Parameters
    ----------
    stator_axial_chord, rotor_axial_chord : float
        Each row's own axial (flow-direction) extent -- e.g. the stator
        blade's own chordwise span (mm), the rotor's own "chord" (CSTAR,
        already dimensionalized if it was designed with r_star).
    stator_r_hub_in/out, stator_r_tip_in/out, rotor_r_hub_in/out,
    rotor_r_tip_in/out : float
        Hub/tip radii at each row's own inlet/outlet.
    gap : float
        Axial (row-to-row) spacing between the stator's own trailing edge
        and the rotor's own leading edge.
    """
    x_stator = [0.0, float(stator_axial_chord)]
    x_rotor_start = float(stator_axial_chord) + float(gap)
    x_rotor = [x_rotor_start, x_rotor_start + float(rotor_axial_chord)]

    return {
        "stator": {
            "x": x_stator,
            "hub": [float(stator_r_hub_in), float(stator_r_hub_out)],
            "tip": [float(stator_r_tip_in), float(stator_r_tip_out)],
        },
        "rotor": {
            "x": x_rotor,
            "hub": [float(rotor_r_hub_in), float(rotor_r_hub_out)],
            "tip": [float(rotor_r_tip_in), float(rotor_r_tip_out)],
        },
        "gap": float(gap),
    }
