# Suggestions for the vortex-flow rotor paper

Hi Andrea,

I have three suggestions that could help strengthen the theoretical background and better motivate the relevance of the vortex-flow design method in the paper.

## 1. Brief derivation of the vortex-flow equations

I think it would be nice to include a short derivation of the free-vortex equations governing the circular-arc region. The derivation is quite elegant and requires only a few assumptions and equations.

Consider a steady, two-dimensional, inviscid, isentropic flow with circular streamlines, such that the radial velocity vanishes:

$$
V_r=0, \qquad V_\theta=V(r)
$$

The radial momentum equation gives

$$
\frac{dp}{dr}=\rho\frac{V^2}{r}
$$

Using the isentropic relation $dh=dp/\rho$:

$$
\frac{dh}{dr}=\frac{V^2}{r}
$$

Conservation of total enthalpy along the streamlines gives

$$
h_0=h+\frac{V^2}{2}=\mathrm{constant}
$$

Assuming the same total enthalpy throughout the vortex region, differentiation yields

$$
\frac{dh}{dr}+V\frac{dV}{dr}=0
$$

Combining the equations:

$$
\frac{V^2}{r}+V\frac{dV}{dr}=0
$$

and therefore

$$
\boxed{rV=\mathrm{constant}}
$$

This is the free-vortex condition. It establishes the relationship between the velocities and curvature radii of the pressure-side and suction-side circular arcs:

$$
\boxed{
r_\mathrm{PS}V_\mathrm{PS}
=
r_\mathrm{SS}V_\mathrm{SS}
}
$$

For a given stagnation state and entropy, the velocities corresponding to the prescribed Mach numbers can be obtained from the fluid thermodynamic properties.

I think this derivation would be worth including because it is concise, physically intuitive, and shows that the vortex-flow relation is not restricted to perfect gases. It remains applicable to equilibrium two-phase flows under the same assumptions.

## 2. Motivation for MoC inlet/outlet transitions versus straight sections

I think that in the paper we should explain explicitly why the MoC transitions are advantageous compared with a conventional double-circular-arc blade with constant passage width and straight inlet/outlet sections.

The important distinction is not the circular-arc geometry itself, but how the flow is connected to the central curved region.

**Conventional geometry with straight transitions**

In a double-circular-arc blade with constant passage width and straight inlet/outlet sections, the flow must adapt to the change in wall direction and curvature. For supersonic flow, this can generate expansion waves and compression waves that may interact within the passage. In particular, compressive turning can lead to oblique shock waves that introduce entropy generation and total-pressure losses.

**MoC-designed transitions**

With the vortex-flow method, the inlet and outlet transitions are constructed to connect the uniform external flow to the prescribed vortex flow through compatible characteristic waves. The objective is to achieve the required acceleration, deceleration, and turning without shocks in the ideal inviscid solution.

This provides a clear physical motivation for the method: rather than relying on the flow to adapt to a geometrically prescribed blade, we construct the geometry to satisfy a desired supersonic flow field.

I think it would be useful to emphasize this contrast in the introduction.


## 3. Prandtl–Meyer function in the two-phase region

I also suggest including plots of the Prandtl–Meyer function for the working fluid in the two-phase region, similar to the plots we included in the wet-to-dry turbine paper.

The interesting point is that **the Prandtl–Meyer function depends on the thermodynamic behavior of the fluid and the local Mach number, rather than on the blade geometry**. More precisely, for a real fluid it also depends on the stagnation state and the isentrope being considered.

I think showing $\nu(M)$ for representative two-phase expansion conditions would help readers understand how the thermodynamic properties enter the MoC formulation.

It would also highlight one of the contributions of our work: extending a classical supersonic aerodynamic design method to fluids with non-ideal thermodynamic behavior.
