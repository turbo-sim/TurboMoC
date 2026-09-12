# Thermodynamic model

The characteristic equations require the speed of sound and density at each
point of the flow. These properties also determine how much the fluid
accelerates for a given pressure decrease. The thermodynamic closure must
therefore be consistent with the physical assumptions used to construct the
characteristic field.

## Equilibrium states

The package uses real-fluid properties through `jaxprop`, with CoolProp's
Helmholtz-energy equation-of-state backend as the default. For a single-phase
stagnation state, pressure and temperature define the thermodynamic state. At
liquid–vapor equilibrium, pressure fixes the saturation temperature, so pressure
and temperature alone do not determine the amount of each phase. The mixture
state is instead specified by pressure and vapor quality.

Under the homogeneous equilibrium model, liquid and vapor share the same
velocity, pressure, and temperature, and remain in phase equilibrium. Mass
transfer adjusts the phase fractions as the fluid expands. The closure neglects
velocity slip, delayed nucleation, and finite-rate thermal or phase relaxation.
It allows the mixture to be treated as a single compressible fluid with
equilibrium thermodynamic properties {cite:p}`cioffiStators2026`.

## Quality, void fraction, and mixture density

Quality is a mass fraction, whereas void fraction measures the portion of volume
occupied by vapor. Consider a mixture of total mass $m$ and volume $\mathcal V$.
The vapor and liquid masses are $Qm$ and $(1-Q)m$, so adding their volumes gives

```{math}
:label: eq-mixture-density
\frac{1}{\rho}=\frac{1-Q}{\rho_\ell}+\frac{Q}{\rho_v}.
```

Dividing each phase volume by $\mathcal V=m/\rho$ gives

```{math}
:label: eq-void-fractions
\alpha_v=\frac{Q\rho}{\rho_v},\qquad
\alpha_\ell=\frac{(1-Q)\rho}{\rho_\ell},\qquad
\alpha_v+\alpha_\ell=1.
```

Equivalently, $\rho=\alpha_\ell\rho_\ell+\alpha_v\rho_v$. Both density
expressions describe the same mixture. A small vapor mass fraction can occupy a
large volume when $\rho_v\ll\rho_\ell$, which explains the strong sensitivity
of density to quality near saturated-liquid conditions.

Specific enthalpy and entropy are mass-weighted averages:

```{math}
:label: eq-mixture-properties
h=(1-Q)h_\ell+Qh_v,\qquad
s=(1-Q)s_\ell+Qs_v.
```

For example, at prescribed pressure and entropy inside the saturation dome,
$Q=(s-s_\ell)/(s_v-s_\ell)$. The saturated properties must all be evaluated at
that pressure. Outside the dome, a single-phase equation of state replaces these
mixture relations; quality is not an independent single-phase property.

## Equilibrium speed of sound

The acoustic response is the isentropic pressure derivative

```{math}
:label: eq-sound-definition
a^2=\left(\frac{\partial p}{\partial\rho}\right)_s.
```

Within an equilibrium mixture, this derivative includes changes in quality.
Compressing or expanding the mixture can transfer mass between phases as well
as change their individual densities. Using only the single-phase sound speeds,
or averaging them by quality, omits this contribution.

The equilibrium expression used in the stator paper is
{cite:p}`cioffiStators2026,flattenRelaxation2011`

```{math}
:label: eq-hem-sound
\frac{1}{\rho a^2}=
\frac{\alpha_\ell}{\rho_\ell a_\ell^2}
+\frac{\alpha_v}{\rho_v a_v^2}
+T\left[
\frac{\alpha_\ell\rho_\ell}{c_{p,\ell}}
\left(\frac{\mathrm d s_\ell}{\mathrm d p}\bigg|_\mathrm{sat}\right)^2
+\frac{\alpha_v\rho_v}{c_{p,v}}
\left(\frac{\mathrm d s_v}{\mathrm d p}\bigg|_\mathrm{sat}\right)^2
\right].
```

Here $a_\ell,a_v$ and $c_{p,\ell},c_{p,v}$ are properties of the individual
saturated phases. The derivatives follow the saturation curves, rather than an
isotherm or an isentrope of an isolated phase. The additional positive terms
increase the equilibrium compressibility and reduce the sound speed.

```{figure} assets/equilibrium_sound_speed.png
:name: fig-equilibrium-sound
:alt: Mixture sound speed under different equilibrium assumptions versus vapor quality.

Mixture sound speed for cyclopentane at a saturation temperature of 200 °C.
Extracted from Fig. 1 of {cite:t}`cioffiStators2026`. The curves compare different
equilibrium assumptions; the MoC uses the full HEM closure. In the source
legend, $\mu$ denotes chemical potential, not the Mach angle used in this guide.
```

The equilibrium mixture limit need not coincide with the single-phase sound
speed as one phase disappears. In particular, the HEM sound speed can jump at
the saturation boundary, even though pressure, density, and enthalpy remain
continuous along the equilibrium path. The implications for initialization and
characteristic directions are developed in
[saturation crossings](saturation_crossings.md).

## Closure along an expansion

For an adiabatic flow without shaft work or dissipative effects, uniform inlet
stagnation enthalpy and entropy give

```{math}
:label: eq-static-closure
s=s_0,\qquad h=h_0-\frac{V^2}{2},\qquad
a=a(h,s_0),\qquad \rho=\rho(h,s_0).
```

Thus velocity determines the local thermodynamic state on the selected
isentrope. The implementation tabulates sound speed, density, and the
Prandtl–Meyer angle against velocity for repeated use in the characteristic
calculations. Tabulation reduces property-evaluation cost; it does not introduce
a constant-$\gamma$ approximation into the thermodynamic closure.

For rotor design the same procedure uses relative stagnation properties and
relative speed $W$. Its frame assumptions are stated in
[rotor design](rotor_design.md).
