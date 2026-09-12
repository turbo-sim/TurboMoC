# Notation and coordinate conventions

The same thermodynamic symbols are used throughout the guide. The rotor chapter
uses relative velocity, while the nozzle and stator chapters use stationary-frame
velocity. Geometric symbols are kept distinct from thermodynamic quantities.

## Flow and thermodynamic quantities

| Symbol | Definition | Units |
| --- | --- | --- |
| $p$, $T$, $\rho$ | Static pressure, temperature, and density | Pa, K, kg m$^{-3}$ |
| $h$, $s$ | Specific enthalpy and entropy | J kg$^{-1}$, J kg$^{-1}$ K$^{-1}$ |
| $h_0$, $s_0$, $p_0$, $T_0$ | Stagnation properties in the frame under consideration | Corresponding SI units |
| $V$ | Stationary-frame speed, $\sqrt{u^2+v^2}$ | m s$^{-1}$ |
| $u$, $v$ | Velocity components in the nozzle $x,y$ frame | m s$^{-1}$ |
| $W$ | Rotor-relative speed | m s$^{-1}$ |
| $a$ | Local equilibrium speed of sound | m s$^{-1}$ |
| $M$ | Mach number, $V/a$ or $W/a$ | Dimensionless |
| $Q$ | Vapor mass fraction (quality) | Dimensionless |
| $\alpha_\ell$, $\alpha_v$ | Liquid and vapor volume fractions | Dimensionless |
| $c_p$, $c_v$ | Specific heat capacities | J kg$^{-1}$ K$^{-1}$ |
| $\gamma$ | Heat-capacity ratio $c_p/c_v$ | Dimensionless |
| $G$ | Mass flux, $\rho V$ | kg m$^{-2}$ s$^{-1}$ |
| $\dot m'$ | Mass flow rate per unit span | kg m$^{-1}$ s$^{-1}$ |
| $\theta$ | Nozzle flow angle from the positive $x$ axis | rad |
| $\mu$ | Mach angle, $\sin^{-1}(1/M)$ | rad |
| $\nu$ | Generalized Prandtl–Meyer angle | rad |
| $\beta$ | Rotor-relative flow angle from its axial direction | rad |
| $\lambda_\pm$ | Characteristic slopes, $\tan(\theta\pm\mu)$ | Dimensionless |

Subscripts $\ell$ and $v$ denote saturated liquid and vapor. Subscripts
$\mathrm{in}$ and $\mathrm{out}$ denote inlet and outlet. A subscript
$\mathrm{ref}$ denotes the reference state of the Prandtl–Meyer integral; it can
be a smooth sonic state or the downstream state of a flashing transition.
For the latter, $V_\mathrm{ref}$ and $a_\mathrm{ref}$ need not be equal.

## Geometry and frames

| Symbol | Definition |
| --- | --- |
| $x,y$ | Nozzle streamwise and transverse coordinates; the symmetry axis is $y=0$ |
| $y_t$, $w_t$ | Throat half-height and full width, $w_t=2y_t$ |
| $R_t$, $R_d$ | Radii used for the initial-line and downstream wall constructions |
| $t_b$ | Blade pitch |
| $c$, $c_a$ | Chord and axial chord |
| $r_\mathrm{TE}$ | Trailing-edge radius |
| $\chi$ | Stator outlet metal angle used in the pitch relation |
| $R$, $R_\mathrm{ref}$ | Vortex radius and its dimensional reference scale |
| $\widehat R$ | Normalized vortex radius $R/R_\mathrm{ref}$ |
| $\zeta$ | Rotation parameter for a rotor circular arc |
| $\xi,\eta$ | Chordwise and pitchwise coordinates passed to the radial map |
| $r,\varphi$ | Radius and azimuth after radial mapping |

The nozzle upper wall has positive $y$, with positive $\theta$ turning away from
the axis. The characteristic family $C_+$ has slope $\tan(\theta+\mu)$, and
$C_-$ has slope $\tan(\theta-\mu)$. These definitions determine the signs of
the [compatibility relations](method_of_characteristics.md).

Stator coordinates are rotated during assembly: the returned profile has its
chordwise extent mainly along its local $y$ direction and its pitch along $x$.
The rotor profile instead uses local $x$ as the axial direction and $y$ as the
pitchwise direction. Neither frame should be inferred from the orientation of a
plot. The [radial mapping](radial_mapping.md) explicitly selects the relevant axes.

## Relation to package names and units

| Theory quantity | Package argument or result | Convention |
| --- | --- | --- |
| $p_0,T_0,Q_0$ | `P0`, `T0`, `Q0` | Pa, K, and a fraction from 0 to 1 |
| $p_\mathrm{out}$ | `p_back` | Pa; a target for designing a new nozzle |
| $M_\mathrm{out}$ | `Noz_Mach`, `design_Noz_Mach` | Dimensionless |
| $y_t,R_t,R_d$ | `y_t`, `rho_t`, `rho_d` | Meters; `rho_t` and `rho_d` are geometric radii, not densities |
| $\theta,\mu$ | Point fields `theta`, `alpha` | Radians; the code's `alpha` is a Mach angle |
| $\chi$ | `metal_angle_out` | Degrees |
| $t_b$ | `pitch` | Millimeters for stator geometry |
| $\beta_\mathrm{in},\beta_\mathrm{out}$ | `beta_inlet`, `beta_outlet` | Degrees at the rotor API boundary |
| $R_\mathrm{ref}$ | `r_star` | Meters when specified; otherwise rotor coordinates are dimensionless |

`design_nozzle` returns coordinates in meters. Stator geometry functions accept
these coordinates, convert them internally, and return millimeters. Rotor
coordinates are dimensionless unless `r_star` is supplied. Radial mapping
requires its input coordinates and radii to use compatible length units.

In the source papers, $s$ can denote pitch, $\alpha$ can denote Mach angle, and
$M^*$ can denote velocity normalized by the sonic speed. Here $s$ is reserved
for entropy, $\mu$ for Mach angle, and normalized velocity is written explicitly.
