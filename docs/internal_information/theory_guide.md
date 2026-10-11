# From a method-of-characteristics blade to a three-dimensional turbine mesh

**A teaching guide to the TurboMoC geometry and meshing implementation**

Prepared 11 October 2026. Equations and implementation descriptions refer to the canonical source under [turbo_moc](../src/turbo_moc). The package promotion changes source ownership, without changing the mathematical procedure described here. The [merging summary](merging_summary.md) distinguishes the canonical stages from preserved legacy structured/repair workflows. The emphasis is on the geometric interpretation of a planar blade, its mapping onto a meridional channel, and the construction of a conforming three-dimensional mesh.

The main development is in Sections [3–5](#3-a-general-meridional-channel), which establish the coordinate mapping, and Sections [7–8](#7-extending-the-reference-mesh-across-span), which explain the volume construction. Section [10](#10-exporting-a-solver-readable-cgns-mesh) follows the assembled mesh into a solver-readable file. Section [12](#12-student-calculations-and-implementation-map) provides calculations that can be checked against the source.

## 1. Physical problem and design procedure

The method of characteristics (MoC) provides a direct way to design the divergent part of a supersonic nozzle. The nozzle wall is subsequently incorporated into a complete stator blade, including its convergent part, leading edge and trailing edge. To use that blade in a turbomachine, its two-dimensional coordinates can be interpreted as a physical linear cascade or related to a meridional channel with a specified hub, shroud and number of blades.

The workflow considered here separates three operations:

1. **Geometry definition:** construct one fitted blade, choose the meridional channel and interpret the blade in a conformal blade-to-blade plane.
2. **Reference meshing:** generate a two-dimensional fluid mesh around that blade, with conforming periodic boundaries and optional near-wall inflation.
3. **Volume construction:** adapt the reference mesh to each span station, map its nodes to physical space, and connect adjacent stations into volume cells.

This separation has a practical consequence. The blade and channel can be inspected before a mesh exists, section-mesh controls can be adjusted before constructing the volume, and span controls can be changed while retaining the reference topology.

The underlying MoC calculations assume the flow model selected in the nozzle solver. In the two-phase applications described in the accompanying manuscripts, the homogeneous equilibrium model supplies the thermodynamic properties and speed of sound. For an isentropic expansion with negligible inlet velocity, the local speed and Mach number satisfy

$$
V=\sqrt{2[h_0-h(p,s_0)]},\qquad
M=\frac{V}{a(p,s_0)}.
$$

Here $h_0$ is stagnation enthalpy, $s_0$ is stagnation entropy, $p$ is static pressure and $a$ is the thermodynamic speed of sound. This relation explains why the fluid model influences the nozzle wall. The geometry and mesh routines discussed below consume that wall; they do not calculate a new three-dimensional flow solution.

It is useful to distinguish the **MoC characteristic network**, which is used to integrate the flow equations, from the **CFD mesh**, which discretizes the fluid domain. Their nodes and connectivity serve different purposes. The Gmsh mesh is generated around the fitted geometry and does not inherit the characteristic network.

The app organizes the design as an ordered collection of blade rows. Each row
owns its geometry and meshing inputs and has a stable identifier. MoC stator
and vortex rotor providers both feed the same mapping and meshing stages.
The rotor inlet can be entered manually or derived from the upstream stator
exit through the existing velocity-triangle calculation. Adjacent row
domains meet at a shared physical interface, as derived in Section 11.1;
their CFD coupling and reference frames are configured in the solver.

The present three-dimensional stator model uses one blade template, with a prescribed similarity scaling across span. It supports general meridional wall curves through the Python interface. The axial and radial YAML definitions are convenient ways of constructing those curves; the app exposes axial, radial and linear geometry types. Mixed flow is outside the current app, despite the broader mathematical applicability of the general meridional mapping. Independently prescribed spanwise blade sections, lean, sweep and twist are outside this single-template model. Mapping onto curved or tapered channels can nevertheless produce spanwise changes in physical position and orientation.

## 2. Coordinate systems and the planar blade

### 2.1. Three coordinate systems with different meanings

The original nozzle plane, the conformal meshing plane and the physical machine coordinates should be kept distinct.

| Symbol | Meaning | Units in the implementation |
| --- | --- | --- |
| $(x_n,y_n)$ | Original planar MoC nozzle coordinates | m at the solver interface |
| $\mathbf b=(x_b,y_b)$ | Fitted blade in the planar cascade frame | mm |
| $\mathbf q=(q_x,q_y)$ | Length-valued reference plane used by Gmsh; conformal for annular rows, physical for linear rows | mm |
| $m$ | Distance along a meridional curve | mm |
| $m'$ | Meridional conformal coordinate, defined by $dm'=dm/r$ | dimensionless |
| $\theta$ | Clockwise circumferential coordinate about the machine axis | rad |
| $\phi=-\theta$ | Conventional counterclockwise cylindrical azimuth | rad |
| $(r,z)$ | Meridional radius and axial position | mm |
| $(X,Y,Z)$ | Physical Cartesian coordinates, with $Z=z$ | mm before export |
| $u$ | Common hub/shroud curve parameter; blade LE at 0 and TE at 1 | dimensionless |
| $\eta$ | Interpolation coordinate from hub to shroud | dimensionless |

In the fitted cascade frame, $x_b$ is pitchwise and downstream motion corresponds to decreasing $y_b$. The blade construction converts the MoC wall from metres to millimetres before applying reflection, rotation, translation and fitting.

For axial and radial rows, the geometric meaning of the Gmsh plane is

$$
q_x=R_{\mathrm{ref}}\theta,\qquad
q_y=-R_{\mathrm{ref}}m',
\tag{1}
$$

where $R_{\mathrm{ref}}$ is one fixed reference radius for the entire machine passage. Consequently, the app displays $(\theta,-m')=(q_x/R_{\mathrm{ref}},q_y/R_{\mathrm{ref}})$. Its negative vertical direction is the downstream direction. In a linear cascade, the same internal coordinate ordering is retained but $\mathbf q$ is a physical planar coordinate; no arbitrary radius or angular coordinate is introduced. Section [5.4](#54-linear-cascade-and-physical-extrusion) gives its Cartesian mapping.

The prime in $m'$ is part of the coordinate name. It does not denote differentiation, and $m'$ is not a distance measured in millimetres. Multiplication by $R_{\mathrm{ref}}$ is what gives the meshing coordinates their length units.

### 2.2. Construction and fitting of the blade

The planar blade parameterization combines the MoC-derived divergent wall with a convergent construction, a reflected and translated pressure-side construction, a leading-edge connection and a prescribed trailing-edge arc. The main blade body is approximated by a clamped cubic B-spline,

$$
\mathbf b(t)=\sum_{j=0}^{n_c-1}N_{j,3}(t)\mathbf P_j,
\tag{2}
$$

where $N_{j,3}$ are cubic B-spline basis functions and $\mathbf P_j$ are control points. After resampling the constructed profile, the interior control points are fitted by least squares with the endpoints fixed.

The resulting main spline has two endpoints at the trailing edge. A separate circular arc closes the blade. Thus the complete contour is closed, although the main fitted spline is not a periodic closed spline. Gmsh receives its control points, degree and knots directly, together with two exact quarter-circle trailing-edge arcs.

This fitting step reduces the number of geometric degrees of freedom without requiring the CFD mesh to follow every original MoC sample. The fitting error remains a geometric approximation error; increasing mesh resolution does not remove it. The current implementation controls the number of fit points and control points rather than enforcing the particular fitting tolerance reported in the stator manuscript.

The source owners are [the blade parameterization](../src/turbo_moc/geometry/blade.py), [the spline utilities](../src/turbo_moc/geometry/spline_tool.py) and [the planar geometry module](../src/turbo_moc/geometry/geometry_2d.py).

## 3. A general meridional channel

### 3.1. Hub, shroud and intermediate surfaces

Let the hub and shroud be parameterized in the meridional plane by

$$
\mathbf h(u)=
\begin{bmatrix}r_h(u)\\z_h(u)\end{bmatrix},
\qquad
\mathbf s(u)=
\begin{bmatrix}r_s(u)\\z_s(u)\end{bmatrix}.
$$

The implementation stores these curves as B-splines on $0\le u\le1$. Corresponding hub and shroud stations are interpolated according to

$$
\mathbf P(u,\eta)
=(1-\eta)\mathbf h(u)+\eta\mathbf s(u)
=\begin{bmatrix}r(u,\eta)\\z(u,\eta)\end{bmatrix},
\qquad 0\le\eta\le1.
\tag{3}
$$

The hub is $\eta=0$ and the shroud is $\eta=1$. Holding $\eta$ fixed gives a meridional curve; revolving that curve about $Z$ gives the surface on which a blade section is placed.

The correspondence between wall parameters matters. Pairing a hub station with an inappropriate shroud station can fold the interpolated channel even if the two walls are individually smooth. The core checks positive radii, nonzero streamwise tangents and a consistent, nonzero meridional orientation determinant at sampled stations:

$$
D(u,\eta)=r_u z_\eta-z_u r_\eta.
\tag{4}
$$

These sampled checks reject many invalid definitions. They do not establish a continuous proof that arbitrary high-order wall curves cannot fold between samples.

The constant-$\eta$ curves are **geometric pseudo-streamlines**. They are prescribed by wall interpolation rather than obtained from a meridional flow solution. Equal increments of $\eta$ also need not correspond to equal physical distances across a general channel.

### 3.2. Meridional distance and conformal distance

On a fixed-$\eta$ curve, the meridional distance increment is

$$
dm=\|\mathbf P_u(u,\eta)\|\,du.
$$

Define the downstream-increasing conformal coordinate by

$$
m'(u,\eta)=
\int_0^u
\frac{\|\mathbf P_u(v,\eta)\|}{r(v,\eta)}\,dv,
\qquad
c'(\eta)=m'(1,\eta).
\tag{5}
$$

Here $c'(\eta)$ is the dimensionless meridional conformal chord between the blade leading and trailing edges. Radius appears in the denominator because the circumferential length increment is $r\,d\theta$. The conformal transformation puts meridional and circumferential increments on the same local scale.

Since the integrand is positive in a valid channel, $m'$ increases monotonically downstream and can be inverted to obtain $u$. Once $u$ is known, Equation (3) supplies $r$ and $z$.

The reference radius is evaluated at midspan and halfway along the **conformal** chord:

$$
u_{\mathrm{ref}}=
u\!\left(\frac{c'(1/2)}2,\frac12\right),
\qquad
R_{\mathrm{ref}}=r(u_{\mathrm{ref}},1/2).
\tag{6}
$$

With $N_b$ physical blades, the reference pitch and length-valued conformal chord are

$$
\Delta\theta=\frac{2\pi}{N_b},\qquad
p_{\mathrm{ref}}=R_{\mathrm{ref}}\Delta\theta,\qquad
L_{\mathrm{ref}}=R_{\mathrm{ref}}c'(1/2).
\tag{7}
$$

$L_{\mathrm{ref}}$ is a length in the reference plane. In a general channel it differs from the physical meridional distance $\int_0^1\|\mathbf P_u\|du$.

### 3.3. Mapping to physical Cartesian coordinates

Given a conformal mesh node $\mathbf q$ at span station $\eta$, first calculate

$$
m'=-\frac{q_y}{R_{\mathrm{ref}}},\qquad
\theta=\frac{q_x}{R_{\mathrm{ref}}},\qquad
\phi=-\theta.
$$

Invert Equation (5), evaluate Equation (3), and finally apply

$$
\mathbf X(\mathbf q,\eta)=
\begin{bmatrix}
r(u,\eta)\cos\phi\\
r(u,\eta)\sin\phi\\
z(u,\eta)
\end{bmatrix}.
\tag{8}
$$

This is the common mapping for axial, radial and curved meridional channels. The rotation axis is always global $Z$; a radial machine changes the meridional curves, not the Cartesian convention.

The clockwise convention is significant. Increasing $q_x$ decreases the conventional azimuth $\phi$. In `MeridionalChannel.map_points`, the local variable named `theta` contains this conventional azimuth; Equations (1) and (8) distinguish the two angular conventions explicitly.

### 3.4. Why the transformation is conformal

On any fixed-$\eta$ surface of revolution, the physical line element is

$$
d\ell^2=dm^2+r^2d\theta^2
=r^2(dm'^2+d\theta^2)
=\left(\frac{r}{R_{\mathrm{ref}}}\right)^2
(dq_x^2+dq_y^2).
\tag{9}
$$

Both in-surface coordinate directions have the same local scale factor. Therefore, infinitesimal section angles are preserved, while a small reference-plane length $h_q$ maps approximately to

$$
h_{\mathrm{surface}}\simeq
\frac{r}{R_{\mathrm{ref}}}h_q.
\tag{10}
$$

For a blade tangent making a metal angle $\beta$ with the downstream direction,

$$
\mathbf t_q=
\begin{bmatrix}\sin\beta\\-\cos\beta\end{bmatrix},
\qquad
\mathbf t_{\mathrm{physical}}
=\cos\beta\,\mathbf t_m-\sin\beta\,\mathbf e_\phi,
\tag{11}
$$

where $\mathbf t_m$ is the downstream meridional unit tangent and $\mathbf e_\phi$ is the counterclockwise azimuthal unit vector. Positive $\beta$ thus produces clockwise turning.

Equation (9) applies within one prescribed stream surface. It does not make the full three-dimensional coordinate system orthogonal. Cross-span coordinate directions can be oblique to the streamwise direction in a tapered or curved channel.

Angle preservation is also a geometric property, not a proof that the planar MoC flow becomes an exact three-dimensional solution. Streamtube area variation, radial pressure gradients, secondary flows and thermodynamic nonequilibrium require additional physical modelling and validation.

### 3.5. Numerical evaluation and inlet/outlet extensions

For straight wall segments, Equation (5) has an analytic inverse. Write

$$
r(u)=r_0+\Delta r\,u,\qquad
z(u)=z_0+\Delta z\,u,\qquad
S=\sqrt{\Delta r^2+\Delta z^2}.
$$

Then

$$
m'(u)=
\begin{cases}
Su/r_0,&\Delta r=0,\\[1mm]
\dfrac{S}{\Delta r}
\ln\!\left(1+\dfrac{\Delta r}{r_0}u\right),&\Delta r\ne0,
\end{cases}
\tag{12}
$$

and

$$
u(m')=
\begin{cases}
r_0m'/S,&\Delta r=0,\\[1mm]
\dfrac{r_0}{\Delta r}
\left[\exp\!\left(\dfrac{\Delta r}{S}m'\right)-1\right],
&\Delta r\ne0.
\end{cases}
\tag{13}
$$

The code uses `log1p` and `expm1` to evaluate these expressions accurately when their arguments are small. This is a general straight-channel calculation, rather than a shortcut selected by a machine-type label.

For curved walls, adaptive quadrature evaluates $c'(\eta)$. The inverse coordinate map is obtained by integrating

$$
\frac{du}{dm'}=
\frac{r(u,\eta)}{\|\mathbf P_u(u,\eta)\|},
\qquad u(0)=0.
\tag{14}
$$

The DOP853 integrator supplies dense output. A forward integration and, when needed, a backward integration provide the mapping for many mesh nodes without solving a separate scalar root problem at every node.

The CFD inlet and outlet extend beyond the blade leading and trailing edges. Outside $0\le u\le1$, each wall continues along its endpoint tangent. For the hub,

$$
\mathbf h_{\mathrm{ext}}(u)=
\begin{cases}
\mathbf h(0)+u\mathbf h_u(0),&u<0,\\
\mathbf h(u),&0\le u\le1,\\
\mathbf h(1)+(u-1)\mathbf h_u(1),&u>1.
\end{cases}
\tag{15}
$$

The shroud follows the same rule. The extensions are tangent continuous; curvature continuity is not imposed. All mapped inlet/outlet nodes must still have finite coordinates and positive radius.

These operations are implemented by `MeridionalChannel` in [meridional_channel.py](../src/turbo_moc/geometry/meridional_channel.py), principally `meridional_points`, `prime_chord`, `parameter_from_prime` and `map_points`.

## 4. Positioning one blade template across span

### 4.1. Fit and place the reference section once

Let $\mathbf b_{\mathrm{raw}}$ be the blade produced from the planar MoC wall. Define its reference leading-edge point and downstream extent by

$$
\mathbf b_{\mathrm{LE}}
=\mathbf b_{\mathrm{raw}}\big|_{\max y_b},
\qquad
c_{\mathrm{raw}}=\max y_b-\min y_b.
$$

Both extrema are evaluated on the complete sampled contour, including its trailing-edge arc.

The reference section is obtained by one similarity transform:

$$
\kappa=\frac{L_{\mathrm{ref}}}{c_{\mathrm{raw}}},
\qquad
\mathbf b_{\mathrm{ref}}
=\kappa(\mathbf b_{\mathrm{raw}}-\mathbf b_{\mathrm{LE}}).
\tag{16}
$$

Both coordinates use the same scale. Normalizing the two axes independently would alter the blade angles and the ratio between its pitchwise and streamwise dimensions.

The machine blade count specifies $p_{\mathrm{ref}}$ through Equation (7). The passage pitch is set to that value without redesigning the already fitted blade shape. Therefore, the original MoC-derived planar pitch and the imposed machine pitch are distinct design quantities. Changing blade count can change the interblade opening and solidity; containment checks establish geometric feasibility, while CFD must assess the resulting passage flow.

If $p_0$ is the original template pitch, its scaled value is $\kappa p_0$. It matches the imposed machine pitch only when $p_{\mathrm{ref}}=\kappa p_0$. The corresponding geometrically compatible blade count would be $N_b^\star=2\pi R_{\mathrm{ref}}/(\kappa p_0)$, which can be noninteger and is not an aerodynamic optimum. The original pitch is used to construct the blade before the stored passage pitch is replaced; replacing that value does not reconstruct the pressure-side coordinates. Away from reference span, the similar template would carry pitch $a(\eta)\kappa p_0$, whereas the machine pitch remains fixed. Reference-span compatibility therefore need not persist across span.

The conformal chord-to-pitch ratio is

$$
\frac{L_{\mathrm{ref}}}{p_{\mathrm{ref}}}
=\frac{N_b c'(1/2)}{2\pi}.
\tag{17}
$$

This ratio provides a direct check on how blade count and meridional dimensions influence the available passage width.

The passage reference line is assembled from a convergent turning construction, a straight divergent construction following the MoC exit direction, and smooth inlet/outlet extensions. It is a geometric construction line, not a recovered mean-thickness camberline or a computed flow streamline. Translating it by $\pm p_{\mathrm{ref}}/2$ gives the two periodic sides.

One manual or automatically selected pitchwise translation centers the passage around the blade. Automatic placement maximizes the smaller sampled clearance to the two periodic sides. It uses a coarse search followed by bounded optimization; the result is not a proof of a global optimum for arbitrary geometry.

### 4.2. Similar sections, fixed pitch and fixed placement

At a span station $\eta$, define

$$
a(\eta)=\frac{c'(\eta)}{c'(1/2)}.
\tag{18}
$$

The blade contour becomes

$$
\mathbf b_\eta=a(\eta)\mathbf b_{\mathrm{ref}}.
\tag{19}
$$

Let $\boldsymbol\Delta=(\Delta_x,0)$ be the passage translation chosen at reference span. If $\mathbf c_{\mathrm{ref}}$ is the shifted construction line, the corresponding line at another span is

$$
\mathbf c_\eta
=a(\eta)(\mathbf c_{\mathrm{ref}}-\boldsymbol\Delta)
+\boldsymbol\Delta.
\tag{20}
$$

The periodic sides are then reconstructed as

$$
\mathbf c_{L,\eta}
=\mathbf c_\eta-
\begin{bmatrix}p_{\mathrm{ref}}/2\\0\end{bmatrix},
\qquad
\mathbf c_{R,\eta}
=\mathbf c_\eta+
\begin{bmatrix}p_{\mathrm{ref}}/2\\0\end{bmatrix}.
\tag{21}
$$

Thus the section scale changes, but the angular pitch and the selected reference-plane translation remain fixed. Scaling both periodic sides directly would produce $a(\eta)p_{\mathrm{ref}}$, which would represent the wrong machine sector.

The blade fit and automatic passage placement are performed once. Subsequent sections apply Equation (19), reconstruct their passage through Equations (20)–(21), and check containment. The source does not repeat the original fit or centering search at every span.

The reference leading edge has $m'=0$. After scaling, its trailing edge has $m'=c'(\eta)$, so the prescribed meridional leading- and trailing-edge stations are respected. For a general curved channel, equal normalized conformal distance $m'/c'$ need not correspond to equal wall parameter $u$ across span.

“One conformal template” consequently means a family of similar sections in the conformal plane. It does not imply one rigid Cartesian extrusion for every machine. These operations are implemented by `build_design` and `Design.section` in [geometry_3d.py](../src/turbo_moc/geometry/geometry_3d.py).

## 5. Axial and radial machines as special cases

### 5.1. Pure axial channel

For an axial machine with hub radius $r_h$, blade height $H$ and hub/tip axial chords $C_h,C_t$, define

$$
r(\eta)=r_h+\eta H,\qquad
C(\eta)=C_h+\eta(C_t-C_h),
\qquad
z(u,\eta)=uC(\eta).
\tag{22}
$$

The radius is constant along each section, giving

$$
m'=\frac{C(\eta)}{r(\eta)}u,\qquad
c'(\eta)=\frac{C(\eta)}{r(\eta)},\qquad
z=r(\eta)m'.
\tag{23}
$$

The map from a local conformal point is therefore

$$
X=r(\eta)\cos\!\left(-\frac{q_x}{R_{\mathrm{ref}}}\right),\quad
Y=r(\eta)\sin\!\left(-\frac{q_x}{R_{\mathrm{ref}}}\right),\quad
Z=-\frac{r(\eta)}{R_{\mathrm{ref}}}q_y.
\tag{24}
$$

Here $R_{\mathrm{ref}}=r_h+H/2$ and $L_{\mathrm{ref}}=(C_h+C_t)/2$. With the supplied axial case,

$$
r_h=100\ \mathrm{mm},\quad H=10\ \mathrm{mm},\quad
C_h=C_t=20\ \mathrm{mm},\quad N_b=31,
$$

the reference quantities are

$$
R_{\mathrm{ref}}=105\ \mathrm{mm},\qquad
c'(1/2)=\frac{20}{105}=0.19047619,\qquad
p_{\mathrm{ref}}=21.28175669\ \mathrm{mm}.
$$

Although the physical axial chord is constant, the conformal scale is not:

$$
a(0)=1.05,\qquad
a(1/2)=1,\qquad
a(1)=\frac{105}{110}=0.95454545.
$$

This result follows because a fixed physical chord occupies a smaller $m'$ interval at larger radius. Multiplying $a(\eta)$ by the local physical scale $r/R_{\mathrm{ref}}$ gives unity for the constant-chord case. Thus the blade has equal dimensions when each cylindrical section is developed into a plane, while its angular extent changes with radius. This is why even a constant-chord axial blade can require reference-mesh deformation.

### 5.2. Pure radial channel

For parallel radial endwalls separated by axial height $H$,

$$
r(u)=R_{\mathrm{in}}+u(R_{\mathrm{out}}-R_{\mathrm{in}}),
\qquad z(\eta)=\eta H.
\tag{25}
$$

Let $\varepsilon=\operatorname{sign}(R_{\mathrm{out}}-R_{\mathrm{in}})$. Equation (5) becomes

$$
m'=\varepsilon\ln\!\left(\frac{r}{R_{\mathrm{in}}}\right),
\qquad
r=R_{\mathrm{in}}\exp(\varepsilon m'),
\qquad
c'=\left|\ln\frac{R_{\mathrm{out}}}{R_{\mathrm{in}}}\right|.
\tag{26}
$$

For radial inflow, $\varepsilon=-1$ and radius decreases downstream. For radial outflow, $\varepsilon=+1$ and radius increases downstream. In both cases $m'$ increases downstream.

The physical map is

$$
r=R_{\mathrm{in}}
\exp\!\left(-\varepsilon\frac{q_y}{R_{\mathrm{ref}}}\right),
\qquad
X=r\cos\!\left(-\frac{q_x}{R_{\mathrm{ref}}}\right),\quad
Y=r\sin\!\left(-\frac{q_x}{R_{\mathrm{ref}}}\right),\quad
Z=\eta H.
\tag{27}
$$

At half the conformal chord, radius is the geometric mean,

$$
R_{\mathrm{ref}}=\sqrt{R_{\mathrm{in}}R_{\mathrm{out}}}.
\tag{28}
$$

For the supplied radial case, $R_{\mathrm{in}}=100$ mm, $R_{\mathrm{out}}=80$ mm, $H=8$ mm and $N_b=12$:

$$
R_{\mathrm{ref}}=89.44271910\ \mathrm{mm},\quad
c'=\ln(1.25)=0.22314355,\quad
L_{\mathrm{ref}}=19.95856598\ \mathrm{mm},\quad
p_{\mathrm{ref}}=46.83209821\ \mathrm{mm}.
$$

The physical radial travel is 20 mm, whereas the length-valued conformal chord is approximately 19.959 mm. This provides a concrete example of the distinction following Equation (7).

Since $c'$ is independent of $\eta$, $a=1$ at every span station. The same conformal mesh can be mapped directly at all heights. The physical radial blade is an axial extrusion of one $(X,Y)$ profile; no harmonic section deformation is needed for this channel.

### 5.3. Curved and mixed-flow channels

Curved hub/shroud definitions use the same interpolation, integration, inversion and Cartesian map. A mixed-flow channel is obtained by prescribing meridional curves for which both $r$ and $z$ vary downstream. There is no need to choose between an axial formula and a radial formula inside the mapping routine.

The current core accepts such definitions through `GeometryDefinition` and the `definition` argument of `build_design`. The annular YAML/UI adapters expose axial and radial meridional channels; the physical linear-cascade mapping is separate. Mixed flow is outside the present app selectors and project schema, while the general Python mapping remains mathematically applicable to curved channels.

![Meridional channels and projected sweep stations](figures/meridional_sweep_projection.png)

**Figure 1.** Axial and radial channel examples, shown relative to zero radius, with enlarged views of the projected sweep stations. The upper panels use equal physical axis scales; the lower panels use unequal scales to make the discretization visible. The streamwise stations are taken from actual reference periodic-boundary mesh nodes. The curves are prescribed geometric pseudo-streamlines; the figure does not show a solved meridional flow field or an independently meshed meridional CFD domain.

### 5.4. Linear cascade and physical extrusion

A linear cascade has straight translational periodicity rather than an annular
sector. The user specifies the meridional, or streamwise, chord $C$ and blade
height $H$. Radius and physical blade count have no role in this model. Let
$C_0$ and $p_0$ denote the streamwise chord and pitch of the fitted source
cascade. A common scaling gives

$$
\kappa=\frac{C}{C_0},\qquad
\mathbf q=\kappa\mathbf b,\qquad
p=\kappa p_0.
\tag{L1}
$$

Any rigid placement applied during passage construction is included in the
stored reference definition. Scaling both coordinates and the pitch together
preserves the profile angles and $p/C$. The present linear option retains
that scaled source pitch; it does not introduce an independent solidity
constraint.

The internal plane remains pitchwise $q_x$ and downstream $-q_y$, allowing the
same blade construction, inflation mesher and connectivity to be reused. The
physical linear mapping is

$$
X=-q_y,\qquad Y=q_x,\qquad Z=\eta H,
\qquad 0\leq\eta\leq1.
\tag{L2}
$$

This in-plane rotation has positive determinant, so it preserves the section
cell winding. Unlike annular mapping, it has a constant in-plane metric:
$d\ell^2=dq_x^2+dq_y^2$. Reference-plane sizes therefore remain physical
millimetre sizes at every span station. The blade is prismatic and its section
scale is $a(\eta)=1$. Periodic partners satisfy

$$
\mathbf X_R-\mathbf X_L=(0,p,0).
\tag{L3}
$$

The three-dimensional mesh repeats the same section nodes at the selected
span stations and connects corresponding cells. There is no change in blade
scale or passage pitch, so the graph-harmonic boundary correction in Section
[7](#7-extending-the-reference-mesh-across-span) is zero. A physical extrusion
needs no conformal radius or radial-specialized deformation rule. The sweep
reuses unchanged section coordinates whenever $a=1$ and constructs the sparse
LU only if a section actually requires deformation; linear rows therefore
skip the factorization.

Two exports have distinct meanings. A **true 2D linear mesh** maps the
reference nodes to $(X,Y)=(-q_y,q_x)$ and retains the actual section cells.
A **3D linear mesh** adds span nodes and volume connectivity according to
Equation (L2). Both preserve the same periodic pitch. Choosing one span
element makes a thin 3D discretization, but still introduces two span boundary
surfaces; their solver boundary conditions determine whether spanwise
variation is permitted or suppressed. A wall condition at those surfaces
describes finite-height endwalls and is not automatically equivalent to a
planar 2D problem.

In the display, linear blade copies translate by integer multiples of $p$.
Their number is a visualization choice, not a physical machine blade count.
The second geometry/grid panel is an extrusion projection rather than a
radius-versus-axial-position meridional view.

## 6. Generating the two-dimensional reference mesh

### 6.1. CAD boundaries and translational periodicity

The fluid surface is bounded by two periodic sides, inlet and outlet caps, and the blade as an inner boundary. The periodic sides preserve the analytic spline constructions rather than replacing the whole passage with a polygon.

In reference coordinates,

$$
\mathbf q_R=\mathbf q_L+
\begin{bmatrix}p_{\mathrm{ref}}\\0\end{bmatrix}.
\tag{29}
$$

Gmsh receives this translation before mesh generation. The right curves are the slaves and the left curves are their masters. The source first generates the one-dimensional boundary mesh, verifies its periodic pairing, updates the core sizing using actual boundary-edge spacing, and then generates the two-dimensional cells.

Periodicity is checked again after surface meshing. Matching positions alone are insufficient: the code checks bijective node pairing, the affine transformation and corresponding edge connectivity. The five named boundaries are `blade`, `inlet`, `outlet`, `periodic_left` and `periodic_right`.

Gmsh distinguishes model entities from physical groups. Entities describe the CAD or discrete topology; physical groups attach names to collections of entities for output and downstream use. Its mesh-size fields, recombination and periodic API are documented in the [Gmsh reference manual](https://gmsh.info/doc/texinfo/). The particular settings and checks described here are those in [meshing_2d.py](../src/turbo_moc/meshing/meshing_2d.py).

### 6.2. Near-wall inflation and core-size transition

Let $h_1$ be the requested first-layer height, $g>1$ the growth ratio and $H_{\max}$ the maximum requested stack thickness. The nominal layer heights and cumulative thickness are

$$
h_k=h_1g^{k-1},\qquad
H_n=h_1\frac{g^n-1}{g-1},
\qquad
n=\left\lfloor
\frac{\ln[1+H_{\max}(g-1)/h_1]}{\ln g}
\right\rfloor.
\tag{30}
$$

The selected $n$ is the largest whole number of nominal layers that fits within the requested stack. Gmsh's `BoundaryLayer` field creates quadrilateral layers normal to the blade in the reference plane. These are requested dimensions: curvature and neighboring boundaries can shorten or merge local columns.

Away from the blade, the core-size callback provides a graded transition toward the requested domain size. It queries nearby sampled blade segments with a spatial tree, evaluates distances to those segments, and includes periodic blade copies so that the size field is consistent at the two passage sides.

The implemented transition uses a nominal area-growth factor $G_A=1.2$. With local tangential boundary-edge spacing $\ell_t$ and last nominal layer height $h_n$, its starting core size is

$$
h_{\mathrm{start}}=
\min\!\left[
\sqrt{\frac{\ell_t h_nG_A}{f}},
h_nG_A
\right],
\qquad
f=
\begin{cases}
1,&\text{recombined sizing},\\
\sqrt{3}/4,&\text{triangular sizing}.
\end{cases}
\tag{31}
$$

At distance $d$ from the blade, the size is graded by

$$
h_{\mathrm{core}}(d)=
\min\!\left[
h_{\mathrm{domain}},
h_{\mathrm{start}}+
(\sqrt{G_A}-1)\max(0,d-H_n)
\right].
\tag{32}
$$

Before the boundary mesh exists, curvature-based estimates supply $\ell_t$; afterward the callback uses actual boundary-edge lengths. These equations define a sizing heuristic, not a guarantee that every adjacent pair of generated cells has area ratio 1.2.

When inflation is disabled, the transition uses $h_{\mathrm{start}}=\ell_t$ and $H_n=0$ instead of Equation (31).

The default algorithm is Gmsh's Frontal-Delaunay for Quads, followed by recombination. The result is generally quadrilateral-dominant; remaining triangles are allowed. A triangle is not discarded in the three-dimensional stage: it produces a prism instead of a hexahedron.

![Actual conformal reference mesh and inflation detail](figures/conformal_reference_mesh.png)

**Figure 2.** An actual Gmsh reference mesh generated around the fitted blade, with an enlarged view of the near-wall layers. The requested values $h_1=0.03$ mm, $g=1.3$ and $H_{\max}=0.28$ mm give five nominal layers with $H_5=0.271293$ mm in the length-valued reference plane. Axes are conformal coordinates; the grid is a CFD discretization, not the characteristic network used to design the nozzle. The figures use reduced cell counts for readability rather than a mesh-convergence study; their exact inputs are recorded in [figure_metadata.json](figures/figure_metadata.json).

### 6.3. Physical interpretation of mesh-size controls

Reference mesh sizes are specified in millimetres in $\mathbf q$ space. They become physical sizes through Equation (10), together with the span scale. For a protected blade-layer spacing,

$$
h_{\mathrm{physical,estimate}}
\simeq h_1a(\eta)\frac{r}{R_{\mathrm{ref}}}.
\tag{33}
$$

In a radial channel this estimate decreases with radius for inflow. In a constant-chord axial channel, $a\,r/R_{\mathrm{ref}}=1$, so the leading-order in-surface layer height remains constant.

Equation (33) describes a small offset within a prescribed section surface. At a tapered three-dimensional blade, that direction is not generally the normal to the complete blade wall. Consequently, the code's sampled height range is neither an exact three-dimensional wall-normal height nor a prediction of $y^+$.

The usual near-wall quantity,

$$
y^+=\frac{\rho u_\tau\Delta n}{\mu},
$$

also requires wall-normal distance $\Delta n$, wall friction velocity $u_\tau$, density $\rho$ and viscosity $\mu$. These flow quantities must be estimated or obtained from CFD. Mesh-size inputs alone cannot establish wall resolution for all operating conditions.

## 7. Extending the reference mesh across span

### 7.1. Why section scaling requires mesh deformation

For annular rows, at each span station, the blade follows Equation (19), while the passage retains the pitch in Equation (21). Simply scaling every mesh node would correctly scale the blade but would also scale the distance between the periodic sides. Leaving every node unchanged would preserve pitch but fail to follow a blade whose conformal dimensions vary across span.

The mesh therefore needs an intermediate operation: prescribe the blade and passage boundaries at their target positions, then move the intervening core nodes smoothly without changing connectivity.

This operation is a geometric **mesh deformation**. It does not solve the fluid momentum equations and does not model elastic deformation of the blade.

### 7.2. Protected blade collar and prescribed outer boundaries

Let $\mathbf q_i^0$ denote node $i$ of the reference mesh. The initial similarity-scaled position is

$$
\widetilde{\mathbf q}_i=a(\eta)\mathbf q_i^0.
\tag{34}
$$

Nodes near the blade are protected so that the inflation layer structure follows the blade. A `cKDTree` query measures the distance to the nearest sampled blade-boundary node,

$$
d_i=\min_{b\in B_{\mathrm{blade}}}
\|\mathbf q_i^0-\mathbf q_b^0\|.
$$

The nominal collar cutoff is $d_0=1.5H_n+h_{\mathrm{blade}}$, where
$h_{\mathrm{blade}}$ is the reference boundary-size target. Its extra core
buffer may exceed the available space near a shared interface even though
the prescribed inflation stack fits. Let $\ell_{\max}$ be the longest sampled
blade edge and $d_{\mathrm{outer}}$ the smallest wall-node distance among
outer-boundary nodes. The implementation uses the sampling allowance
$d_{\mathrm{req}}=\sqrt{H_n^2+(\ell_{\max}/2)^2}$ when inflation is enabled,
or zero otherwise, and sets

$$
d_{\mathrm{protect}}=\max\!\left[
d_{\mathrm{req}},\min\!\left(d_0,
\frac{d_{\mathrm{req}}+d_{\mathrm{outer}}}{2}\right)\right].
$$

Nodes satisfying $d_i\le d_{\mathrm{protect}}$ move with the blade. The
sampling allowance accounts for normal layer nodes midway along a sampled
wall edge; it remains a practical sampled-distance rule, rather than an
exact distance calculation against the continuous spline. If
$d_{\mathrm{outer}}\le d_{\mathrm{req}}$, generation is rejected with a
clearance error. This adjustment changes the extra core buffer, not the
requested inflation heights or thickness. Cell validity is checked after
each section deformation and again after the volume is assembled.

The collar retains Equation (34). Outer boundaries receive the coordinates prescribed by the target passage:

- Ordinary inlet/outlet nodes retain their normalized pitchwise fraction
  between the cap endpoints. A shared physical cap instead fixes its normal
  coordinate $q_y$ while allowing its interior nodes to slide in $q_x$ through
  the harmonic solve. Its periodic corners remain fixed.
- Similar periodic sides use the exact similarity transform of their CAD
  nodes. When shared caps reshape the extensions, nodes retain the spline
  parameter within each corresponding CAD segment. This preserves the
  blade-adjacent joins while changing only the relevant extension.

This explicit correspondence keeps opposite periodic nodes paired. It does not use a nearest-point projection that could reorder stations.

For a sliding shared cap, the free/constrained partitions differ between
$q_x$ and $q_y$. Each coordinate uses the same graph-harmonic equations
below with its own partition. Two sparse factorizations are then reused
across span, instead of one factorization for both coordinate columns.

### 7.3. A graph-harmonic correction

The reference cells define a graph: each node is a graph vertex and each unique cell edge joins a pair of graph vertices. For neighboring nodes $i,j$, the weight is

$$
w_{ij}=\frac{1}{\|\mathbf q_i^0-\mathbf q_j^0\|^2}.
$$

The graph Laplacian is

$$
L_{ii}=\sum_{j\in\mathcal N(i)}w_{ij},\qquad
L_{ij}=-w_{ij}\quad\text{for neighboring }i\ne j,
\tag{35}
$$

with zero entries for non-neighboring pairs. If $\mathbf d_i$ is the displacement correction, a free node satisfies

$$
\sum_{j\in\mathcal N(i)}
w_{ij}(\mathbf d_i-\mathbf d_j)=0.
\tag{36}
$$

Thus its displacement is the positive weighted average of neighboring displacements. This is a graph-harmonic model; it is not a finite-element discretization of linear elasticity or an exact continuous Laplace problem in physical space.

Equivalently, the correction minimizes the graph energy

$$
E(\mathbf d)=\frac12
\sum_{\{i,j\}\in\mathcal E}
w_{ij}\|\mathbf d_i-\mathbf d_j\|^2
\tag{37}
$$

subject to the prescribed boundary displacements. This interpretation explains why the core motion is smooth: large differences between neighboring displacements increase the energy.

Partition the nodes into free nodes $F$ and prescribed nodes $B$. The solve is

$$
L_{FF}\mathbf d_F=-L_{FB}\mathbf d_B,\qquad
\mathbf d_B=
\mathbf q_B^{\mathrm{target}}-
\widetilde{\mathbf q}_B,
\qquad
\mathbf q_i=\widetilde{\mathbf q}_i+\mathbf d_i.
\tag{38}
$$

The collar has zero correction relative to Equation (34). The outer-boundary corrections account for the difference between the scaled mesh and the fixed-pitch target passage.

### 7.4. Why sparse LU is used and reused

Each node is connected to only a small number of neighbors. Most entries of $L$ are therefore zero, so storing a dense matrix would waste memory and computation. Set $A=L_{FF}$. The implementation uses sparse matrices and factors $A$ with SciPy's sparse LU routine:

$$
P_r A P_c=L_{\mathrm{LU}}U_{\mathrm{LU}}.
\tag{39}
$$

Here $P_r,P_c$ are permutations, while $L_{\mathrm{LU}}$ and $U_{\mathrm{LU}}$ are the lower and upper triangular factors. These factor names explain the abbreviation LU; they are distinct from the graph Laplacian $L$. The permutations support an efficient, stable sparse factorization.

Once the factors exist, each solve consists of triangular substitutions. There is no need to form a matrix inverse. The same factors serve both coordinate right-hand sides and every span station because the graph, reference-edge weights and free/prescribed partition remain unchanged.

The constrained system is nonsingular when every free graph component is connected to prescribed nodes. The mesh construction supplies that connection in the intended passage topology. A failed factorization would indicate that the deformation problem cannot be solved as posed.

LU is a linear-solver choice, not an additional physical assumption. With the positive graph weights and each free component anchored, $A$ is symmetric positive definite, so sparse Cholesky or a suitable iterative method would also be possible. The current implementation uses the available sparse LU solver and amortizes its factorization cost over the many coordinate and span right-hand sides.

The deformation object is created only if some station has $a(\eta)\ne1$. A pure radial channel with the present parallel endwalls bypasses the solve. A constant-chord axial channel generally does not, as shown in Section 5.1.

After deformation, the code verifies

$$
\mathbf q_i^R-\mathbf q_i^L=
\begin{bmatrix}p_{\mathrm{ref}}\\0\end{bmatrix}
$$

for every periodic pair. It also checks a positive corner cross product at each triangle or quadrilateral corner. Positive total polygon area alone would not exclude a folded quadrilateral.

![Reference, similarity-scaled and corrected section meshes](figures/conformal_mesh_deformation.png)

**Figure 3.** Mesh deformation for an axial channel with 20/30 mm hub/tip chords and 24 blades. The illustrated tip scale is $a(1)=1.14545$. The similarity-only mesh moves the blade correctly but changes periodic pitch. Prescribing the target outer boundaries and solving the graph-harmonic correction restores one machine pitch while preserving the protected blade collar. The intermediate similarity-only panel is an explanatory construction, not an accepted CFD section.

The numerical implementation is `CoreDeformation` in [meshing_3d.py](../src/turbo_moc/meshing/meshing_3d.py). The `volume_mesh` span loop in the same module reuses it; it does not regenerate the reference mesh.

## 8. Physical nodes, volume cells and Gmsh topology

### 8.1. Span stations and the meridional mesh preview

For $N_s$ volume layers, the code creates $N_s+1$ span stations. Let $t_j=j/N_s$. With clustering parameter $\lambda_s$, the stations are

$$
\eta_j=
\begin{cases}
t_j,&\lambda_s\simeq0,\\[1mm]
\dfrac12\left[
1+\dfrac{\tanh[\lambda_s(2t_j-1)]}{\tanh\lambda_s}
\right],&\lambda_s>0.
\end{cases}
\tag{40}
$$

The clustering is symmetric toward hub and shroud. These are prescribed span stations; they are not additional blade profiles.

The meridional mesh view projects streamwise stations from the actual reference periodic-left boundary onto the channel at these $\eta_j$. It provides a preview of the sweep distribution without constructing all volume cells. The view remains empty until a current reference mesh exists.

The projected grid is not an independently generated meridional CFD mesh. For general curved channels, its lines need not be orthogonal, and spacing shown in the meridional plane should not be interpreted as the complete three-dimensional cell dimensions.

### 8.2. Connecting adjacent mapped sections

After section deformation, Equation (8) maps every node at every span station into physical space. Let $n$ be the number of reference nodes. For zero-based section-node index $i$ and span index $j$, the one-based Gmsh node tag is

$$
I(i,j)=i+jn+1.
\tag{41}
$$

A reference quadrilateral $(a,b,c,d)$ creates a hexahedron between stations $j$ and $j+1$:

$$
\big[
I(a,j),I(b,j),I(c,j),I(d,j),
I(a,j+1),I(b,j+1),I(c,j+1),I(d,j+1)
\big].
\tag{42}
$$

A triangle creates a six-node prism in the same way. Sweeping a section boundary edge creates a quadrilateral boundary face. The hub and shroud use the reference cell faces at the first and last stations.

The counts follow directly:

$$
N_{\mathrm{nodes}}=n(N_s+1),\qquad
N_{\mathrm{hex}}=N_{\mathrm{quad}}N_s,\qquad
N_{\mathrm{prism}}=N_{\mathrm{tri}}N_s.
\tag{43}
$$

Connectivity, not node positions alone, tells a CFD solver which nodes form a cell and which cells share a face. The construction preserves the reference topology at every station, so there is no interpolation between unrelated section meshes.

The final cells are first-order hexahedra and prisms. A smooth underlying coordinate map does not turn these cells into curved high-order elements; their faces and edges are represented by their mapped vertices.

### 8.3. Orientation and rotational periodicity

Node ordering must be consistent with the physical map. The sign of its geometric Jacobian follows from Equation (4):

$$
\det\frac{\partial(X,Y,Z)}{\partial(q_x,q_y,\eta)}
=-\frac{r^2}
{R_{\mathrm{ref}}^2\|\mathbf P_u\|}
D.
\tag{44}
$$

The implementation uses $-\operatorname{sign}(D)$ to choose the section winding for volume assembly. This handles radial inflow and outflow through the wall orientation, rather than through a special machine-type rule. The hub face winding is opposite to the shroud face winding.

Equation (29) gives equal $q_y$ on a periodic pair. Therefore, the pair has equal $u,r,z$ after mapping, and its physical positions satisfy

$$
\mathbf X_R=
R_Z(-\Delta\theta)\mathbf X_L,
\qquad \Delta\theta=2\pi/N_b.
\tag{45}
$$

$R_Z$ denotes the standard rotation matrix about $Z$. The minus sign follows from clockwise $\theta$.

The code checks the entire swept node correspondence and verifies the rotational match to a tolerance proportional to $R_{\mathrm{ref}}$. Checking only the hub and shroud would miss pairing errors at interior span stations.

### 8.4. Registering the assembled mesh in Gmsh

The sweep constructs node coordinates and connectivity explicitly. It does not call Gmsh's three-dimensional meshing algorithm or its CAD extrusion routine.

To make this data a Gmsh model, `assemble_volume`:

1. Creates seven discrete surface entities and one discrete volume entity.
2. Inserts the mapped nodes and the hexahedral/prismatic volume connectivity.
3. Inserts the swept boundary faces and hub/shroud faces.
4. Reclassifies nodes onto the appropriate entities and constructs the discrete topology.
5. Assigns physical names to the surfaces and to `fluid_domain`.
6. Registers the left-to-right periodic transformation: rotational for annular rows, translational for linear rows.

The seven surfaces are `blade`, `inlet`, `outlet`, `periodic_left`, `periodic_right`, `hub` and `shroud`. They bound the fluid passage; they are not a separate mesh of the solid blade.

The saved MSH file contains this physical volume mesh. Browser boundary datasets are extracted from the same connectivity before Gmsh closes.

![Actual swept volume boundary meshes](figures/swept_volume_mesh.png)

**Figure 4.** Boundary surfaces extracted from actual axial and radial volume meshes, with blade and hub shown and shroud hidden to expose the passage. Three rotated display copies illustrate the cascade. The generated and exported computational domain is one periodic passage; display copies do not multiply its cell count.

## 9. Mesh checks and interpretation of the quality report

### 9.1. Element validity

For an element interpolated from nodal coordinates,

$$
\mathbf X(\boldsymbol\xi)
=\sum_aN_a(\boldsymbol\xi)\mathbf X_a,
\qquad
J=\frac{\partial\mathbf X}{\partial\boldsymbol\xi},
$$

a negative determinant indicates an inverted element, while a zero determinant indicates degeneracy. The volume check rejects nonfinite or nonpositive Gmsh minimum determinant and minimum signed inverse-condition measures.

These checks complement the planar corner tests. A valid deformed section can still produce an invalid volume if adjacent span stations are too far apart or the physical channel bends strongly.

### 9.2. Three explicitly defined volume metrics

The report in [quality.py](../src/turbo_moc/meshing/quality.py) computes the following geometric quantities for first-order hexahedra and prisms.

**Equiangular face skewness.** For face $f$ with corner angles $\alpha$, define

$$
S_f=\max\!\left[
\frac{\alpha_{\max}-\alpha_*}{180^\circ-\alpha_*},
\frac{\alpha_*-\alpha_{\min}}{\alpha_*}
\right],
\qquad
S_K=\max_{f\subset K}S_f,
\tag{46}
$$

where $\alpha_*=60^\circ$ for triangles and $90^\circ$ for quadrilaterals. The reported result is clipped to $[0,1]$. Lower values indicate smaller angular departures from the ideal face.

**Orthogonality.** With face normal $\mathbf n_f$ and centroid connector $\mathbf d_f$,

$$
O_K=\min_{f\subset K}
\frac{|\mathbf n_f\cdot\mathbf d_f|}
{\|\mathbf n_f\|\|\mathbf d_f\|}.
\tag{47}
$$

Interior faces use the connector between adjacent cell centroids; exterior faces use the connector from cell centroid to face centroid. At periodic faces, the neighbor centroid is transformed into the local passage frame before evaluating the connector: rotated for an annular row, translated for a linear row. Higher values indicate better alignment.

**Edge aspect ratio.**

$$
A_K=\frac{\max_{e\subset K}\ell_e}
{\min_{e\subset K}\ell_e}.
\tag{48}
$$

The centroids in the three-dimensional report are arithmetic means of vertex positions. These definitions are explicit implementation choices and should not be assumed identical to similarly named Fluent quantities. The richer two-dimensional report also uses a different orthogonality definition based on signed outward edge normals and polygon-area centroids.

A large aspect ratio can be deliberate in a near-wall layer. It should be interpreted together with orientation, skewness, orthogonality, the local flow direction and the required physical resolution. A favorable histogram does not demonstrate mesh independence; that requires comparison of the relevant CFD results on systematically refined meshes.

### 9.3. Boundary-face auditing

Before CGNS export, the code enumerates the faces of all volume cells and identifies each face by its node set. A face with one owner is exterior; a face with two owners is interior; more than two owners indicates nonmanifold connectivity.

The audit verifies that the seven named patches cover the complete exterior exactly once and that every boundary face is oriented away from its owner-cell center. It matches full faces, rather than inferring a boundary from a few vertices that happen to touch a wall.

This distinguishes three questions: are the cells locally valid, is periodic correspondence complete, and does the exported boundary partition represent the volume exterior correctly? Each requires its own check.

## 10. Exporting a solver-readable CGNS mesh

### 10.1. CGNS structures and the ADF container

CGNS provides a hierarchy for coordinates, cells, boundary conditions and interface connectivity. The hierarchy and element-numbering conventions are specified by the [CGNS Standard Interface Data Structures](https://cgns.org/cgns-archives/CGNS_docs_current/sids/gridflow.html). The current exporter constructs that hierarchy and writes an ADF container using [cgns_adf.py](../src/turbo_moc/io/cgns_adf.py); it does not use Gmsh's generic CGNS export.

The ADF reader/writer handles node labels, array types and payloads. The mesh exporters decide what those arrays mean. ADF and HDF5 are different storage containers; this implementation writes ADF.

For three-dimensional output, the principal tree is:

```text
CGNSLibraryVersion = 3.4
TurboMoC [CGNSBase_t: cell dimension 3, physical dimension 3]
  DataClass = Dimensional
  DimensionalUnits
  fluid_family [Family_t]
  fluid_domain [Zone_t: Nnodes, Nvolume_cells, 0]
    ZoneType = Unstructured
    FamilyName = fluid_family
    GridCoordinates
      CoordinateX, CoordinateY, CoordinateZ
    fluid_domain [Elements_t]
      ElementRange
      ElementConnectivity
    blade, inlet, outlet, periodic_left, periodic_right, hub, shroud
      [Elements_t with their own ranges and connectivity]
    ZoneBC
      Each named patch [BC_t]
        GridLocation = FaceCenter
        PointRange = first/last boundary element ID
    ZoneGridConnectivity
      right_to_left, left_to_right [GridConnectivity_t]
        GridConnectivityType = Abutting1to1
        GridLocation = Vertex
        PointList, PointListDonor
        GridConnectivityProperty/Periodic
          RotationCenter, RotationAngle, Translation
```

This is the actual exporter structure in [meshing_3d.py](../src/turbo_moc/meshing/meshing_3d.py), including its fluid family, named boundary sections and reciprocal periodic maps.

### 10.2. Renumbering, element types and one fluid cell zone

Internal NumPy connectivity is zero-based. Gmsh node tags and CGNS connectivity are one-based. Moreover, Gmsh tags are not necessarily consecutive. The exporter sorts the Gmsh node tags and builds a contiguous CGNS numbering. For Gmsh tag $t$,

$$
I_{\mathrm{CGNS}}(t)=
1+\operatorname{position}(t,\text{sorted node tags}).
\tag{49}
$$

This conversion is essential: a valid node tag is not automatically the correct array index in an exported coordinate array.

| Cell or boundary element | Gmsh type | CGNS type |
| --- | ---: | --- |
| Line | 1 | `BAR_2` = 3 |
| Triangle | 2 | `TRI_3` = 5 |
| Quadrilateral | 3 | `QUAD_4` = 7 |
| Prism | 6 | `PENTA_6` = 14 |
| Hexahedron | 5 | `HEXA_8` = 17 |

Hexahedra and prisms are written into one combined volume element section. When both are present, its CGNS type is `MIXED` = 20. The exporter declares CGNS 3.4 and uses the corresponding type-prefixed connectivity representation: each mixed element begins with its type code, followed by its node IDs.

One combined volume section is intentional. Fluent can create cell zones from element sections; separating hexahedra and prisms into independent sections can split what should be one fluid region.

The volume cells receive the first element IDs. Boundary face IDs follow them. A BC's face range refers to those boundary **element IDs**, not to node IDs. The zone's cell count includes volume cells, not the additional surface elements.

### 10.3. Units and the two-dimensional exporter

The default length conversion is

$$
\mathbf X_{\mathrm{CGNS}}
=10^{-3}\mathbf X_{\mathrm{mesh}},
\tag{50}
$$

so millimetres become metres. The CGNS units are kilogram, metre, second, kelvin and radian. This scaling changes the exported coordinates; it does not rescale the existing MSH file.

The two-dimensional exporter follows the same principles, with differences dictated by cell dimension:

| Item | Two-dimensional mesh | Three-dimensional mesh |
| --- | --- | --- |
| Base dimensions | $[2,2]$ | $[3,3]$ |
| Coordinates | $X,Y$ in the planar cascade | Physical $X,Y,Z$ |
| Fluid zone | `fluid` | `fluid_domain` |
| Fluid elements | Triangles/quadrilaterals | Prisms/hexahedra |
| Boundary elements | Lines | Triangles/quadrilaterals |
| BC location | `EdgeCenter` | `FaceCenter` |
| Periodic transformation | Translation | Rotation |

The planar boundary sections additionally record owner-cell IDs and local edge positions in `ParentElements` and `ParentElementsPosition`. This helps downstream importers associate boundary edges with their fluid cells.

The length-valued reference plane used for a three-dimensional sweep is an intermediate coordinate system. Exporting that plane as a two-dimensional physical domain does not reproduce the radius-varying physical radial section. The independent planar Fluent workflow instead uses the physical planar cascade generated by the two-dimensional example.

A project row can also export its 2D reference directly through
`export_reference_mesh`. For a linear row it writes physical $(X,Y)$ and
translational periodicity. For an annular row it writes the length-valued
conformal reference plane, which is a construction domain rather than the
complete physical annular flow. Generating this export neither regenerates
the 2D mesh nor requires a 3D volume.

### 10.4. Periodic metadata: direction matters

Gmsh's periodic map is stored from left master to right slave:

$$
\mathbf X_R=R_Z(-\Delta\theta)\mathbf X_L.
$$

CGNS periodic metadata describes the transformation from the current interface to its donor. For annular rows, the exporter writes

$$
\text{right to left}: +\Delta\theta,\qquad
\text{left to right}: -\Delta\theta.
\tag{51}
$$

The two-dimensional equivalent is a translation of $(-p,0)$ from right to left and $(+p,0)$ from left to right, expressed in the exported units. Reciprocal point lists retain the node bijection.

For a linear row, the master-to-slave displacement is $(0,p,0)$. The
left/current-to-right/donor translation is therefore $(0,p,0)$, and the
reciprocal translation is $(0,-p,0)$; both rotations are zero. Blade quality
neighbours use this same translational relationship.

These direction conventions follow the [CGNS periodic connectivity interface](https://cgns.org/cgns-archives/CGNS_docs_current/midlevel/connectivity.html). Reversing master/slave or current/donor without reversing the transformation is a common source of an apparently correct geometry with an incorrect interface.

### 10.5. What is established before Fluent runs

The three-dimensional exporter audits exterior faces before writing, reloads the ADF file afterward, and checks base/zone dimensions and the named BC types, locations and ranges. The two-dimensional exporter checks its topology before writing but does not perform this serialized-file readback.

The current export retains named BC patches on the periodic sides alongside vertex-level periodic connectivity, so downstream solver setup can identify the two face zones. This is an importer-oriented representation. The [CGNS SIDS](https://cgns.org/standard/SIDS/multizone.html) expresses a design intent to separate boundary-condition coverage from interface-connectivity coverage. Acceptance of the combined representation should therefore be checked with the target reader. The local ADF readback verifies stored data rather than every SIDS rule or a solver's interpretation; it does not constitute an independent CGNS conformance assessment.

The existing [two-dimensional Fluent workflow](../src/turbo_moc/fluent/workflow_2d.py) imports the CGNS mesh, checks it, identifies the named zones, assigns the relevant boundary types, and verifies a one-pitch translational correspondence between imported periodic vertices. It reuses an imported periodic pair or explicitly creates a conformal translational pair.

Thus three steps have different meanings: exporting mesh metadata, importing a mesh into a solver, and configuring that solver's periodic boundary condition. A mesh file alone is not a complete CFD case.

The three-dimensional exporter supplies audited connectivity, boundary names and periodic metadata (rotational or translational) intended for the Fluent import workflow. Automated three-dimensional Fluent setup has not been implemented; [fluent_3D/](../examples/fluent_3D/readme.md) remains a placeholder. Periodic pairing and the imported cell/boundary zones still need target-solver acceptance checks before a three-dimensional simulation can be treated as configured.

## 11. Interpreting the app views and numerical assumptions

The app has two workspaces, Geometry generation and Mesh generation. The selected row supplies
its section, generation actions and quality report; the sidebar shows
collapsible geometry/meshing settings for all active rows. Shared physical figures assemble
available row geometry. Row IDs accompany jobs and exports. Changing the
selected row does not change the mathematical basis of a stored artifact.
An annular conformal section and a physical linear section are labeled
differently because their coordinates have different meanings.

### 11.1. Axial interspace and radial clearance

Assembly placement remains distinct from row-local scientific geometry and
mesh nodes. In a paired axial machine, the rotor input
`geometry.axial_interspace_mm` specifies a nonnegative minimum clearance $g_z$
from the upstream stator trailing-edge envelope to the rotor leading-edge
plane. Let $z_{S,\mathrm{LE}}$ be the stator LE origin and let $C_{S,h}$ and
$C_{S,t}$ be its hub and tip meridional chords. The farthest prescribed TE is

$$
z_{S,\mathrm{TE}}^{\max}
=z_{S,\mathrm{LE}}+\max(C_{S,h},C_{S,t}).
$$

The derived rotor origin is

$$
z_{R,\mathrm{LE}}
=z_{S,\mathrm{TE}}^{\max}+g_z,
\qquad g_z\geq0.
$$

Using the greater chord ensures that taper cannot reduce the minimum
blade-envelope gap below the requested value. For a 20/30 mm upstream
hub/tip chord and $g_z=5$ mm, a stator LE origin of 0 mm gives the rotor LE
at 35 mm: the tip gap is 5 mm and the hub gap is 15 mm. Relative rotor
placement is derived from this input, rather than a competing manual offset.
For new inputs, the default is 10% of the rotor midspan chord,
$g_z=0.1(C_{R,h}+C_{R,t})/2$. This initializes a length in millimetres;
subsequent chord edits do not silently change it. Existing explicit values
are preserved. The gap is inactive when no upstream row participates, as in
rotor-only mode. A solitary axial row retains its existing display offset.

For radial rows, the prescribed radii already locate both rows. Let
$r_{S,\mathrm{in}}$ and $r_{S,\mathrm{out}}$ be the upstream LE/TE radii, and
let $r_{R,\mathrm{in}}$ be the downstream LE radius. Define

$$
s_r=\operatorname{sign}
\left(r_{S,\mathrm{out}}-r_{S,\mathrm{in}}\right),
\qquad
g_r=s_r\left(r_{R,\mathrm{in}}-r_{S,\mathrm{out}}\right).
$$

Then $g_r>0$ denotes clearance, $g_r=0$ denotes touching envelopes and $g_r<0$
denotes overlap for either inflow or outflow. For inflow from 100 to 80 mm,
a downstream LE at 75 mm gives $g_r=5$ mm; an LE at 85 mm gives $g_r=-5$ mm.
No extra radial gap input or radial Cartesian translation is introduced.
Planning overlaps are reported for inspection rather than interpreted as a
valid coupled flow domain.

`assembly_placement` supplies the resolved row origins and angular phases;
`assembly_interspaces` supplies the signed adjacent-row diagnostic. Before a
row has a current profile solution, the meridional view shows its prescribed
LE/TE envelope. Once computed, both stator and rotor blades use the same
physical mapping and mesh sweep.

The adjacent fluid domains terminate on one shared physical interface.
Generation requires strictly positive clearance. For axial rows, this is
the plane halfway across the minimum blade gap:

$$
z_I=z_{S,\mathrm{TE}}^{\max}+\frac{g_z}{2}.
$$

Its upstream outlet and downstream inlet extensions are therefore

$$
\ell_{S,\mathrm{out}}(\eta)
=z_I-z_{S,\mathrm{LE}}-C_S(\eta),\qquad
\ell_{R,\mathrm{in}}=z_{R,\mathrm{LE}}-z_I.
$$

For the 20/30 mm stator example above, $z_I=32.5$ mm. The stator extensions
are 12.5 mm at the hub and 2.5 mm at the tip; the rotor inlet extension is
2.5 mm. Applying one fixed outlet chord fraction across the span would not
produce this common plane. Instead, at each span station the implementation
inverts the physical target into the row's conformal chart. For an axial
channel with radius $r(\eta)$ and row origin $z_{\mathrm{LE}}$,

$$
m'_I(\eta)=\frac{z_I-z_{\mathrm{LE}}}{r(\eta)},\qquad
q_{y,I}(\eta)=-R_{\mathrm{ref}}m'_I(\eta).
$$

For radial rows, the interface is a cylinder with

$$
r_I=\frac{r_{S,\mathrm{out}}+r_{R,\mathrm{in}}}{2},\qquad
m'_I=s_r\ln\!\left(\frac{r_I}{r_{\mathrm{LE}}}\right),\qquad
q_{y,I}=-R_{\mathrm{ref}}m'_I.
$$

The logarithm is evaluated with each row's own leading-edge radius and
reference radius. Thus the two conformal cap coordinates generally differ,
although their physical surfaces coincide. The same expression covers
radial inflow and outflow.

Only the inlet/outlet extension splines are rebuilt against these targets.
The fitted blade, its span scaling and the single reference placement are
retained. The extension handles retain the prescribed tangent directions
and length ratios. During mesh deformation, nodes on each periodic CAD
segment retain that segment's spline parameter; stretching the entire
passage by a global station fraction would move its blade-adjacent joins.
The harmonic core solve then extends the prescribed boundary displacement
into the section interior.

Current channel families require the same axial hub radius and blade height,
or the same radial blade height and flow direction, in both rows. An unequal
endwall transition needs its own geometry definition. Incompatible inputs
remain editable but cannot generate a paired domain. The external stator
inlet and rotor outlet remain user-defined. Saved internal chord fractions
are inactive in a pair and restored for standalone row operation.

Computational arrays remain local to each row. Paired CGNS export applies
the physical assembly placement before converting millimetres to metres:

$$
\mathbf X_{\mathrm{CGNS}}
=10^{-3}\left[R_Z(\phi)\mathbf X_{\mathrm{local}}
+z_{\mathrm{LE}}\mathbf e_Z\right].
$$

For radial rows the axial translation is zero. The exporter checks every
internal boundary node against $z_I$ or $r_I$ and records the boundary pair.
Different blade counts and section meshes need not produce matching nodes;
pitch-change treatment, nonconformal interface coupling and rotating frames
remain solver setup. The optional inlet velocity-triangle coupling is a
separate flow-state calculation, not a solution of the interspace flow.
Gap edits retain valid MoC/vortex profile solutions but invalidate both
affected domains and meshes.

### 11.2. Meaning of the displayed views

The app presents different representations because each answers a different question:

| Representation | Question it answers |
| --- | --- |
| Conformal blade/passage geometry | What section and periodic passage are being designed? |
| Meridional geometry | Where are the blade and its leading/trailing stations relative to the rotation axis? |
| Three-dimensional geometry preview | What does the mapped blade family look like before meshing? |
| Reference section mesh | How are the blade wall, inflation stack and passage core discretized? |
| Meridional sweep projection | How are actual section boundary stations distributed across span? |
| Three-dimensional mesh boundaries | What exterior surfaces belong to the assembled physical volume mesh? |

The geometry preview uses display tessellation independent of CFD mesh sizing. It includes caps at blade ends and a continuous hub surface of revolution to improve visibility. These display surfaces do not modify the CFD boundary topology. Conversely, the mesh viewer extracts faces from the actual volume connectivity; it is not another geometric remeshing.

Three assumptions deserve particular attention when interpreting the design:

1. **Prescribed stream surfaces.** Hub/shroud interpolation defines the channel geometry. It does not solve radial equilibrium or secondary flow.
2. **A single similar section family.** Span scaling preserves conformal section angles but does not introduce independently optimized spanwise profiles.
3. **A geometric transfer of a planar design.** The mapping provides a realizable three-dimensional blade and mesh. The aerodynamic consequences of the imposed pitch, channel shape and span variation remain a CFD or experimental question.

These distinctions preserve the role of the MoC as a preliminary design tool while making the subsequent validation requirements clear.

## 12. Student calculations and implementation map

### 12.1. Suggested calculations

The following calculations connect the equations to the implementation.

1. **Recover an axial blade chord.** For the axial example, take $q_y=-20$ mm at reference span. Equation (24) gives $Z=20$ mm. At the hub, first apply $a=1.05$, then map $q_y=-21$ mm; the same physical chord follows.
2. **Locate the radial trailing edge.** For inflow, insert $q_y=-L_{\mathrm{ref}}$ into Equation (27). Show that $r=80$ mm rather than $100-L_{\mathrm{ref}}$ mm.
3. **Check periodic closure.** Add one $p_{\mathrm{ref}}$ to any conformal point's $q_x$. Verify that $r,z$ are unchanged and the mapped point rotates by $-2\pi/N_b$.
4. **Predict cell counts.** A reference mesh with 100 quadrilaterals, two triangles and 150 nodes, swept through 20 layers, produces 2,000 hexahedra, 40 prisms and 3,150 nodes.
5. **Interpret a layer target.** For a radial inflow blade, compare the physical in-surface estimate at inlet and outlet for $h_1=0.03$ mm. The radius ratio gives $h_{\mathrm{out}}/h_{\mathrm{in}}=0.8$.
6. **Explain the sparse solve.** Identify why changing span count does not change $L_{FF}$, while changing the reference mesh generally does. Relate this to the benefit of tuning the section mesh before generating the volume.
7. **Recover a linear extrusion.** For a 20 mm chord and 10 mm height, map a section point $(q_x,q_y)=(5,-20)$ mm at $\eta=0.25$. Equation (L2) gives $(X,Y,Z)=(20,5,2.5)$ mm. Add one pitch to $q_x$ and verify Equation (L3). Explain why changing blade height cannot alter the 2D section mesh or its in-plane first-layer target.

For a further exercise, construct curved hub and shroud B-splines in Python and compare the numerical inversion of Equation (14) with Equation (13) as the curves approach straight lines. Then inspect whether the chosen station correspondence preserves the sign of $D$.

### 12.2. Where the equations enter the source

| Operation | Main source and functions |
| --- | --- |
| Fresh MoC nozzle solve and wall validation | [moc.py](../src/turbo_moc/geometry/moc.py): `solve_nozzle` |
| Planar blade fitting | [blade.py](../src/turbo_moc/geometry/blade.py): `parametrize_stator_blade` |
| Passage construction/placement | [geometry_2d.py](../src/turbo_moc/geometry/geometry_2d.py): `generate_flow_domain`, `assess_clearance` |
| General meridional channel and inverse map | [meridional_channel.py](../src/turbo_moc/geometry/meridional_channel.py): `MeridionalChannel`, `prime_chord`, `parameter_from_prime`, `map_points` |
| Linear physical mapping | [linear_cascade.py](../src/turbo_moc/geometry/linear_cascade.py): `LinearCascadeDefinition`, `LinearCascadeMapping`, `build_linear_section` |
| Single reference design/span sections | [geometry_3d.py](../src/turbo_moc/geometry/geometry_3d.py): `build_design`, `Design.section` |
| Shared physical interface targets | [configuration_project.py](../src/turbo_moc/configuration_project.py): `shared_interfaces`, `row_assembly`; [flow_interfaces.py](../src/turbo_moc/geometry/flow_interfaces.py): `interface_prime`, `apply_interfaces` |
| Reference CAD, inflation and sizing | [meshing_2d.py](../src/turbo_moc/meshing/meshing_2d.py): `add_blade_boundary`, `add_periodic_passage`, `calculate_inflation_profile`, `configure_core_transition` |
| Reference assembly | [meshing_3d.py](../src/turbo_moc/meshing/meshing_3d.py): `generate_section` |
| Harmonic mesh correction | [meshing_3d.py](../src/turbo_moc/meshing/meshing_3d.py): `CoreDeformation` |
| Physical volume connectivity/validation | [meshing_3d.py](../src/turbo_moc/meshing/meshing_3d.py): `assemble_volume`, `validate_volume` |
| Reusable stages and artifact files | [meshing_3d.py](../src/turbo_moc/meshing/meshing_3d.py): `reference_mesh`, `volume_mesh`, `export_mesh` |
| Shared cell/face/edge geometry and plotting segments | [quality.py](../src/turbo_moc/meshing/quality.py): `cell_geometry`, `edge_topology`, `face_geometry`, `unique_plot_segments` |
| Volume report definitions | [quality.py](../src/turbo_moc/meshing/quality.py): `analyze_volume` |
| 3D boundary audit/CGNS hierarchy | [meshing_3d.py](../src/turbo_moc/meshing/meshing_3d.py): `_face_audit`, `export_cgns`, `add_cgns_periodicity` |
| 2D CGNS hierarchy | [meshing_2d.py](../src/turbo_moc/meshing/meshing_2d.py): `export_fluent_cgns` |
| Row-scoped 2D reference export | [meshing_3d.py](../src/turbo_moc/meshing/meshing_3d.py): `export_reference_mesh` |
| Ordered machine/row configuration | [configuration_project.py](../src/turbo_moc/configuration_project.py): `load_project`, `normalize_project`, `row_configuration`, `assembly_placement`, `assembly_interspaces` |
| ADF storage | [cgns_adf.py](../src/turbo_moc/io/cgns_adf.py): `AdfNode`, `write_adf`, `read_adf` |
| Solver import and translational periodicity | [fluent/workflow_2d.py](../src/turbo_moc/fluent/workflow_2d.py): `configure_periodicity` and the setup workflow |

## References and figure provenance

The two supplied manuscripts inform the physical motivation, terminology and exposition. The equations describing the channel, section scaling, mesh deformation and exporter are derived from the current implementation; they should not be attributed to the papers as if all these software features were described there.

1. Cioffi, A., Parisi, S., Agromayor, R., and Haglind, F. (2026). *Extension of the method of characteristics for the design of supersonic stators in two-phase turbines*. Journal of Turbomachinery, supplied accepted manuscript. In particular, Sections 2.1–2.4 distinguish nozzle design, blade parameterization, CFD discretization and thermodynamic modelling.
2. Cioffi, A., Parisi, S., Agromayor, R., and Haglind, F. (2026). *Non-ideal compressible flow dynamics of two-phase supersonic expansions and their implications for nozzle design*. 6th International Seminar on Non-Ideal Compressible Fluid Dynamics, supplied manuscript. The discussion emphasizes the physical assumptions behind two-phase MoC design and verification.
3. [Gmsh reference manual](https://gmsh.info/doc/texinfo/): mesh entities, fields, element types and periodic mesh operations.
4. [CGNS grid coordinates and elements](https://cgns.org/cgns-archives/CGNS_docs_current/sids/gridflow.html), [multizone connectivity](https://cgns.org/standard/SIDS/multizone.html) and [periodic API conventions](https://cgns.org/cgns-archives/CGNS_docs_current/midlevel/connectivity.html).

Figures 1–4 are generated from the shared implementation rather than copied from the manuscripts. Figure generation solves the MoC nozzle once and uses that freshly solved result for all illustrated cases. The inputs, result digest, mesh counts and validation results are stored in [figure_metadata.json](figures/figure_metadata.json); [the figure README](figures/readme.md) describes regeneration. PNG versions are embedded for convenient Markdown viewing, with SVG counterparts available for export. Numerical outputs used during figure generation stay separate from the app's job artifacts.

The coordinate inversion, in-surface metric, mapping-Jacobian sign and rotational correspondence were also checked numerically at representative points in axial, radial, tapered axial and curved channels. The recorded node and cell counts satisfy Equation (43). These checks establish consistency between the derivations and the implemented geometric operations; aerodynamic validation remains a separate step.
