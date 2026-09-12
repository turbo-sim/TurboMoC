# Method of characteristics

The MoC replaces the two-dimensional flow equations with compatibility relations
along two families of curves. Their intersections provide the local velocity
and direction, while the equation of state supplies the remaining properties.
This chapter states the planar formulation and the point constructions used by
the current implementation.

## Governing equations

For steady planar flow, continuity and irrotationality are

```{math}
:label: eq-planar-conservation
\frac{\partial(\rho u)}{\partial x}
+\frac{\partial(\rho v)}{\partial y}=0,\qquad
u_y-v_x=0.
```

Subscripts on velocity components denote partial derivatives. Uniform entropy
and total enthalpy imply $\mathrm dp=-\rho(u\,\mathrm du+v\,\mathrm dv)$.
Substituting this relation and $\mathrm d\rho=\mathrm dp/a^2$ into continuity,
then using $v_x=u_y$, gives

```{math}
:label: eq-velocity-pde
(u^2-a^2)u_x+2uv\,u_y+(v^2-a^2)v_y=0.
```

These are the equations underlying the planar construction in
{cite:t}`cioffiStators2026`. The local sound speed follows the
[thermodynamic model](thermodynamic_model.md); no perfect-gas equation of state
is required to obtain this form.

```{admonition} Review note — governing equation in the supplied paper
Equation (1) in the accepted stator manuscript prints $(u^2-a^2)$ as the
coefficient of $v_y$. Equation {eq}`eq-velocity-pde` uses $(v^2-a^2)$, obtained
directly from continuity above. Confirm the correction against the authors'
source. This chapter deliberately restricts the formulation to planar flow.
```

## Characteristic directions and compatibility

The discriminant of the velocity system is proportional to
$a^2(V^2-a^2)$. It is positive for $M>1$, giving two real characteristic
directions. With $u=V\cos\theta$ and $v=V\sin\theta$, they are

```{math}
:label: eq-characteristic-directions
C_\pm:\qquad \frac{\mathrm dy}{\mathrm dx}
=\lambda_\pm=\tan(\theta\pm\mu),\qquad
\mu=\sin^{-1}\!\left(\frac{1}{M}\right).
```

Along either direction, the corresponding compatibility equation is

```{math}
:label: eq-component-compatibility
(u^2-a^2)\,\mathrm du
+\left[2uv-(u^2-a^2)\lambda_\pm\right]\mathrm dv=0.
```

The slope specifies where a characteristic travels; compatibility specifies how
the state changes along it. At exactly sonic conditions the two characteristic
directions coalesce as lines and the usual marching construction becomes
singular. An [initial line](nozzle_design.md) is therefore needed on the
supersonic side of the throat.

## Generalized Prandtl–Meyer relation

Writing the velocity differentials in terms of $V$ and $\theta$ reduces
Eq. {eq}`eq-component-compatibility` to

```{math}
:label: eq-angle-compatibility
C_+:\quad \mathrm d\theta-\frac{\sqrt{M^2-1}}{V}\,\mathrm dV=0,
\qquad
C_-:\quad \mathrm d\theta+\frac{\sqrt{M^2-1}}{V}\,\mathrm dV=0.
```

Define the Prandtl–Meyer angle relative to a specified state on the isentrope:

```{math}
:label: eq-generalized-pm
\nu(V)=\int_{V_\mathrm{ref}}^V
\frac{\sqrt{[\widetilde V/a(\widetilde V,s_0)]^2-1}}{\widetilde V}
\,\mathrm d\widetilde V.
```

The integrand is real on the supersonic branch. For smooth sonic initialization,
$V_\mathrm{ref}=a_\mathrm{ref}$. For a flashing jump, the implementation places
the zero of $\nu$ at the downstream reference state. An additive shift in $\nu$
does not alter the compatibility equations, but it must be used consistently in
the fan-turning and wall constructions.

The invariants are consequently

```{math}
:label: eq-pm-invariants
I_+=\theta-\nu\quad\text{on }C_+,
\qquad I_-=\theta+\nu\quad\text{on }C_-.
```

The current `FluidManager` integrates Eq. {eq}`eq-generalized-pm` numerically
and builds interpolants for $\nu(V)$, $V(\nu)$, $a(V)$, and $\rho(V)$. The
velocity parameter is useful because $V$ increases along an expansion even when
$M$ does not. The invariants are exact for the stated continuous planar model;
their numerical evaluation still has quadrature and interpolation error.

## Interior-point construction

Let point $A$ supply an incoming $C_+$ characteristic and point $B$ an incoming
$C_-$. At their intersection $P$, the two invariants give

```{math}
:label: eq-interior-state
\theta_P=\frac{I_+(A)+I_-(B)}{2},\qquad
\nu_P=\frac{I_-(B)-I_+(A)}{2},\qquad V_P=V(\nu_P).
```

The resulting state determines $\mu_P$. The code approximates each curved
characteristic segment by a straight line whose angle is averaged between its
end states:

```{math}
:label: eq-averaged-slopes
\bar\lambda_+=\tan\!\left(\frac{\theta_A+\theta_P}{2}
+\frac{\mu_A+\mu_P}{2}\right),\qquad
\bar\lambda_-=\tan\!\left(\frac{\theta_B+\theta_P}{2}
-\frac{\mu_B+\mu_P}{2}\right).
```

Intersecting the two lines then gives

```{math}
:label: eq-interior-position
x_P=\frac{y_B-y_A+\bar\lambda_+x_A-\bar\lambda_-x_B}
{\bar\lambda_+-\bar\lambda_-},\qquad
y_P=y_A+\bar\lambda_+(x_P-x_A).
```

This separates the thermodynamic compatibility calculation from the geometric
approximation. Nearly parallel characteristic segments make the intersection
sensitive to small slope errors, so resolving the geometry remains important
even when the invariants are evaluated accurately.

```{admonition} Figure placeholder — one characteristic cell
Draw $A$ below $B$ and the downstream intersection $P$. Label $AP$ as $C_+$,
$BP$ as $C_-$, the positive $x,y$ axes, and the flow and Mach angles at each
foot point. Show the straight segments as approximations to curved
characteristics. This figure should use the signs of
Eq. {eq}`eq-characteristic-directions`.
```

## Symmetry-axis and wall points

At the symmetry axis, $y=0$ and $\theta=0$. An incoming $C_-$ from $B$ therefore
sets $\nu_P=\theta_B+\nu_B$. The position follows by intersecting its averaged
direction with $y=0$. Reflection supplies the outgoing family for the next
part of the characteristic net.

At a prescribed upper wall, the velocity must be tangent to the wall. If the
wall angle is $\theta_w$ and an incoming $C_+$ carries $I_+$, then
$\nu_w=\theta_w-I_+$. The wall point is obtained by satisfying this state
relation together with the wall geometry and characteristic intersection. At a
sharp corner, several expansion-fan states occupy the same geometric point;
their flow angles differ, as explained in [nozzle design](nozzle_design.md).

The inverse problem instead finds a wall that supports a prescribed downstream
flow. The implementation uses mass conservation to recover the turning contour.
This construction, rather than repeated solution of a full field on a guessed
wall, makes the MoC useful as a direct geometry-design method.
