# What the sparse deformation code does, and why it uses LU

For the complete coordinate derivation and illustrated meshing/export procedure, see the [geometry and meshing teaching guide](../theory_guide.md). This note focuses on the deformation implementation and its computational cost.

The phrase “Deformation already uses cKDTree, sparse matrices, and reusable sparse LU. Volume assembly and exterior-face auditing already use array operations” describes the mesh sweep in [the shared meshing module](../src/meshing/meshing_3d.py), particularly `CoreDeformation`, `assemble_volume` and the CGNS boundary audit. The `reference_mesh`, `volume_mesh` and `export_mesh` stages in the same module reuse these operations.

It means these parts already call compiled numerical libraries. The expensive V1 repetition was largely geometry construction and passage centering. LU is used for mesh-node placement; it does not solve the MoC flow equations.

## Linear and annular rows

The app now retains independent row settings and jobs under an ordered
project schema. The deformation described below concerns the annular
section scale and fixed machine pitch. A linear cascade shares the same
reference mesher, but its coordinates are physical lengths, its pitch scales
with the requested chord, and every span section has $a(\eta)=1$. Its
mapping is

$$
(X,Y,Z)=(-q_y,q_x,\eta H).
$$

Identical planar sections give zero graph-harmonic correction; the volume is
a physical extrusion. Linear periodic partners differ by $(0,p,0)$ rather
than a rotation. True 2D linear export remains available independently of
this extrusion. One span cell is optional, and its endwall boundary conditions
still determine the physical model. See the
[linear-cascade derivation](../theory_guide.md#54-linear-cascade-and-physical-extrusion).

## Coordinates and the reference section

Let the hub and shroud curves be $\mathbf h(u)$ and $\mathbf s(u)$, with values $(r,z)$. `MeridionalChannel` in [meridional_channel.py](../src/geometry/meridional_channel.py) implements the general mapping

$$
\mathbf P(u,\eta)=(1-\eta)\mathbf h(u)+\eta\mathbf s(u),\qquad
dm=\|\mathbf P_u\|\,du,\qquad
m'(u,\eta)=\int_0^u\frac{\|\mathbf P_u(v,\eta)\|}{r(v,\eta)}\,dv.
$$

Here $\eta=0$ is the hub, $\eta=1$ the shroud, and the geometric pseudo-streamlines have constant $\eta$. They do not assert that a solved 3D flow follows those curves.

The Gmsh reference plane uses lengths in mm:

$$
q_x=R_{\rm ref}\theta,\qquad q_y=-R_{\rm ref}m',\qquad
\phi=-\theta,\qquad
(X,Y,Z)=(r\cos\phi,r\sin\phi,z).
$$

Thus $\theta$ increases clockwise about global $Z$; the app plots $(\theta,-m')$. The span scale is

$$
a(\eta)=\frac{m'(1,\eta)}{m'(1,1/2)}.
$$

“One conformal template” means the reference blade contour is multiplied by $a(\eta)$. It does not promise identical physical sections. Constant axial chord can still require a changing conformal scale because radius changes.

## Find and protect near-wall nodes with cKDTree

The reference 2D mesh has blade-boundary nodes $\mathbf q_b$. A tree query computes, for every node $i$,

$$
d_i=\min_{b\in B}\|\mathbf q_i-\mathbf q_b\|.
$$

The shared code protects nodes for which $d_i\le 1.5t_{\rm stack}+h_{\rm blade}$. These nodes move with the scaled blade, together with the inflation collar. Distances are to sampled wall **nodes**, rather than exact distances to continuous CAD curves. The cutoff conservatively protects extra nearby rows.

A spatial tree avoids building a large all-nodes-by-all-blade-nodes distance array. This use of cKDTree is separate from the V1 automatic-clearance search, which still performs dense distance calculations.

## Smooth the remaining mesh through a sparse linear system

Construct a graph from the unique reference-mesh cell edges, and set

$$
w_{ij}=\frac{1}{\|\mathbf q_i-\mathbf q_j\|^2},\qquad
L_{ii}=\sum_j w_{ij},\qquad L_{ij}=-w_{ij}\quad(i\ne j).
$$

Most node pairs are not neighbors, so most entries are zero. The code stores $L$ as a sparse matrix; it does not allocate a dense $N\times N$ array.

At each span station, first scale all reference nodes:

$$
\widetilde{\mathbf q}_i=a(\eta)\mathbf q_i.
$$

The blade/collar keeps this position. Outer-boundary positions are prescribed from the target passage while preserving streamwise station and inlet/outlet pitch fraction. Their correction is

$$
\mathbf d_C=\mathbf q_C^{\rm target}-\widetilde{\mathbf q}_C.
$$

Free core nodes minimize the graph displacement energy

$$
E(\mathbf d)=\frac12\sum_{(i,j)\in\mathcal E}
w_{ij}\|\mathbf d_i-\mathbf d_j\|^2.
$$

Partition nodes into free nodes $F$ and constrained nodes $C$. Stationarity gives

$$
L_{FF}\mathbf d_F=-L_{FC}\mathbf d_C,\qquad
\mathbf q_F^{\rm new}=\widetilde{\mathbf q}_F+\mathbf d_F.
$$

Both coordinate columns of $\mathbf d_F$ are solved together. The graph and its weights are defined in the reference mesh and retained across span. They are not rebuilt from each warped section.

## Why LU, and why reuse it?

We need a linear solver because every free node's displacement depends on its neighbors. Assigning boundary nodes alone would leave discontinuities or poor cells in the core.

LU factorization converts the system into triangular solves. With sparse permutations, schematically

$$
P L_{FF} Q=L_{\rm tri}U_{\rm tri}.
$$

The lower/upper triangular factors here should not be confused with the graph Laplacian $L$. For each new boundary displacement, apply permutations, forward substitution, and backward substitution. There is no explicit matrix inverse.

`scipy.sparse.linalg.splu` uses SuperLU and accepts CSC matrices efficiently; the returned object exposes a reusable solve operation. [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html)

V2 creates this factorization once per sweep if any section needs deformation, then reuses it for every span station and both coordinate columns. It skips deformation entirely when $a=1$. LU is the current implementation choice, not a physical requirement. A properly constrained graph Laplacian is symmetric positive definite, so sparse Cholesky or a preconditioned iterative solver could also work. Compare setup cost, memory, residuals, and final mesh quality before changing solvers.

Harmonic smoothing does **not** guarantee an uninverted mesh. The shared implementation checks section convexity and periodic-node correspondence, then checks positive volume Jacobians and SICN after mapping.

## What “array operations” means for volume assembly and auditing

A reference quadrilateral $(i_0,i_1,i_2,i_3)$ becomes a hexahedron between stations $j$ and $j+1$:

$$
(i_0+jN,\ldots,i_3+jN,\ i_0+(j+1)N,\ldots,i_3+(j+1)N).
$$

A triangle becomes a prism in the same way. NumPy broadcasting creates connectivity blocks for many cells/stations at once; Gmsh receives those blocks through its bulk element API. This avoids a Python call per volume cell.

For the exterior-face audit, cell face-node patterns are gathered into arrays. Sorting each face's node IDs provides an orientation-independent key. A key appearing once identifies an exterior face; twice identifies an interior face. The audit checks that exterior faces are partitioned into the seven expected patches and oriented outward. This validates export topology; it is separate from geometric quality.

## V2's one-time passage placement

V1 fits and optimizes the passage repeatedly while constructing span sections. V2 fits once and performs auto-centering once at $\eta=1/2$. Write the reference centerline as $\mathbf c_{\rm ref}=\mathbf c_0+\boldsymbol\Delta$, where $\boldsymbol\Delta=(\Delta_x,0)$ is that reference translation. V2 uses

$$
\mathbf c_\eta=a(\eta)(\mathbf c_{\rm ref}-\boldsymbol\Delta)+\boldsymbol\Delta,
\qquad
\mathbf c_\eta^\pm=\mathbf c_\eta
 \pm\left(\frac{\pi R_{\rm ref}}{N_{\rm blades}},0\right).
$$

It scales the underlying shape/chord, retains the reference translation in q-space, and preserves the fixed pitch $2\pi R_{\rm ref}/N_{\rm blades}$. The translation is not reoptimized and does not scale with span.

This deliberately changes V1's per-span placement policy. V2 checks blade containment at endpoint and sweep/preview stations and rejects inadmissible sections; it does not silently fall back to per-span auto-centering. The construction uses general hub/shroud curves for axial, radial, and curved channels. Continuous-span collision proofs and automated repair remain future work.

## Displayed quality definitions

These diagnostics use actual first-order hex/prism coordinates/connectivity. They are explicitly defined rather than presented as identical to every CFD solver's metrics.

For each triangular/quadrilateral face, with ideal angle $\alpha_0=60^\circ$ or $90^\circ$,

$$
s_f=\max\left(
\frac{\alpha_{\max}-\alpha_0}{180^\circ-\alpha_0},
\frac{\alpha_0-\alpha_{\min}}{\alpha_0}
\right),\qquad s_{\rm cell}=\max_f s_f.
$$

Zero is ideal; one is degenerate. Orthogonality is

$$
o_{\rm cell}=\min_f\frac{|\mathbf n_f\cdot\mathbf d_f|}
{\|\mathbf n_f\|\,\|\mathbf d_f\|}.
$$

For an internal face, $\mathbf d_f$ joins adjacent cell centroids. For an exterior face, it joins the cell and face centroids. Periodic neighbors are rotated into the same frame before forming this connector. Centroids are arithmetic means of vertices, and the face normal is the polygon's area-vector sum. One is ideal. These formulas do not constitute a complete finite-volume nonorthogonality analysis.

The displayed aspect ratio is the **edge** ratio

$$
AR_{\rm cell}=\frac{\max_{e\in{\rm cell}}\ell_e}{\min_{e\in{\rm cell}}\ell_e}.
$$

Large ratios can be intentional in inflation layers; acceptance criteria should account for cell location and the target solver. These three histograms complement the mandatory positive-Jacobian/SICN checks.

## Physical size estimates

Locally on a fixed stream sheet, the conformal mapping has length scale $r/R_{\rm ref}$. A reference-plane first-height target $h_0$, scaled with the blade, gives

$$
h_{\rm mapped}(u,\eta)\approx
h_0\,a(\eta)\frac{r(u,\eta)}{R_{\rm ref}}.
$$

The app samples this estimate at hub/midspan/shroud and LE/mid-chord/TE. It reports the sampled range, not a bound over all cells. This is an in-surface height estimate, not the 3D shortest distance normal to a tapered blade. Measuring actual wall-normal distances and estimating y+ require further work and flow assumptions.

