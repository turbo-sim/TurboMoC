"""Structured multi-block (O-H) 2D mesh of a stator blade passage.

TurboGrid-like topology on the same fluid domain as the unstructured mesh
(stator_mesh_2d.stator_flow_domain): an O-grid around the blade carries the
boundary layer -- exactly `num_layers` cells at `growth_ratio`, first cell
`first_height` -- and five H-blocks fill the passage:

            inlet
     +------------------+
     |     upstream     |
     +---------O--------+          O = LE' / TE': the O-grid's offset curve
     | channel |channel |              at the blade's axial max / min, where
     |  (one   O  (other|              pitchwise connectors start
     |  side)  |   side)|
     +---------O--------+
     |    downstream    |
     +------------------+
            outlet

Every block is a transfinite (TFI) quad patch, so the mesh is fully
structured. Curves are split at block corners by exact knot insertion, so
the geometry is the passage's own B-splines (no resampling). The pitchwise
connectors are horizontal (constant y): from the O-grid's axial extremes
they cannot cross the O-grid, and they meet each periodic side at the
pitch-translated point of the other, keeping left/right periodic pieces exact
translations.

Raises on anything it cannot build cleanly (offset self-intersection,
inverted cells, ...); stator_mesh_2d.mesh_sized_stator then falls back to
the unstructured mesh.
"""

import logging

import gmsh
import numpy as np
from scipy.interpolate import BSpline

from turbo_moc.meshing import stator_mesh_2d as base
from turbo_moc.meshing.wall_spacing import inflation_stack

logger = logging.getLogger(__name__)

SAMPLES = 400
MAX_CORE_GROWTH = 1.15
# Smallest accepted angle between a TE/LE connector and the wake/stagnation
# line or the O-grid outline. High outlet metal angles (~70 deg) squeeze the
# TE connectors to ~6-9 deg; cells there stay valid (block min scaled
# Jacobian checked after meshing), so only reject clearly degenerate corners.
MIN_CONNECTOR_ANGLE = 4.0


# =============================================================================
# 1. Python curve pieces: exact B-splines, lines and circular arcs
# =============================================================================


class Bspl:
    def __init__(self, spline):
        self.s = spline
        self.u0, self.u1 = spline.t[spline.k], spline.t[-spline.k - 1]

    def eval(self, u):
        return self.s(np.atleast_1d(u))

    def samples(self, n=SAMPLES):
        return self.s(np.linspace(self.u0, self.u1, n))

    def split(self, u):
        k, t = self.s.k, self.s.t
        # Snap only true numerical coincidences with an existing knot; a
        # looser match (np.isclose defaults) counted a nearby-but-different
        # knot as this one and produced a decreasing knot vector.
        near = np.abs(t - u) <= 1e-12 * max(1.0, abs(self.u1 - self.u0))
        if np.any(near):
            u = float(t[near][0])
        multiplicity = int(np.sum(t == u))
        s = self.s.insert_knot(u, k - multiplicity) if multiplicity < k else self.s
        i = int(np.searchsorted(s.t, u, side="right"))
        left = BSpline(np.r_[s.t[:i], u], s.c[: i - k], k)
        right = BSpline(np.r_[u, s.t[i - k:]], s.c[i - k - 1:], k)
        return Bspl(left), Bspl(right)

    def reversed(self):
        t = self.u0 + self.u1 - self.s.t[::-1]
        return Bspl(BSpline(t, self.s.c[::-1], self.s.k))


class Line:
    def __init__(self, a, b):
        self.a, self.b = np.asarray(a, float), np.asarray(b, float)
        self.u0, self.u1 = 0.0, 1.0

    def eval(self, u):
        u = np.atleast_1d(u)[:, None]
        return self.a + u * (self.b - self.a)

    def samples(self, n=SAMPLES):
        return self.eval(np.linspace(0, 1, n))

    def split(self, u):
        m = self.eval(u)[0]
        return Line(self.a, m), Line(m, self.b)

    def reversed(self):
        return Line(self.b, self.a)


class Arc:
    """Circular arc from angle a0 to a1 (radians, signed sweep)."""

    def __init__(self, centre, radius, a0, a1):
        self.c, self.r, self.u0, self.u1 = np.asarray(centre, float), float(radius), a0, a1

    def eval(self, u):
        u = np.atleast_1d(u)
        return self.c + self.r * np.column_stack((np.cos(u), np.sin(u)))

    def samples(self, n=SAMPLES // 4):
        return self.eval(np.linspace(self.u0, self.u1, n))

    def split(self, u):
        return Arc(self.c, self.r, self.u0, u), Arc(self.c, self.r, u, self.u1)

    def reversed(self):
        return Arc(self.c, self.r, self.u1, self.u0)


def _length(piece):
    p = piece.samples()
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def _start(piece):
    return piece.eval(piece.u0)[0] if not isinstance(piece, Bspl) else piece.s.c[0]


def _end(piece):
    return piece.eval(piece.u1)[0] if not isinstance(piece, Bspl) else piece.s.c[-1]


def _param_at_y(piece, y):
    """Parameter where a monotone-in-y piece reaches `y` (bisection)."""
    lo, hi = piece.u0, piece.u1
    f = lambda u: piece.eval(u)[0][1] - y  # noqa: E731
    flo = f(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


# =============================================================================
# 2. Blade loop: pieces in order, split at the axial extrema (LE / TE corners)
# =============================================================================


def _blade_pieces(blade):
    """Closed blade loop as [spline, arc_upper, arc_lower] in loop order, the
    same CAD as stator_mesh_2d.add_blade_boundary (fitted B-spline from the
    first to the last control point, then the circular trailing edge)."""
    cps = np.asarray(blade["control_points"], float)
    spline = Bspl(BSpline(np.asarray(blade["knots"], float), cps, 3))
    te = np.column_stack((blade["trailing_edge"]["x"], blade["trailing_edge"]["y"]))
    centre = 0.5 * (cps[0] + cps[-1])
    mid = te[len(te) // 2]
    radius = float(np.linalg.norm(cps[-1] - centre))
    angle = lambda p: np.arctan2(p[1] - centre[1], p[0] - centre[0])  # noqa: E731
    a_end, a_mid, a_start = angle(cps[-1]), angle(mid), angle(cps[0])
    # Sweep through the TE midpoint (the short way round that contains it).
    sweep1 = (a_mid - a_end + np.pi) % (2 * np.pi) - np.pi
    sweep2 = (a_start - a_mid + np.pi) % (2 * np.pi) - np.pi
    return [spline, Arc(centre, radius, a_end, a_end + sweep1),
            Arc(centre, radius, a_mid, a_mid + sweep2)]


def _support(piece, direction, u_guess, span):
    """Refine the parameter maximising p . direction near u_guess (golden section)."""
    lo, hi = max(piece.u0, u_guess - span), min(piece.u1, u_guess + span)
    if lo > hi:
        lo, hi = hi, lo
    g = (np.sqrt(5) - 1) / 2
    f = lambda u: float(piece.eval(u)[0] @ direction)  # noqa: E731
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    for _ in range(100):
        if f(a) > f(b):
            hi, b = b, a
            a = hi - g * (hi - lo)
        else:
            lo, a = a, b
            b = lo + g * (hi - lo)
    return 0.5 * (lo + hi)


def _split_loop_at_extrema(pieces, upstream, downstream):
    """Split the loop at its support points in the `upstream` direction (LE
    corner: outward normal facing the inflow) and the `downstream` direction
    (TE corner: outward normal along the outflow); return the two sides
    LE->TE and TE->LE as piece lists in loop order."""
    best = {}
    for i, piece in enumerate(pieces):
        u = np.linspace(piece.u0, piece.u1, 4001)
        pts = piece.eval(u)
        for key, direction in (("le", upstream), ("te", downstream)):
            j = int(np.argmax(pts @ direction))
            value = float(pts[j] @ direction)
            if key not in best or value > best[key][0]:
                refined = _support(piece, direction, u[j], 2 * (u[1] - u[0]))
                best[key] = (value, i, refined)
    # Split the later piece first so indices of earlier ones stay valid.
    cuts = sorted(((best[k][1], best[k][2], k) for k in ("le", "te")), key=lambda c: (c[0], c[1]))
    loop = list(pieces)
    marks = {}
    for i, u, key in reversed(cuts):
        piece = loop[i]
        position = (u - piece.u0) / (piece.u1 - piece.u0)
        if position < 1e-6:  # corner already is a piece junction: no split
            marks[key] = i
            continue
        if position > 1 - 1e-6:
            marks[key] = i + 1
            continue
        first, second = piece.split(u)
        loop[i:i + 1] = [first, second]
        for other in marks:
            if marks[other] > i:
                marks[other] += 1
        marks[key] = i + 1  # the corner is the start of `second`
    n = len(loop)
    ile, ite = marks["le"] % n, marks["te"] % n
    walk = lambda a, b: [loop[(a + k) % n] for k in range((b - a) % n)]  # noqa: E731
    return walk(ile, ite), walk(ite, ile)


CHUNK_CELLS = 8
SPACING_GROWTH = 0.1  # blade spacing grows ~10% per cell away from LE / TE


def _sub_piece(piece, ua, ub):
    """Exact sub-curve of `piece` from parameter ua to ub (piece direction)."""
    if isinstance(piece, Arc):
        return Arc(piece.c, piece.r, ua, ub)
    if isinstance(piece, Line):
        return Line(piece.eval(ua)[0], piece.eval(ub)[0])
    if not np.isclose(ub, piece.u1):
        piece = piece.split(ub)[0]
    if not np.isclose(ua, piece.u0):
        piece = piece.split(ua)[1]
    return piece


def _closest_param(pieces, point):
    """(piece index, parameter) of the point on `pieces` closest to `point`."""
    best = None
    for k, piece in enumerate(pieces):
        u = np.linspace(piece.u0, piece.u1, 2001)
        d = np.linalg.norm(piece.eval(u) - point, axis=1)
        j = int(np.argmin(d))
        if best is None or d[j] < best[0]:
            lo, hi = max(piece.u0, u[j] - 2 * (u[1] - u[0])), min(piece.u1, u[j] + 2 * (u[1] - u[0]))
            g = (np.sqrt(5) - 1) / 2
            f = lambda v: float(np.linalg.norm(piece.eval(v)[0] - point))  # noqa: E731
            a, b = hi - g * (hi - lo), lo + g * (hi - lo)
            for _ in range(80):
                if f(a) < f(b):
                    hi, b = b, a
                    a = hi - g * (hi - lo)
                else:
                    lo, a = a, b
                    b = lo + g * (hi - lo)
            best = (f(0.5 * (lo + hi)), k, 0.5 * (lo + hi))
    return best[1], best[2]


def _matched_periodic_split(middle, chunk_points, chunk_counts, chunk_ratios):
    """Give a channel's periodic side the same RELATIVE node distribution as
    the facing O-grid side: split it at the same arc-length fractions as the
    O-grid chunk boundaries, with the same node counts and progressions.
    Transfinite interpolation joins node i to node i across the channel, so
    matching distributions keep those lines from crossing; a uniform
    periodic side against the clustered O-grid nodes near the LE gave
    strongly slanted lines. (Closest-point and same-level matching were both
    tried and were worse: not monotone around the LE / TE noses.) Returns
    (split pieces, nodes per piece, progression per piece, (piece index,
    parameter) splits) -- the splits are reused unchanged on the other,
    pitch-translated periodic side."""
    chunk_lengths = [float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) for p in chunk_points]
    fractions = np.cumsum(chunk_lengths)[:-1] / np.sum(chunk_lengths)
    piece_lengths = [_length(p) for p in middle]
    starts = np.r_[0.0, np.cumsum(piece_lengths)]
    splits, cells, ratios = [], [], []
    pending, merged = 0, False
    for fraction, count, ratio in zip(fractions, chunk_counts[:-1], chunk_ratios[:-1]):
        pending += count - 1
        target = fraction * starts[-1]
        k = min(int(np.searchsorted(starts, target, side="right")) - 1, len(middle) - 1)
        within = (target - starts[k]) / piece_lengths[k]
        if within < 1e-6 or within > 1 - 1e-6:  # on a piece junction: merge into the next chunk
            merged = True
            continue
        u = np.linspace(middle[k].u0, middle[k].u1, 2001)
        pts = middle[k].eval(u)
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        splits.append((k, float(np.interp(within * arc[-1], arc, u))))
        cells.append(pending)
        ratios.append(1.0 if merged else ratio)
        pending, merged = 0, False
    cells.append(pending + chunk_counts[-1] - 1)
    ratios.append(1.0 if merged else chunk_ratios[-1])
    pieces, interval_of = _apply_splits(middle, splits)
    counts, progressions = [], []
    for j in range(len(cells)):
        members = [i for i, owner in enumerate(interval_of) if owner == j]
        if len(members) == 1:
            counts.append(cells[j] + 1)
            progressions.append(ratios[j])
        else:  # an original piece junction inside the interval
            per = _distribute([_length(pieces[i]) for i in members], cells[j] + 1)
            counts.extend(per)
            progressions.extend([1.0] * len(per))
    return pieces, counts, progressions, splits


def _apply_splits(pieces, splits):
    """Split `pieces` at (piece index, parameter) points given in order;
    return the new pieces and, for each, the index of the interval (between
    consecutive splits) it lies in."""
    by_piece = {}
    for k, u in splits:
        by_piece.setdefault(k, []).append(u)
    result, interval_of, interval = [], [], 0
    for k, piece in enumerate(pieces):
        params = sorted(by_piece.get(k, []))
        edges = [piece.u0] + params + [piece.u1]
        for j, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            result.append(_sub_piece(piece, a, b))
            interval_of.append(interval)
            if j < len(params):
                interval += 1
    return result, interval_of


def _side_table(side):
    """Dense samples of a side (pieces in order): points, arc length, and the
    piece index / parameter of every sample (junctions belong to the earlier
    piece; the later piece's duplicate first sample is dropped)."""
    pts, piece_of, params = [], [], []
    for k, piece in enumerate(side):
        u = np.linspace(piece.u0, piece.u1, SAMPLES)
        p = piece.eval(u)
        if k:
            p, u = p[1:], u[1:]
        pts.append(p)
        piece_of.append(np.full(len(u), k))
        params.append(u)
    pts = np.vstack(pts)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    return pts, arc, np.concatenate(piece_of), np.concatenate(params)


def _offset_polyline(pts, arc, distance, outward_sign, start, end, window):
    """Offset samples by `distance` along the outward normal, ends pinned to
    `start` / `end` (the exact offsets of the LE / TE corners, shared by both
    sides). Tangents are averaged over an arc-length `window`: the fitted
    blade spline meets the circular TE with a small tangent kink (a few
    degrees), and a raw offset folds at a concave kink. The window must stay
    below the TE radius, or it smears the offset around a small TE."""
    tangent = np.gradient(pts, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    lo = np.searchsorted(arc, arc - window / 2, side="left")
    hi = np.searchsorted(arc, arc + window / 2, side="right")
    summed = np.vstack([np.zeros(2), np.cumsum(tangent, axis=0)])
    smooth = summed[hi] - summed[lo]
    smooth /= np.linalg.norm(smooth, axis=1)[:, None]
    shifted = pts + distance * outward_sign * np.column_stack((smooth[:, 1], -smooth[:, 0]))
    shifted[0], shifted[-1] = start, end
    segments = np.diff(shifted, axis=0)
    if np.any(np.einsum("ij,ij->i", segments[:-1], segments[1:]) < 0):
        raise ValueError("O-grid offset curve folds (concave curvature < inflation thickness).")
    return shifted


def _node_positions(length, h_start, h_end, h_max, n_cells=None):
    """Arc positions of the blade-side nodes: spacing h_start / h_end at the
    two corners, growing by SPACING_GROWTH per cell up to h_max."""
    s = np.linspace(0.0, length, 4001)
    h = np.minimum.reduce([np.full_like(s, h_max), h_start + SPACING_GROWTH * s,
                           h_end + SPACING_GROWTH * (length - s)])
    cells_so_far = np.r_[0.0, np.cumsum(0.5 * (1 / h[1:] + 1 / h[:-1]) * np.diff(s))]
    if n_cells is None:
        n_cells = max(2, int(np.ceil(cells_so_far[-1])))
    return np.interp(np.linspace(0.0, cells_so_far[-1], n_cells + 1), cells_so_far, s), n_cells


def _o_grid_side(side, distance, outward_sign, start, end, h_start, h_end, h_max, n_cells=None,
                 window=None):
    """Blade side and its O-grid offset, cut into matching chunks of about
    CHUNK_CELLS cells. Gmsh spaces transfinite nodes by arc length, and the
    blade and its offset differ in length by curvature, so a single curve
    pair would drift out of alignment along the chord (slanted wall cells).
    Chunk ends are exact normal pairs and each chunk pair gets the same node
    count and progression, so wall-normal lines stay nearly orthogonal.

    Returns (blade_chunks, offset_chunk_points, nodes_per_chunk,
    progression_per_chunk, n_cells)."""
    pts, arc, piece_of, params = _side_table(side)
    shifted = _offset_polyline(pts, arc, distance, outward_sign, start, end,
                               distance if window is None else window)
    nodes, n_cells = _node_positions(arc[-1], h_start, h_end, h_max, n_cells)
    # Piece junctions must be chunk ends (each chunk = one exact CAD curve).
    junction_s = [arc[np.flatnonzero(piece_of == k)[-1]] for k in range(len(side) - 1)]
    fixed = {0, n_cells}
    for s_j in junction_s:
        i = int(np.argmin(np.abs(nodes[1:-1] - s_j))) + 1
        if i in fixed:
            raise ValueError("Blade side too coarse to resolve a curve junction.")
        nodes[i] = s_j
        fixed.add(i)
    if np.any(np.diff(nodes) <= 0):
        raise ValueError("Non-monotone blade node distribution.")
    ends = []
    marks = sorted(fixed)
    for a, b in zip(marks[:-1], marks[1:]):
        pieces = max(1, int(round((b - a) / CHUNK_CELLS)))
        ends.extend(np.linspace(a, b, pieces + 1).round().astype(int)[:-1].tolist())
    ends = sorted(set(ends + [n_cells]))

    blade_chunks, offset_chunks, counts, ratios = [], [], [], []
    for i0, i1 in zip(ends[:-1], ends[1:]):
        s0, s1 = nodes[i0], nodes[i1]
        k = int(piece_of[min(np.searchsorted(arc, 0.5 * (s0 + s1)), len(arc) - 1)])
        mask = piece_of == k
        if k:  # include the junction sample that starts this piece
            mask[np.flatnonzero(mask)[0] - 1] = True
        piece_arc, piece_u = arc[mask], params[mask]
        if k:
            piece_u[0] = side[k].u0
        u0, u1 = np.interp([s0, s1], piece_arc, piece_u)
        u0 = side[k].u0 if np.isclose(s0, piece_arc[0]) else u0
        u1 = side[k].u1 if np.isclose(s1, piece_arc[-1]) else u1
        blade_chunks.append(_sub_piece(side[k], u0, u1))
        inside = (arc > s0) & (arc < s1)
        ends_xy = np.column_stack([np.interp([s0, s1], arc, shifted[:, j]) for j in (0, 1)])
        offset_chunks.append(np.vstack([ends_xy[0], shifted[inside], ends_xy[1]]))
        cells = i1 - i0
        counts.append(cells + 1)
        first, last = nodes[i0 + 1] - nodes[i0], nodes[i1] - nodes[i1 - 1]
        ratios.append((last / first) ** (1.0 / (cells - 1)) if cells > 1 else 1.0)
    return blade_chunks, offset_chunks, counts, ratios, n_cells


# =============================================================================
# 3. Node distributions
# =============================================================================


def _geometric(length, first, n_min=2, max_last=None):
    """(n_nodes, progression) for a curve of `length` whose first cell is
    `first`, growing at most MAX_CORE_GROWTH, last cell <= max_last."""
    for cells in range(max(n_min - 1, 1), 2000):
        if first * cells >= length:
            return cells + 1, 1.0
        lo, hi = 1.0 + 1e-12, 10.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            total = first * (mid**cells - 1) / (mid - 1)
            lo, hi = (mid, hi) if total < length else (lo, mid)
        ratio = 0.5 * (lo + hi)
        if ratio <= MAX_CORE_GROWTH and (max_last is None or first * ratio ** (cells - 1) <= max_last):
            return cells + 1, ratio
    raise ValueError("Could not grade a connector line.")


def _distribute(lengths, total_nodes):
    """Split `total_nodes` (shared ends counted once) over pieces by length."""
    cells = total_nodes - 1
    raw = np.asarray(lengths) / np.sum(lengths) * cells
    counts = np.maximum(1, np.floor(raw).astype(int))
    while counts.sum() < cells:
        counts[np.argmax(raw - counts)] += 1
    while counts.sum() > cells:
        counts[np.argmax(counts)] -= 1
    return (counts + 1).tolist()


# =============================================================================
# 4. Gmsh model
# =============================================================================


class _Model:
    def __init__(self):
        self.points = []  # (xy, tag)

    def point(self, xy):
        xy = np.asarray(xy, float)
        for known, tag in self.points:
            if np.linalg.norm(known - xy) < 1e-9:
                return tag
        tag = gmsh.model.occ.addPoint(xy[0], xy[1], 0.0)
        self.points.append((xy, tag))
        return tag

    def curve(self, piece):
        start, end = self.point(_start(piece)), self.point(_end(piece))
        if isinstance(piece, Line):
            return gmsh.model.occ.addLine(start, end), start, end
        if isinstance(piece, Arc):
            mid = 0.5 * (piece.u0 + piece.u1)
            centre = gmsh.model.occ.addPoint(*piece.c, 0.0)
            if abs(piece.u1 - piece.u0) < np.radians(179.0):
                return gmsh.model.occ.addCircleArc(start, centre, end), start, end
            # two half arcs keep each arc < 180 deg, as addCircleArc requires
            m = self.point(piece.eval(mid)[0])
            a = gmsh.model.occ.addCircleArc(start, centre, m)
            b = gmsh.model.occ.addCircleArc(m, centre, end)
            return (a, b), start, end
        s = piece.s
        tags = [start] + [gmsh.model.occ.addPoint(x, y, 0.0) for x, y in s.c[1:-1]] + [end]
        knots, mults = np.unique(s.t, return_counts=True)
        return gmsh.model.occ.addBSpline(tags, degree=s.k, knots=knots.tolist(),
                                         multiplicities=mults.tolist()), start, end

    def spline_through(self, pts):
        start, end = self.point(pts[0]), self.point(pts[-1])
        inner = [gmsh.model.occ.addPoint(x, y, 0.0) for x, y in pts[1:-1]]
        return gmsh.model.occ.addSpline([start] + inner + [end]), start, end


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _angle(a, b):
    return float(np.degrees(np.arccos(np.clip(_unit(a) @ _unit(b), -1.0, 1.0))))


def mesh_structured_passage(blade, curves, sizes):
    """Structured O-H mesh of one periodic stator passage (hub section, mm).

    `curves` is stator_mesh_2d.stator_flow_domain(blade, ...); `sizes` as in
    mesh_sized_stator (blade_size, domain_size, first_height, growth_ratio,
    num_layers, transition_ratio). Returns the same dict layout as
    mesh_sized_stator, with topology "structured".

    Blocks: the O-grid (two halves), upstream left/right (split by the
    stagnation line: the passage centreline shifted through the O-grid's LE
    corner), the two blade-to-periodic channels, and downstream left/right
    (split by the wake line: the centreline shifted through the TE corner).
    Diagonal connectors leave the LE / TE corners at ~45 deg to the periodic
    sides, at one level per end so left and right pieces stay translations.
    """
    from turbo_moc.meshing.stator_passage import metal_direction

    h1, ratio, layers = float(sizes["first_height"]), float(sizes["growth_ratio"]), int(sizes["num_layers"])
    thickness, last_height = inflation_stack(h1, ratio, layers)
    blade_size, domain_size = float(sizes["blade_size"]), float(sizes["domain_size"])
    first_core = float(sizes.get("transition_ratio", base.CORE_AREA_GROWTH)) * last_height
    pitch = float(blade["pitch"])
    upstream = -metal_direction(blade["metal_angle_in"])
    downstream = metal_direction(blade["metal_angle_out"])

    # ---- blade sides, O-grid corners and offsets ---------------------------------
    side_a, side_b = _split_loop_at_extrema(_blade_pieces(blade), upstream, downstream)
    loop_pts = np.vstack([p.samples() for p in side_a + side_b])
    area = 0.5 * (np.dot(loop_pts[:, 0], np.roll(loop_pts[:, 1], -1))
                  - np.dot(loop_pts[:, 1], np.roll(loop_pts[:, 0], -1)))
    outward = 1.0 if area > 0 else -1.0  # CCW loop: outward normal = tangent rotated clockwise
    le, te = _start(side_a[0]), _start(side_b[0])
    # At a support point the outward normal IS the support direction.
    le_o, te_o = le + thickness * upstream, te + thickness * downstream
    # Corner spacings: ~40 cells per full circle of the local curvature.
    radius_te = min(p.r for p in side_a + side_b if isinstance(p, Arc))
    pts, arc, _, _ = _side_table(side_a)
    window = arc < 0.01 * arc[-1]
    heading = np.unwrap(np.arctan2(*np.gradient(pts[window], axis=0).T[::-1]))
    radius_le = abs(arc[window][-1] / (heading[-1] - heading[0] + 1e-12))
    h_le = min(blade_size, 2 * np.pi * radius_le / 40)
    h_te = min(blade_size, 2 * np.pi * radius_te / 40)
    window = min(thickness, radius_te)
    n_cells = max(_o_grid_side(side_a, thickness, outward, le_o, te_o, h_le, h_te, blade_size, window=window)[4],
                  _o_grid_side(side_b, thickness, outward, te_o, le_o, h_te, h_le, blade_size, window=window)[4])
    chunks_a = _o_grid_side(side_a, thickness, outward, le_o, te_o, h_le, h_te, blade_size, n_cells, window)
    chunks_b = _o_grid_side(side_b, thickness, outward, te_o, le_o, h_te, h_le, blade_size, n_cells, window)
    off_a, off_b = chunks_a[1], chunks_b[1]
    n_chord = n_cells + 1

    # ---- passage centreline copies and periodic sides ----------------------------
    def passage_pieces(offset):
        pieces = []
        for name in ("inlet_spline", "convergent_spline", "divergent", "outlet_spline"):
            if name == "divergent":
                pts = curves[name][[0, -1]] + [offset, 0.0]
                pieces.append(Line(pts[0], pts[1]))
            else:
                s = curves[name]
                pieces.append(Bspl(BSpline(s.t, s.c + [offset, 0.0], s.k)))
        return pieces

    def x_at_y(pieces, y):
        for piece in pieces:
            ys = (_start(piece)[1], _end(piece)[1])
            if min(ys) <= y <= max(ys):
                return float(piece.eval(_param_at_y(piece, y))[0][0])
        raise ValueError("Level outside the passage.")

    def split_at(pieces, y):
        for i, piece in enumerate(pieces):
            ys = (_start(piece)[1], _end(piece)[1])
            if min(ys) < y < max(ys):
                a, b = piece.split(_param_at_y(piece, y))
                return pieces[:i] + [a], [b] + pieces[i + 1:]
        raise ValueError("Split level outside the passage.")

    reference = passage_pieces(0.0)
    dx_le, dx_te = (corner[0] - x_at_y(reference, corner[1]) for corner in (le_o, te_o))
    if max(abs(dx_le), abs(dx_te)) >= 0.45 * pitch:
        raise ValueError("Blade corner too close to a periodic side for an O-H topology.")
    stagnation, _ = split_at(passage_pieces(dx_le), le_o[1])  # inlet -> LE'
    _, wake = split_at(passage_pieces(dx_te), te_o[1])        # TE' -> outlet
    y_in, y_out = float(curves["inlet"][0][1]), float(curves["outlet"][0][1])
    left_side = passage_pieces(-pitch / 2)

    def choose_level(corner, direction, y_lo, y_hi):
        """Level whose left / right periodic points are both seen from
        `corner` well inside their quadrant (between `direction` and the
        perpendicular towards that side): maximise the worst corner angle."""
        normal = np.array([-direction[1], direction[0]])
        toward_left = normal if normal[0] < 0 else -normal
        best = (-np.inf, None)
        for y in np.linspace(y_lo, y_hi, 241):
            point_left = np.array([x_at_y(left_side, y), y])
            margins = []
            for target, n in ((point_left, toward_left), (point_left + [pitch, 0.0], -toward_left)):
                v = target - corner
                a_d, a_n = _angle(v, direction), _angle(v, n)
                margins.append(min(a_d, a_n) if a_d + a_n <= 90.5 else -1.0)
            if min(margins) > best[0]:
                best = (min(margins), y)
        if best[0] < MIN_CONNECTOR_ANGLE:
            raise ValueError(f"No good connector level (best corner angle {best[0]:.1f} deg).")
        return best[1], best[0]

    y_up, angle_up = choose_level(le_o, upstream, le_o[1] + 0.03 * (y_in - le_o[1]),
                                  y_in - 0.15 * (y_in - le_o[1]))
    y_dn, angle_dn = choose_level(te_o, downstream, y_out + 0.15 * (te_o[1] - y_out),
                                  te_o[1] - 0.03 * (te_o[1] - y_out))
    sides = {}
    for name, offset in (("left", -pitch / 2), ("right", pitch / 2)):
        upper, rest = split_at(passage_pieces(offset), y_up)
        middle, lower = split_at(rest, y_dn)
        sides[name] = {"upper": upper, "middle": middle, "lower": lower}

    def mid_distance(offset_pts, side):
        piece = offset_pts[len(offset_pts) // 2]
        probe = piece[len(piece) // 2]
        pts = np.vstack([p.samples() for p in sides[side]["middle"]])
        return np.min(np.linalg.norm(pts - probe, axis=1))
    a_faces = "left" if mid_distance(off_a, "left") < mid_distance(off_a, "right") else "right"

    def facing(side_name):
        """O-grid offset chunks facing `side_name`, in flow order LE' -> TE'."""
        if side_name == a_faces:
            return chunks_a[1], chunks_a[2], chunks_a[3]
        return ([pts[::-1] for pts in chunks_b[1][::-1]], chunks_b[2][::-1],
                [1.0 / ratio for ratio in chunks_b[3][::-1]])

    def clearance(side_name):
        pts = np.vstack(facing(side_name)[0])[::4]
        per = np.vstack([p.samples() for p in sides[side_name]["middle"]])[::4]
        return float(np.min(np.linalg.norm(pts[:, None] - per[None], axis=2)))

    # The narrower channel drives the periodic-side distribution; the other
    # periodic side gets identical (translated) splits, as periodicity needs.
    driver = min(("left", "right"), key=clearance)
    other_side = "right" if driver == "left" else "left"
    middle_pieces, middle_counts, middle_ratios, splits = _matched_periodic_split(
        sides[driver]["middle"], *facing(driver))
    sides[driver]["middle"] = middle_pieces
    sides[other_side]["middle"] = _apply_splits(sides[other_side]["middle"], splits)[0]

    # ---- node counts ------------------------------------------------------------
    targets = {f"{end}_{side}": _end(sides[side]["upper" if end == "le" else "middle"][-1])
               for end in ("le", "te") for side in ("left", "right")}
    corner_of = {"le": le_o, "te": te_o}
    diag_length = {key: float(np.linalg.norm(t - corner_of[key[:2]])) for key, t in targets.items()}
    n_diag = {side: max(_geometric(diag_length[f"{end}_{side}"], first_core, max_last=domain_size)[0]
                        for end in ("le", "te")) for side in ("left", "right")}

    def run_total(pieces):
        return max(5, int(np.ceil(sum(_length(p) for p in pieces) / domain_size)) + 1, len(pieces) + 1)
    n_up = max(run_total(stagnation), run_total(sides["left"]["upper"]))
    n_down = max(run_total(wake), run_total(sides["left"]["lower"]))

    # ---- Gmsh CAD -----------------------------------------------------------------
    with base._gmsh_session({"verbosity": 2}):
        gmsh.model.add("stator_structured")
        model = _Model()

        def add_side(pieces):  # per-piece tag groups (an arc is two half-arcs)
            groups = []
            for piece in pieces:
                tag, _, _ = model.curve(piece)
                groups.append(list(tag) if isinstance(tag, tuple) else [tag])
            return groups

        P = model.point
        blade_a, blade_b = add_side(chunks_a[0]), add_side(chunks_b[0])
        offset_a = [[model.spline_through(pts)[0]] for pts in off_a]
        offset_b = [[model.spline_through(pts)[0]] for pts in off_b]
        radial_le = gmsh.model.occ.addLine(P(le), P(le_o))
        radial_te = gmsh.model.occ.addLine(P(te), P(te_o))
        diag = {key: gmsh.model.occ.addLine(P(corner_of[key[:2]]), P(t)) for key, t in targets.items()}
        stag, wk = add_side(stagnation), add_side(wake)
        periodic = {side: {part: add_side(sides[side][part]) for part in ("upper", "middle", "lower")}
                    for side in ("left", "right")}
        left_in, right_in = P(_start(sides["left"]["upper"][0])), P(_start(sides["right"]["upper"][0]))
        left_out, right_out = P(_end(sides["left"]["lower"][-1])), P(_end(sides["right"]["lower"][-1]))
        m_in, m_out = P(_start(stagnation[0])), P(_end(wake[-1]))
        inlet_l, inlet_r = gmsh.model.occ.addLine(left_in, m_in), gmsh.model.occ.addLine(m_in, right_in)
        outlet_r, outlet_l = gmsh.model.occ.addLine(right_out, m_out), gmsh.model.occ.addLine(m_out, left_out)
        gmsh.model.occ.synchronize()

        flat = lambda groups: [tag for group in groups for tag in group]  # noqa: E731
        rev = lambda tags: [-t for t in reversed(tags)]  # noqa: E731
        other = "right" if a_faces == "left" else "left"
        offset_fwd = {a_faces: flat(offset_a), other: rev(flat(offset_b))}
        c = {"le": P(le), "te": P(te), "le_o": P(le_o), "te_o": P(te_o), "m_in": m_in, "m_out": m_out,
             "left_in": left_in, "right_in": right_in, "left_out": left_out, "right_out": right_out,
             **{key: P(t) for key, t in targets.items()}}
        loops = {
            "o_a": (flat(blade_a) + [radial_te] + rev(flat(offset_a)) + [-radial_le], ("le", "te", "te_o", "le_o")),
            "o_b": (flat(blade_b) + [radial_le] + rev(flat(offset_b)) + [-radial_te], ("te", "le", "le_o", "te_o")),
            "upstream_left": ([inlet_l] + flat(stag) + [diag["le_left"]] + rev(flat(periodic["left"]["upper"])),
                              ("left_in", "m_in", "le_o", "le_left")),
            "upstream_right": ([inlet_r] + flat(periodic["right"]["upper"]) + [-diag["le_right"]] + rev(flat(stag)),
                               ("m_in", "right_in", "le_right", "le_o")),
            "downstream_left": ([-diag["te_left"]] + flat(wk) + [outlet_l] + rev(flat(periodic["left"]["lower"])),
                                ("te_left", "te_o", "m_out", "left_out")),
            "downstream_right": ([diag["te_right"]] + flat(periodic["right"]["lower"]) + [outlet_r] + rev(flat(wk)),
                                 ("te_o", "te_right", "right_out", "m_out")),
        }
        for side in ("left", "right"):
            loops[f"channel_{side}"] = (
                offset_fwd[side] + [diag[f"te_{side}"]] + rev(flat(periodic[side]["middle"])) + [-diag[f"le_{side}"]],
                ("le_o", "te_o", f"te_{side}", f"le_{side}"))
        faces = {}
        for name, (signed, corner_keys) in loops.items():
            loop = gmsh.model.occ.addCurveLoop(signed)
            faces[name] = (gmsh.model.occ.addPlaneSurface([loop]), [c[k] for k in corner_keys])
        gmsh.model.occ.synchronize()

        # ---- transfinite counts ---------------------------------------------------
        def set_group(groups, counts, kind="Progression", coef=1.0):
            for group, n in zip(groups, counts):
                if len(group) == 2:  # arc as two half-arcs
                    n1 = max(2, n // 2 + 1)
                    gmsh.model.mesh.setTransfiniteCurve(group[0], n1)
                    gmsh.model.mesh.setTransfiniteCurve(group[1], n - n1 + 1)
                else:
                    gmsh.model.mesh.setTransfiniteCurve(group[0], n, kind, coef)

        for groups, offs, chunks in ((blade_a, offset_a, chunks_a), (blade_b, offset_b, chunks_b)):
            for group, off, n, progression in zip(groups, offs, chunks[2], chunks[3]):
                set_group([group], [n], "Progression", progression)
                gmsh.model.mesh.setTransfiniteCurve(off[0], n, "Progression", progression)
        for radial in (radial_le, radial_te):
            gmsh.model.mesh.setTransfiniteCurve(radial, layers + 1, "Progression", ratio)
        for key, tag in diag.items():
            n = n_diag[key.split("_")[1]]
            gmsh.model.mesh.setTransfiniteCurve(tag, n, "Progression", _ratio_for(diag_length[key], first_core, n))
        for tag, side in ((inlet_l, "left"), (outlet_l, "left"), (inlet_r, "right"), (outlet_r, "right")):
            gmsh.model.mesh.setTransfiniteCurve(tag, n_diag[side])
        set_group(stag, _distribute([_length(p) for p in stagnation], n_up))
        set_group(wk, _distribute([_length(p) for p in wake], n_down))
        for side in ("left", "right"):
            set_group(periodic[side]["upper"], _distribute([_length(p) for p in sides["left"]["upper"]], n_up))
            for group, n, progression in zip(periodic[side]["middle"], middle_counts, middle_ratios):
                set_group([group], [n], "Progression", progression)
            set_group(periodic[side]["lower"], _distribute([_length(p) for p in sides["left"]["lower"]], n_down))
        for tag, corner_tags in faces.values():
            gmsh.model.mesh.setTransfiniteSurface(tag, cornerTags=corner_tags)
            gmsh.model.mesh.setRecombine(2, tag)
        surfaces = [tag for tag, _ in faces.values()]

        left = flat(periodic["left"]["upper"]) + flat(periodic["left"]["middle"]) + flat(periodic["left"]["lower"])
        right = flat(periodic["right"]["upper"]) + flat(periodic["right"]["middle"]) + flat(periodic["right"]["lower"])
        gmsh.model.mesh.setPeriodic(1, right, left, [1, 0, 0, pitch, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        boundaries = {"blade": flat(blade_a) + flat(blade_b), "periodic_left": left, "periodic_right": right,
                      "inlet": [inlet_l, inlet_r], "outlet": [outlet_r, outlet_l]}
        for name, tags in boundaries.items():
            gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, tags), name)
        gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, surfaces), "fluid")

        gmsh.model.mesh.generate(2)
        block_quality = {}
        for name, (tag, _) in faces.items():
            _, element_tags, _ = gmsh.model.mesh.getElements(2, tag)
            block_quality[name] = float(np.min(gmsh.model.mesh.getElementQualities(
                np.concatenate(element_tags), "minSJ")))
        bad = {name: q for name, q in block_quality.items() if q <= 0}
        if bad:
            details = []
            for name in bad:
                if name.startswith("channel_"):
                    gap = clearance(name.split("_")[1])
                    details.append(f"the {name.split('_')[1]} channel block folds (inverted cells); its "
                                   f"narrowest O-grid-to-periodic gap is {gap:.3g} mm "
                                   f"({gap / pitch:.1%} of the pitch)")
                else:
                    details.append(f"inverted cells in block {name}")
            raise RuntimeError("; ".join(details))
        statistics = base.mesh_statistics(surfaces, boundaries)
        if statistics["quadrilateral_fraction"] < 1.0:
            raise RuntimeError("Structured blocks produced non-quadrilateral cells.")
        xy, cells, named_edges, local = base.collect_fluid_mesh(surfaces, boundaries)
        current, donor, _ = base.collect_periodic_nodes(boundaries, local, xy)

    return {
        "topology": "structured",
        "fallback_reason": None,
        "xy": xy.tolist(),
        "cells": [cell.tolist() for cell in cells],
        "named_edges": {name: edges.tolist() for name, edges in named_edges.items()},
        "periodic": {"current": current.tolist(), "donor": donor.tolist()},
        "pitch": pitch,
        "statistics": {
            "cells": {name: int(count) for name, count in statistics["cells"].items()},
            "quadrilateral_fraction": float(statistics["quadrilateral_fraction"]),
            "minimum_scaled_jacobian": float(statistics["minimum_scaled_jacobian"]),
            "periodic_node_pairs": int(statistics["periodic_node_pairs"]),
            "inflation_layers": layers,
            "stack_thickness_mm": thickness,
            "block_min_scaled_jacobian": block_quality,
            "connector_corner_angles_deg": {"leading_edge": angle_up, "trailing_edge": angle_dn},
        },
    }


def _ratio_for(length, first, nodes):
    """Geometric ratio giving `nodes` nodes over `length` with first cell `first`."""
    cells = nodes - 1
    if first * cells >= length:
        return 1.0
    lo, hi = 1.0 + 1e-12, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        total = first * (mid**cells - 1) / (mid - 1)
        lo, hi = (mid, hi) if total < length else (lo, mid)
    return 0.5 * (lo + hi)
