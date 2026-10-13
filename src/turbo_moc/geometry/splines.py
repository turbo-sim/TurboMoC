"""
turbo_moc.geometry.splines -- B-spline fitting/evaluation used by
turbo_moc.geometry.stator_blade to reduce a dense blade-profile polyline to a small set
of control points (for CAD/optimization use) and reconstruct a smooth
curve from them.

Originally ported from opt_framework/test_2/01_geometry_parametrization/utils/
spline_gen.py (BSplineTool). Basis assembly/evaluation use SciPy's compiled routines;
parameterization, normal-equation fitting, and endpoint conventions are retained.
"""

import numpy as np
from scipy import linalg
from scipy.interpolate import BSpline

__all__ = ["BSplineTool"]


class BSplineTool:
    def __init__(self, degree=3):
        """
        Initializes the B-Spline tool with a specific degree (default k=3).
        """
        self.n = degree

    def _get_basis(self, i, k, u, knots):
        """
        Recursive Cox-de Boor algorithm to compute basis functions.
        """
        if k == 0:
            return 1.0 if knots[i - 1] <= u < knots[i] else 0.0

        denom1 = knots[i + k - 1] - knots[i - 1]
        denom2 = knots[i + k] - knots[i]

        val = 0
        if denom1 != 0:
            val += (u - knots[i - 1]) / denom1 * self._get_basis(i, k - 1, u, knots)
        if denom2 != 0:
            val += (knots[i + k] - u) / denom2 * self._get_basis(i + 1, k - 1, u, knots)
        return val

    def _get_params_and_knots(self, data_points, num_cp):
        P_len = len(data_points)
        # Centripetal Parameterization
        dists = np.sqrt(np.sum(np.diff(data_points, axis=0) ** 2, axis=1))
        w = np.zeros(P_len)
        w[1:] = np.cumsum(np.power(dists, 0.5))
        w /= w[-1]

        # Clamped Knot Sequence (Ensures endpoint interpolation)
        knots = np.concatenate([
            np.zeros(self.n + 1),
            np.linspace(0, 1, num_cp - self.n + 1)[1:-1],
            np.ones(self.n + 1)
        ])
        return w, knots

    def reparameterize_equidistant(self, data_points, num_points=500):
        """
        Re-samples data points so they are physically equidistant along the arc length.
        Prevents clustered data from distorting CP distribution.
        """
        # Calculate cumulative arc length
        dists = np.sqrt(np.sum(np.diff(data_points, axis=0) ** 2, axis=1))
        arc_length = np.concatenate(([0], np.cumsum(dists)))

        # Create a uniform distribution of length
        uniform_lengths = np.linspace(0, arc_length[-1], num_points)

        # Interpolate to find new equidistant (x, y) coordinates
        new_x = np.interp(uniform_lengths, arc_length, data_points[:, 0])
        new_y = np.interp(uniform_lengths, arc_length, data_points[:, 1])

        return np.vstack((new_x, new_y)).T

    def backward(self, data_points, num_cp):
        """
        Generates control points from data points using Least Squares.
        Uses a Uniform Knot Sequence for equispaced control variables.
        """
        P = len(data_points)

        # 1. Centripetal Parameterization
        dists = np.sqrt(np.sum(np.diff(data_points, axis=0) ** 2, axis=1))
        w = np.zeros(P)
        w[1:] = np.cumsum(np.power(dists, 0.5))
        w /= w[-1]

        # 2. Knot Sequence
        knots = np.concatenate([
            np.zeros(self.n + 1),
            np.linspace(0, 1, num_cp - self.n + 1)[1:-1],
            np.ones(self.n + 1)
        ])

        # 3. Build Linear System A*x = B solved with Cholesky
        N_mat = self._basis_matrix(w, knots)

        A = N_mat.T @ N_mat
        B = N_mat.T @ data_points

        L_cho = linalg.cholesky(A, lower=True)
        control_points = linalg.cho_solve((L_cho, True), B)

        return control_points, knots

    # The following employs the backward approach while fixing the start and end points
    def backward_fixed(self, data_points, num_cp):
        w, knots = self._get_params_and_knots(data_points, num_cp)

        N_mat = self._basis_matrix(w, knots)

        # Force d0 and dL to match first and last data points
        d0, dL = data_points[0], data_points[-1]
        target_B = data_points - (np.outer(N_mat[:, 0], d0) + np.outer(N_mat[:, -1], dL))

        N_inner = N_mat[:, 1:-1]
        A_red = N_inner.T @ N_inner
        B_red = N_inner.T @ target_B

        L_cho = linalg.cholesky(A_red, lower=True)
        inner_cps = linalg.cho_solve((L_cho, True), B_red)

        control_points = np.vstack([d0, inner_cps, dL])
        return control_points, knots

    def forward(self, control_points, knots, num_points=500):
        """
        Generates the curve from control points (Forward Method).
        """
        u_range = np.linspace(knots[self.n], knots[-self.n - 1], num_points)
        # Preserve the legacy upper-end sampling so CAD endpoints and existing
        # fitted contours do not change during this performance refactor.
        u_range[u_range == knots[-self.n - 1]] -= 1e-10
        return BSpline(knots, np.asarray(control_points), self.n)(u_range)

    def _basis_matrix(self, parameters, knots):
        """Compiled basis evaluation with the historical half-open endpoint."""
        matrix = BSpline.design_matrix(parameters, knots, self.n).toarray()
        # The recursive reference returns zero for all bases at the upper knot.
        # Retain that convention in both fit methods (including unconstrained
        # fits), rather than changing the least-squares problem accidentally.
        matrix[np.asarray(parameters) == knots[-self.n - 1]] = 0.0
        return matrix
