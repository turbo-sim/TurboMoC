"""Differentiable (JAX) rotor design path of TurboMoC (2026-11-16/17), OPTIONAL: nothing in the app imports it.

The NumPy implementation in ``turbo_moc.core`` stays the reference used by the app and the stage workflows; this
package gives exact derivatives (``jax.grad``/``jacfwd``) and batched evaluation (``jit``/``vmap``) of the same rotor,
for sensitivity maps, sweeps and gradient-based design. ``tests/test_jax_rotor_equivalence.py`` fails when the two
drift apart: a change to ``core.vortex_blade`` / ``core.rotor_starting`` must be mirrored here. Full verification
(260 blades, 4 fluids, flashing / wet-to-dry): worktree ``TurboMoC_jax``, ``tests/jax_rotor/VERIFICATION.md``.
Plan: ``dev/jax_differentiability_plan.md``.

- ``isentrope``: real-gas Prandtl-Meyer / sound speed / density lookups along the relative isentrope, evaluated in JAX
  from the same table nodes and spline coefficients as ``thermo_characteristics.FluidManager``.
- ``vortex_blade``: Goldman & Scullin (NASA TN D-4421) vortex-flow rotor blade, as ``core.vortex_blade``.
- ``starting``: geometric throat and Kantrowitz starting check, as ``core.rotor_starting``.
"""
import jax

jax.config.update("jax_enable_x64", True)  # the blade geometry needs float64
