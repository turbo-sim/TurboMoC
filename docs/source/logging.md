# Logging and progress

TurboMoC reports through Python's standard `logging` module, under one
`turbo_moc` logger hierarchy:

| Logger | Source |
| --- | --- |
| `turbo_moc.api` | `design_nozzle`: inputs, retries, result summary |
| `turbo_moc.solvers.conventional`, `turbo_moc.solvers.mln` | Solver stages and progress |
| `turbo_moc.meshing.*` | Gmsh meshing helpers |
| `turbo_moc.workflows.gmsh.*`, `turbo_moc.workflows.fluent.*` | Example workflows |

Library modules never install handlers. Without configuration, Python's default
applies: warnings and errors reach stderr, routine progress stays silent.

## Console and log file

Wrap a run in `logging_context` to see its progress:

```python
from turbo_moc import design_nozzle
from turbo_moc.reporting import logging_context

with logging_context("nozzle.log", level="INFO"):  # file optional
    result = design_nozzle(fluid_name="CO2", P0=8.85e6, T0=308.15, p_back=3.97e6)
```

It configures only the `turbo_moc` logger (never the root logger), writes each
record once to the console and to a fresh UTF-8 file, and restores the previous
handlers, level and propagation on exit, including after exceptions and Ctrl+C. An
exception leaving the block is logged there once, with its traceback, and
re-raised. The Gmsh and Fluent workflows use the same context from
`turbo_moc.reporting`, so MoC, meshing and workflow messages share one
console and one log file.

A short successful run looks like this:

```text
  MoC: conventional; Nitrogen; P0 500.0 kPa, T0 300.00 K; 9 seed points; target pressure 100.0 kPa
  MoC: fluid properties initialized in 3.38 s (first-call setup and JIT compilation included)
  MoC [1/4] Throat front (uniform post-sonic state)
  MoC [1/4] Throat front (uniform post-sonic state) completed in 0.10 s
  MoC [2/4] Initial-value line
  MoC [2/4] Initial-value line completed in 0.25 s
  MoC [3/4] Kernel region
  MoC [3/4] Kernel region reached the target in 0.93 s
  MoC [4/4] Reflex region
  MoC [4/4] Reflex region completed in 0.07 s

  MoC results
    Status            Converged
    Exit pressure     100.0 kPa
    Target pressure   100.0 kPa
    Exit Mach         1.7088 (design 1.7088)
    Total runtime     4.79 s
```

## Levels

- **INFO**: inputs and target, fluid initialization, each stage's start and
  completion with its elapsed time, progress of long stages at most once per
  second, and the result summary.
- **DEBUG**: per-iteration diagnostics (kernel angle, adaptive step, axis state,
  mass flow), critical-state details, fluid-manager reuse. These are computed
  only when DEBUG is enabled.
- **WARNING**: recoverable property failures, an exhausted angular range, use of
  the last valid front, automatic retries with their modified inlet quality, and
  non-converged results. Each says what happened and what the solver does next.

Finishing all stages is not the same as reaching the design target: the summary's
*Status* uses the solver's own convergence check (exit Mach within 5 % of the
design value), and a non-converged result is also reported as a warning.

## Progress callbacks

`design_nozzle(..., progress_callback=f)` calls `f(fraction, stage_label)` on every
update, independently of the log level and of the console's once-per-second
throttling; `turbo_moc.make_progress_reporter` turns it into the app's progress bar.
Labels start with the stage (`"1 of 4"` to `"4 of 4"`, `"4A of 4"` for the
conventional reflex region). Fractions are finite, within [0, 1] and never decrease
within a stage:

- counted loops report work done over total work;
- the conventional kernel reports the fraction of the initial gap to its
  termination target closed (pressure, or Mach without a back pressure). This is
  an estimate of convergence, not of the remaining runtime;
- the minimum-length kernel reports the swept fraction of its prescribed fan angle.

An automatic retry restarts at stage 1 with labels such as
`"1 of 4 (retry 2 of 6)"`. The final update names the outcome:
`"4 of 4 (finished)"` or `"4 of 4 (finished, not converged)"`.

## Timing

Elapsed times are measured once with `time.perf_counter()` at stage boundaries.
They include first-call setup and JAX compilation where these happen (notably fluid
initialization), which is not measured separately.
