"""
turbo_moc.progress -- combines each solver stage's own progress callback
(fraction: float, stage_label: str) into ONE overall (fraction, message)
stream, for a UI progress bar -- e.g. Dash's set_progress(...) inside a
background=True callback.

Stage weights below are rough, from profiling the reference case (Stage 3,
the kernel marching, dominates runtime by far; Stage 1/2/4 are comparatively
quick) -- meant to make the bar move at roughly the rate work is actually
happening, not to be exact. Every turbo_moc solver stage method
(run_stage_1_sauer, run_stage_2_ivp/..., run_stage_3_kernel/_mln,
run_stage_4_reflex) accepts progress_callback=<callable(fraction, label)>,
called with the label it already prints to the terminal ("1 of 4", "2 of
4", "3 of 4", "4 of 4"/"4A of 4") -- only the leading digit is used here to
route to a stage weight, so both solvers' label variants work unchanged.
"""

DEFAULT_STAGE_WEIGHTS = {
    "1": 0.03,
    "2": 0.10,
    "3": 0.80,
    "4": 0.07,
}

__all__ = ["DEFAULT_STAGE_WEIGHTS", "make_progress_reporter"]


def make_progress_reporter(set_progress, weights=None):
    """Create a callback that combines progress from all solver stages.

    Parameters
    ----------
    set_progress : callable
        Receives the ``(percent_string, message_string)`` tuple expected by
        a Dash background callback.
    weights : dict, optional
        Relative weights for solver stages 1 through 4. The defaults reflect
        their approximate runtime.

    Returns
    -------
    callable
        A reporter accepting ``(fraction, stage_label)``, suitable for a
        solver's ``progress_callback`` argument.

    Examples
    --------
    Use the returned reporter inside a Dash background callback::

        reporter = make_progress_reporter(set_progress)
        data = design_nozzle(..., progress_callback=reporter)
    """
    weights = weights or DEFAULT_STAGE_WEIGHTS
    order = ["1", "2", "3", "4"]
    cumulative_before = {}
    acc = 0.0
    for key in order:
        cumulative_before[key] = acc
        acc += weights.get(key, 0.0)
    total_weight = acc if acc > 0 else 1.0

    def reporter(fraction, stage_label):
        stage_key = stage_label.strip()[:1] or "1"
        base = cumulative_before.get(stage_key, 0.0)
        w = weights.get(stage_key, 0.0)
        local = max(0.0, min(1.0, fraction))
        overall = min(1.0, (base + w * local) / total_weight)
        set_progress((f"{overall * 100:.0f}%", f"Stage {stage_key} of 4"))

    return reporter
