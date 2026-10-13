"""Separate an app project from additional automation-only settings."""
from copy import deepcopy

PROJECT_FIELDS = {"machine", "workflow", "rows"}


def split_workflow(content, required, *, optional=()):
    """Accept an inline app-project superset or the legacy design_file form.

    Never choose silently between embedded and referenced geometry. Copies
    keep path resolution and adapters from mutating the user's project.
    """
    if not isinstance(content, dict):
        raise ValueError("Workflow YAML must be a mapping.")
    inline = bool(PROJECT_FIELDS & content.keys())
    expected = (required - {"design_file"}) | PROJECT_FIELDS if inline else required
    if not expected <= content.keys() or content.keys() - expected - set(optional):
        raise ValueError(f"Workflow YAML requires {sorted(expected)}; "
                         "use inline machine/workflow/rows or design_file, never both.")
    project = {key: deepcopy(content[key]) for key in PROJECT_FIELDS} if inline else None
    options = {key: deepcopy(value) for key, value in content.items()
               if not inline or key not in PROJECT_FIELDS}
    return options, project
