import os
import sys

# Add package root dir to path
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")),
)

from moc import __version__

# Define project metadata
project = "Method of Characteristics"
copyright = "2026, Sustainable Thermal Power Research group at DTU Construct"
author = "Andrea Cioffi, Roberto Agromayor, Fredrik Haglind"
release = __version__

# Project-specific settings used by build_docs.py.
# Relative paths are resolved from the directory containing this file.
docs_build_config = {
    "source_dir": "source",
    "build_dir": "_build/html",
    "api_output_dir": "source/api",
    "src_dir": "../src",
    "exclude_modules": [],
    "sphinx_build_options": ["--fail-on-warning", "--keep-going"],
    "zotero_api_key_env": "ZOTERO_API_KEY",
    "zotero_group_id": "5271514",
    "bibliography_file": "source/references/bibliography.bib",
}

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinxcontrib.bibtex",
    "myst_parser",
    "sphinx_togglebutton",
    "sphinx_design",
]

# ---------------------------------------------------------------------------
# MyST-Parser — .md files are parsed by MyST, .rst kept for apidoc stubs
# ---------------------------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst",
}

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "substitution",
    "tasklist",
]

myst_heading_anchors = 3

# ---------------------------------------------------------------------------
# Autodoc / autosummary
# ---------------------------------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
numpydoc_class_members_toctree = False
todo_include_todos = True

# ---------------------------------------------------------------------------
# Bibliography
# ---------------------------------------------------------------------------
bibtex_bibfiles = [docs_build_config["bibliography_file"]]
bibtex_default_style = "alpha"
bibtex_reference_style = "author_year"

# The docname (path relative to this file, without extension, "/"-separated)
# of the page holding the ``{bibliography}`` directive.
bibliography_docname = "references/bibliography"


def _always_rebuild_bibliography(app, env, added, changed, removed):
    """Always treat the bibliography page as outdated.

    sphinxcontrib-bibtex populates the ``{bibliography}`` directive from
    citations it collects while *other* pages are read. Sphinx's incremental
    build (used by both plain `sphinx-build` reruns and `sphinx-autobuild`)
    only re-reads pages whose own source file changed on disk, so adding or
    removing a `{cite}` role on some other page does not, by itself, mark
    bibliography.md as needing a rebuild -- it silently goes stale. This is
    a known limitation of sphinxcontrib-bibtex with incremental builds; the
    ``env-get-outdated`` hook is the officially suggested workaround: it lets
    an extension add extra docnames to the outdated set for a given build,
    without touching any files on disk, so it can't create the file-watcher
    feedback loop that a `touch()`-based pre-build hook risks under
    sphinx-autobuild. It runs the same way for `sphinx-build`,
    `sphinx-autobuild`, and CI.
    """
    return [bibliography_docname]


def setup(app):
    app.connect("env-get-outdated", _always_rebuild_bibliography)


# ---------------------------------------------------------------------------
# Build settings
# ---------------------------------------------------------------------------
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "notes",
    "theory/old",
]

# ---------------------------------------------------------------------------
# HTML theme
# ---------------------------------------------------------------------------
html_theme = "sphinx_book_theme"
html_title = "Method of Characteristics"
html_baseurl = "https://turbo-sim.github.io/method_of_characteristics/"

html_theme_options = {
    "repository_url": "https://github.com/turbo-sim/method_of_characteristics",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    # Path from the repository root to the Sphinx source directory.
    # sphinx_book_theme prefixes this to each docname to build the
    # "Edit this page" links, so it must match the actual folder name.
    "path_to_docs": "docs/source",
    "show_toc_level": 2,
}
