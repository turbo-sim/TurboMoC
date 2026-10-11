# TurboMoC application implementation

The parent directory contains the two launch scripts, YAML cases and theory/roadmap notes. This directory contains the application and its implementation dependencies:

| File | Responsibility |
| --- | --- |
| `application.py` | Dash app factory, layout, controls and callbacks. |
| `workflow.py` | Dependency keys, geometry caching, artifact IO and worker launch/poll/cancel. |
| `worker.py` | Isolated background execution with live progress and logs. |
| `configuration.py` | Current configuration names, legacy-file adaptation and validation. |
| `geometry_source.py` | Canonical reference design and general meridional mapping adapters. |
| `preview.py` | Smooth, capped geometry tessellation and display normals. |
| `meshing_source.py` | Reference mesh, volume sweep, artifact publication and export adapters. |
| `quality_source.py` | Actual volume quality quantities and histogram data. |
| `inspection_source.py` | Optional backend slicing/worst-cell extraction. |
| `plots.py` | Labeled 2D engineering plots and quality histograms. |
| `assets/studio.js` | Browser-side VTK scenes, display copies and live mapper updates. |
| `assets/mesh_edges.js` | Cached display triangles and antialiased screen-space cell-edge shader. |
| `assets/studio.css` | Application layout and styling. |

The application reuses the existing numerical code in `turbo_moc`, `examples/gmsh`, and `examples/gmsh_3D`; those shared implementations are not copied here. Third-party dependencies remain managed by the repository's `pyproject.toml`: Dash/dash-vtk for the interface, NumPy/SciPy for arrays and numerical geometry, Gmsh for meshing/export, VTK/PyVista for visualization, diskcache for geometry reuse, and PyYAML for configuration. Use the existing project environment.

Run `python -m examples.gmsh_3D_v2.run_mesh_app` from the repository root. Workers launch as `examples.gmsh_3D_v2.app.worker`; configuration and output locations stay relative to the parent example directory rather than this implementation directory.
