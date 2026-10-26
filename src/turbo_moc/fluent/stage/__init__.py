"""Stator–rotor stage in Fluent: case setup from the app design and the CFD report.

``setup.main`` runs the app jobs, builds the barotropic material and opens a
configured, unsolved Fluent case. ``report.main`` reads the solved case and
writes figures and a Markdown/HTML/PDF report against the MoC design. Importing
this package does not launch Fluent.
"""
