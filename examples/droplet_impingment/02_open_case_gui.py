"""Open the stage case in the Fluent GUI (left open) and list its zone types.

python 02_open_case_gui.py [processors]
"""
import sys

import ansys.fluent.core as pyfluent
from turbo_moc.fluent.common import boundary_types

from settings import CASE_NAME, RUN_DIR

solver = pyfluent.launch_fluent(product_version="24.2", mode="solver", dimension=3, precision="double",
                                processor_count=int(sys.argv[1]) if len(sys.argv) > 1 else 12, ui_mode="gui",
                                start_timeout=600, cwd=str(RUN_DIR), cleanup_on_exit=False)
solver.settings.file.read_case(file_name=str(RUN_DIR / CASE_NAME))
types = boundary_types(solver.settings.setup.boundary_conditions)
print({kind: sorted(name for name, k in types.items() if k == kind) for kind in sorted(set(types.values()))})
print("interfaces:", solver.settings.setup.mesh_interfaces.interface.get_object_names())
solver.exit = lambda *args, **kwargs: None  # Leave the GUI open
