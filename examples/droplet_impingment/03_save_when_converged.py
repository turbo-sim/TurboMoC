"""Connect to the open GUI session, follow its console, and save case and data once it stops iterating.

python 03_save_when_converged.py <serverinfo file>
Finished = no new iteration line for IDLE_S after at least one iteration was seen.
"""
import re
import sys
import time
from datetime import datetime

import ansys.fluent.core as pyfluent

from settings import RUN_DIR

OUT = str(RUN_DIR / "stage_fluent_solved.cas.h5")
IDLE_S = 900
ITERATION = re.compile(r"^\s*(\d+)\s+[-+0-9.]+e[-+]\d+")
state = {"iteration": 0, "time": time.time(), "lines": []}


def follow(text):
    for line in text.splitlines():
        match = ITERATION.match(line)
        if match:
            state["iteration"], state["time"] = int(match.group(1)), time.time()
        elif any(word in line.lower() for word in ("floating point", "divergence", "converged")):
            print(datetime.now().strftime("%H:%M"), "FLUENT:", line.strip(), flush=True)


def main(server_info):
    solver = pyfluent.connect_to_fluent(server_info_file_name=server_info, cleanup_on_exit=False)
    solver.transcript.register_callback(follow)
    print(datetime.now().strftime("%H:%M"), "connected", flush=True)
    reported = 0
    while True:
        time.sleep(60)
        if state["iteration"] - reported >= 250:
            reported = state["iteration"]
            print(datetime.now().strftime("%H:%M"), "iteration", reported, flush=True)
        if state["iteration"] and time.time() - state["time"] > IDLE_S:
            print(datetime.now().strftime("%H:%M"), f"idle since iteration {state['iteration']}: saving", flush=True)
            solver.settings.file.batch_options.confirm_overwrite = False
            solver.settings.file.write_case_data(file_name=OUT)
            print(datetime.now().strftime("%H:%M"), "SAVED", OUT, flush=True)
            return


if __name__ == "__main__":
    main(*sys.argv[1:])
