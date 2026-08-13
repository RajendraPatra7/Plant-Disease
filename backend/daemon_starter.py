import subprocess
import time
import os
import sys

try:
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    python_bin = "/Users/thelucifer/anaconda3/envs/tensorflow/bin/python3"

    env = os.environ.copy()
    env["PYTHONPATH"] = backend_dir

    cmd = [python_bin, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

    proc = subprocess.Popen(
        cmd,
        cwd=backend_dir,
        env=env,
        start_new_session=True
    )
    print(f"Backend started with PID: {proc.pid}", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)
    import traceback
    traceback.print_exc()
