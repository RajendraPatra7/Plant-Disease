import sys
import os
import traceback

backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)
out_log = os.path.join(backend_dir, "debug_err.txt")

with open(out_log, "w") as f:
    f.write("DEBUG RUNNER STARTED\n")
    try:
        from app.main import app
        import uvicorn
        f.write("APP LOADED SUCCESSFULLY. STARTING UVIOCRN...\n")
        f.flush()
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
    except Exception as e:
        f.write(f"EXCEPT: {e}\n")
        traceback.print_exc(file=f)
        f.flush()
