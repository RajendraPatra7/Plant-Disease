import sys
import os

print("DEBUG 1: Script started", flush=True)

try:
    from app.main import app
    print("DEBUG 2: App imported successfully", flush=True)
except Exception as e:
    print(f"DEBUG ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()

if __name__ == "__main__":
    import uvicorn
    print("DEBUG 3: Starting uvicorn", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
