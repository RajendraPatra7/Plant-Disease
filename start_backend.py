import os
import sys
import asyncio
import uvicorn

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

from backend.app.main import app

async def run_server():
    config = uvicorn.Config(app=app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    print("Starting Smart Spray X FastAPI Server on http://0.0.0.0:8000 ...", flush=True)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_server())
