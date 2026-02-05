#!/usr/bin/env python3
"""
Ditto TalkingHead Service
Single service with proxy + background GPU workers
Automatic load balancing based on GPU utilization
"""

import os
import sys
import signal
import subprocess
import asyncio
import httpx
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import Response, HTMLResponse, StreamingResponse
import uvicorn

# Configuration from environment
PROXY_PORT = int(os.environ.get("DITTO_PORT", "5097"))
WORKER_PORT_GPU0 = int(os.environ.get("DITTO_WORKER_PORT_0", "5095"))
WORKER_PORT_GPU1 = int(os.environ.get("DITTO_WORKER_PORT_1", "5096"))
WORKER_PORTS = {0: WORKER_PORT_GPU0, 1: WORKER_PORT_GPU1}
WORKER_SCRIPT = os.environ.get("DITTO_WORKER_SCRIPT", os.path.join(os.path.dirname(__file__), "ditto_worker.py"))
PYTHON_PATH = os.environ.get("DITTO_PYTHON", sys.executable)
WORK_DIR = os.environ.get("DITTO_WORK_DIR", os.path.dirname(__file__) or ".")

# Global worker processes
workers = {}

# Session to worker mapping (session_id -> port)
session_workers = {}

app = FastAPI(title="Ditto TalkingHead Service")


def get_gpu_utilization() -> dict:
    """Get GPU utilization."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        )
        stats = {}
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(',')
                if len(parts) >= 3:
                    gpu_id = int(parts[0].strip())
                    stats[gpu_id] = {
                        "util": float(parts[1].strip()),
                        "mem": float(parts[2].strip())
                    }
        return stats
    except Exception:
        return {}


async def check_worker_health(port: int) -> bool:
    """Check if worker is healthy."""
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            r = await client.get(f"http://127.0.0.1:{port}/")
            return r.status_code == 200
    except Exception:
        return False


def start_worker(gpu_id: int, port: int) -> subprocess.Popen:
    """Start a worker process for specific GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["DITTO_PORT"] = str(port)

    proc = subprocess.Popen(
        [PYTHON_PATH, WORKER_SCRIPT],
        cwd=WORK_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(f"[Proxy] Started worker GPU{gpu_id} on port {port} (PID: {proc.pid})")
    return proc


def stop_workers():
    """Stop all worker processes."""
    for gpu_id, proc in workers.items():
        if proc and proc.poll() is None:
            print(f"[Proxy] Stopping worker GPU{gpu_id} (PID: {proc.pid})")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


@app.on_event("startup")
async def startup():
    """Start worker processes on startup."""
    print(f"[Proxy] Starting Ditto Service on port {PROXY_PORT}")
    print(f"[Proxy] Workers: {WORKER_PORTS}")

    for gpu_id, port in WORKER_PORTS.items():
        workers[gpu_id] = start_worker(gpu_id, port)

    # Wait for workers to be ready
    print("[Proxy] Waiting for workers to initialize...")
    for _ in range(60):  # Max 60 seconds
        await asyncio.sleep(1)
        all_ready = True
        for gpu_id, port in WORKER_PORTS.items():
            if not await check_worker_health(port):
                all_ready = False
                break
        if all_ready:
            print("[Proxy] All workers ready!")
            break
    else:
        print("[Proxy] Warning: Some workers may not be ready")


@app.on_event("shutdown")
async def shutdown():
    """Stop worker processes on shutdown."""
    stop_workers()


def select_best_worker() -> tuple:
    """Select worker with lowest GPU utilization. Returns (gpu_id, port)."""
    stats = get_gpu_utilization()

    best_gpu = 0
    best_util = float('inf')

    for gpu_id, port in WORKER_PORTS.items():
        util = stats.get(gpu_id, {}).get("util", 100)
        if util < best_util:
            best_util = util
            best_gpu = gpu_id

    return best_gpu, WORKER_PORTS[best_gpu]


@app.get("/", response_class=HTMLResponse)
async def root():
    """Status page."""
    stats = get_gpu_utilization()

    worker_status = []
    for gpu_id, port in WORKER_PORTS.items():
        healthy = await check_worker_health(port)
        util = stats.get(gpu_id, {}).get("util", -1)
        mem = stats.get(gpu_id, {}).get("mem", -1)
        worker_status.append(f"GPU{gpu_id} (:{port}): {'✅' if healthy else '❌'} | {util}% | {mem}MB")

    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head><title>Ditto Service</title>
    <style>
        body {{ font-family: Arial; max-width: 800px; margin: 0 auto; padding: 20px; }}
        .status {{ background: #e9ecef; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .upload {{ background: #f5f5f5; padding: 20px; border-radius: 10px; }}
        input {{ margin: 10px 0; padding: 10px; width: 100%; }}
        button {{ background: #007bff; color: white; padding: 15px 30px; border: none; border-radius: 5px; cursor: pointer; }}
        pre {{ background: #f8f9fa; padding: 10px; border-radius: 5px; }}
    </style>
    </head>
    <body>
        <h1>🎭 Ditto TalkingHead Service</h1>
        <div class="status">
            <h3>Workers</h3>
            <pre>{"<br>".join(worker_status)}</pre>
        </div>
        <div class="upload">
            <h3>Test</h3>
            <input type="file" id="img" accept="image/*">
            <input type="file" id="aud" accept="audio/*">
            <button onclick="run()">🚀 Generate</button>
            <div id="result"></div>
        </div>
        <script>
        async function run() {{
            const img = document.getElementById('img').files[0];
            const aud = document.getElementById('aud').files[0];
            if (!img || !aud) return alert('Select files');
            document.getElementById('result').innerHTML = '⏳ Processing...';
            const fd = new FormData();
            fd.append('image_file', img);
            fd.append('wav_file', aud);
            try {{
                const r = await fetch('/inference', {{method: 'POST', body: fd}});
                if (r.ok) {{
                    const blob = await r.blob();
                    document.getElementById('result').innerHTML = '<video controls src="' + URL.createObjectURL(blob) + '" style="max-width:100%"></video>';
                }} else {{
                    document.getElementById('result').innerHTML = '❌ ' + await r.text();
                }}
            }} catch(e) {{
                document.getElementById('result').innerHTML = '❌ ' + e;
            }}
        }}
        </script>
    </body>
    </html>
    """)


@app.get("/api/gpu_status")
async def gpu_status():
    """Get GPU and worker status."""
    stats = get_gpu_utilization()
    result = []

    for gpu_id, port in WORKER_PORTS.items():
        healthy = await check_worker_health(port)
        result.append({
            "gpu_id": gpu_id,
            "port": port,
            "healthy": healthy,
            "utilization": stats.get(gpu_id, {}).get("util", -1),
            "memory_used": stats.get(gpu_id, {}).get("mem", -1),
        })

    return {"workers": result}


@app.post("/inference")
async def inference(
    image_file: UploadFile = File(...),
    wav_file: UploadFile = File(...)
):
    """Route inference to best worker."""
    gpu_id, port = select_best_worker()
    print(f"[Proxy] Routing to GPU{gpu_id} (:{port})")

    image_data = await image_file.read()
    wav_data = await wav_file.read()

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            files = {
                "image_file": ("image.jpg", image_data, "image/jpeg"),
                "wav_file": ("audio.wav", wav_data, "audio/wav"),
            }
            response = await client.post(f"http://127.0.0.1:{port}/inference", files=files)

            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)

            return Response(
                content=response.content,
                media_type="video/mp4",
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Worker timeout")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Worker error: {str(e)}")


# === Streaming API endpoints ===

@app.post("/api/create_simple_session")
async def create_simple_session(
    source_image: UploadFile = File(...),
    audio_file: UploadFile = File(...)
):
    """Create streaming session, route to best worker."""
    gpu_id, port = select_best_worker()
    print(f"[Proxy] Creating session on GPU{gpu_id} (:{port})")

    image_data = await source_image.read()
    audio_data = await audio_file.read()

    async with httpx.AsyncClient(timeout=30) as client:
        files = {
            "source_image": ("image.png", image_data, "image/png"),
            "audio_file": ("audio.wav", audio_data, "audio/wav"),
        }
        response = await client.post(f"http://127.0.0.1:{port}/api/create_simple_session", files=files)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        data = response.json()
        session_id = data.get("session_id")

        # Track which worker handles this session
        session_workers[session_id] = port
        print(f"[Proxy] Session {session_id} -> GPU{gpu_id} (:{port})")

        return data


@app.get("/api/stream_simple/{session_id}")
async def stream_simple(session_id: str, request: Request):
    """Proxy video stream from worker."""
    port = session_workers.get(session_id)
    if not port:
        raise HTTPException(status_code=404, detail="Session not found")

    async def stream_generator():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", f"http://127.0.0.1:{port}/api/stream_simple/{session_id}") as response:
                async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                    yield chunk

    return StreamingResponse(
        stream_generator(),
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/status_simple/{session_id}")
async def status_simple(session_id: str):
    """Proxy status from worker."""
    port = session_workers.get(session_id)
    if not port:
        raise HTTPException(status_code=404, detail="Session not found")

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(f"http://127.0.0.1:{port}/api/status_simple/{session_id}")
        return response.json()


@app.get("/api/download_simple/{session_id}")
async def download_simple(session_id: str):
    """Proxy download from worker."""
    port = session_workers.get(session_id)
    if not port:
        raise HTTPException(status_code=404, detail="Session not found")

    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.get(f"http://127.0.0.1:{port}/api/download_simple/{session_id}")

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        # Cleanup session tracking
        del session_workers[session_id]

        return Response(
            content=response.content,
            media_type="video/mp4",
            headers={"Content-Disposition": f"attachment; filename=generated_{session_id}.mp4"}
        )


if __name__ == "__main__":
    # Handle signals for clean shutdown
    def signal_handler(sig, frame):
        print("\n[Proxy] Shutting down...")
        stop_workers()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"[Proxy] Starting Ditto Service")
    uvicorn.run(
        "ditto_proxy:app",
        host="0.0.0.0",
        port=PROXY_PORT,
        reload=False,
        workers=1,
    )
