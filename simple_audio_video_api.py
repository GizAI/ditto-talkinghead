#!/usr/bin/env python3
"""
Simple Audio+Video Streaming API
Uses modified StreamSDK with built-in audio+video generation
"""

import os
import time
import uuid
import threading
import asyncio
from typing import Dict, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
import librosa

# Prefer system ffmpeg with libx264 if available
os.environ.setdefault("FFMPEG_BINARY", "/usr/bin/ffmpeg")

# Import modified Ditto SDK
from stream_pipeline_online import StreamSDK

app = FastAPI(title="Simple Audio+Video Streaming API")

# Global sessions storage
sessions: Dict[str, "SimpleAVSession"] = {}

class SimpleAVSession:
    def __init__(self, session_id: str, source_image_path: str, audio_path: str):
        self.session_id = session_id
        self.source_image_path = source_image_path
        self.audio_path = audio_path
        
        # Output path (will contain audio+video)
        self.output_path = f"./tmp/av_{session_id}.mp4"
        
        # State
        self.is_processing = False
        self.is_complete = False
        self.processed_frames = 0
        self.total_frames = 0
        
        # SDK
        self.sdk = None
        
    def start_processing(self, audio: np.ndarray):
        """Start processing with built-in audio+video generation"""
        if self.is_processing:
            return
            
        self.is_processing = True
        self.total_frames = len(audio) // 640  # Estimate frames
        
        # Start processing in background thread
        thread = threading.Thread(target=self._process_with_builtin_audio, args=(audio,))
        thread.daemon = True
        thread.start()
        
    def _process_with_builtin_audio(self, audio: np.ndarray):
        """Process video with built-in audio+video generation"""
        try:
            # Initialize SDK
            self.sdk = StreamSDK(
                "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl",
                "./checkpoints/ditto_trt_Ampere_Plus"
            )
            
            # Setup SDK with audio path for built-in audio+video generation
            setup_kwargs = {
                "online_mode": True,
                "sampling_timesteps": 10,
                "overlap_v2": 70,
                "smo_k_d": 3,
                "max_size": 1024,
            }
            
            # Setup SDK with audio path for built-in audio+video generation
            self.sdk.setup(self.source_image_path, self.output_path, audio_path=self.audio_path, **setup_kwargs)
            
            # Load audio and calculate total frames
            import math
            total_frames = math.ceil(len(audio) / 16000 * 25)
            self.sdk.setup_Nd(N_d=total_frames)
            
            # Hook into writer with a proxy so close() still works
            original_writer = self.sdk.writer

            class _ProgressWriterProxy:
                def __init__(self, inner, session_ref, sdk_ref):
                    self._inner = inner
                    self._session = session_ref
                    self._sdk = sdk_ref

                def __call__(self, frame, fmt="rgb"):
                    self._session.processed_frames += 1
                    if hasattr(self._inner, "__call__"):
                        self._inner(frame, fmt)

                def close(self):
                    if hasattr(self._inner, "close"):
                        self._inner.close()

                def __getattr__(self, name):
                    return getattr(self._inner, name)

            self.sdk.writer = _ProgressWriterProxy(original_writer, self, self.sdk)
            
            # Process audio chunks
            chunksize = (2, 3, 1)
            audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
            split_len = int(sum(chunksize) * 0.04 * 16000) + 80
            
            for i in range(0, len(audio), chunksize[1] * 640):
                audio_chunk = audio[i:i + split_len]
                if len(audio_chunk) < split_len:
                    audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
                
                self.sdk.run_chunk(audio_chunk, chunksize)
            
            # Close SDK (this will finalize the MP4 and stop ffmpeg live muxer)
            try:
                if hasattr(self.sdk, 'close'):
                    self.sdk.close()
            except Exception:
                pass

            # Remux a final MP4 for download with proper moov and duration
            try:
                import subprocess
                final_path = self.output_path.replace('.mp4', '.final.mp4')
                cmd = [
                    'ffmpeg', '-v', 'error', '-y',
                    '-i', self.output_path,
                    '-c', 'copy', '-movflags', '+faststart',
                    final_path,
                ]
                subprocess.run(cmd, check=True)
                self.final_path = final_path
            except Exception:
                self.final_path = None

            self.is_complete = True

        except Exception as e:
            print(f"Error in audio+video processing: {e}")
        finally:
            self.is_processing = False

@app.get("/", response_class=HTMLResponse)
async def get_client():
    """Serve simple HTML client"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Simple Audio+Video Streaming</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .upload-section { background: #f5f5f5; padding: 20px; border-radius: 10px; margin: 20px 0; }
            .file-input { margin: 10px 0; padding: 10px; width: 100%; }
            .btn { background: #007bff; color: white; padding: 15px 30px; border: none; border-radius: 5px; cursor: pointer; }
            .btn:hover { background: #0056b3; }
            .status { background: #e9ecef; padding: 15px; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>🎭 Simple Audio+Video Streaming</h1>
        
        <div class="upload-section">
            <h3>📁 Upload Files</h3>
            <input type="file" id="sourceImage" class="file-input" accept="image/*" placeholder="Source Image">
            <input type="file" id="audioFile" class="file-input" accept="audio/*" placeholder="Audio File">
            <button class="btn" onclick="createSession()">🚀 Generate Audio+Video</button>
        </div>
        
        <div id="status" class="status" style="display: none;">
            <h3>📊 Status</h3>
            <p id="statusText">Processing...</p>
            <p id="progressText">Frames: 0 / 0</p>
        </div>
        
        <div id="result" style="display: none;">
            <h3>🎬 Result</h3>
            <p id="resultText">Video ready!</p>
        </div>

        <script>
            let currentSessionId = null;
            let statusInterval = null;

            async function createSession() {
                const sourceImage = document.getElementById('sourceImage').files[0];
                const audioFile = document.getElementById('audioFile').files[0];

                if (!sourceImage || !audioFile) {
                    alert('Please select both image and audio files');
                    return;
                }

                try {
                    const formData = new FormData();
                    formData.append('source_image', sourceImage);
                    formData.append('audio_file', audioFile);

                    const response = await fetch('/api/create_simple_session', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();
                    currentSessionId = result.session_id;

                    document.getElementById('status').style.display = 'block';
                    startStatusMonitoring();

                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }

            function startStatusMonitoring() {
                window.streamStarted = false; // Reset streaming flag
                statusInterval = setInterval(async () => {
                    if (!currentSessionId) return;

                    try {
                        const response = await fetch(`/api/status_simple/${currentSessionId}`);
                        const status = await response.json();

                        document.getElementById('progressText').textContent = 
                            `Frames: ${status.processed_frames} / ${status.total_frames}`;

                        if (status.is_complete) {
                            document.getElementById('statusText').textContent = 'Complete!';
                            document.getElementById('result').style.display = 'block';

                            // If streaming hasn't started yet, open it now
                            if (!window.streamStarted) {
                                const videoUrl = `/api/stream_simple/${currentSessionId}`;
                                window.open(videoUrl, '_blank');
                            }

                            document.getElementById('resultText').innerHTML =
                                `<p>✅ Processing complete! Final high-quality video ready.</p>
                                 <a href="/api/download_simple/${currentSessionId}" target="_blank">📥 Download MP4</a>`;

                            clearInterval(statusInterval);
                        } else if (status.is_processing) {
                            document.getElementById('statusText').textContent = 'Generating audio+video...';

                            // Start streaming as soon as we have some content (after ~5 frames)
                            console.log('Status:', status.processed_frames, 'frames, streamStarted:', window.streamStarted);
                            if (status.processed_frames >= 5 && !window.streamStarted) {
                                window.streamStarted = true;
                                const videoUrl = `/api/stream_simple/${currentSessionId}`;
                                window.open(videoUrl, '_blank');

                                document.getElementById('result').style.display = 'block';
                                document.getElementById('resultText').innerHTML =
                                    `<p>🎬 Real-time streaming started in new tab!</p>
                                     <p>Video will continue updating as processing completes...</p>`;
                            }
                        }
                    } catch (error) {
                        console.error('Status error:', error);
                    }
                }, 1000);
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/api/create_simple_session")
async def create_simple_session(
    source_image: UploadFile = File(...),
    audio_file: UploadFile = File(...)
):
    """Create new simple audio+video session"""
    session_id = str(uuid.uuid4())
    
    # Create tmp directory
    os.makedirs("./tmp", exist_ok=True)
    
    # Save uploaded files
    source_image_path = f"./tmp/source_{session_id}.jpg"
    audio_path = f"./tmp/audio_{session_id}.wav"
    
    with open(source_image_path, "wb") as f:
        f.write(await source_image.read())
    
    with open(audio_path, "wb") as f:
        f.write(await audio_file.read())
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Create session
    session = SimpleAVSession(session_id, source_image_path, audio_path)
    sessions[session_id] = session
    
    # Start processing
    session.start_processing(audio)
    
    return {
        "session_id": session_id,
        "status": "processing_started"
    }

@app.get("/api/status_simple/{session_id}")
async def get_simple_status(session_id: str):
    """Get simple session status"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    return {
        "session_id": session_id,
        "is_processing": session.is_processing,
        "is_complete": session.is_complete,
        "processed_frames": session.processed_frames,
        "total_frames": session.total_frames,
        "output_size": os.path.getsize(session.output_path) if os.path.exists(session.output_path) else 0
    }

@app.get("/api/stream_simple/{session_id}")
async def stream_simple(session_id: str, request: Request):
    """Stream generated audio+video file with real-time updates and basic Range support"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    # Parse Range header if present (e.g., bytes=12345-)
    # Ignore Range to keep connection open and stream as file grows
    range_header = None
    range_start = 0
    range_end_requested: Optional[int] = None
    if range_header and range_header.startswith('bytes='):
        try:
            range_spec = range_header.split('=')[1]
            start_str, end_str = range_spec.split('-', 1)
            if start_str:
                range_start = max(0, int(start_str))
            if end_str:
                range_end_requested = max(range_start, int(end_str))
        except Exception:
            range_start = 0
            range_end_requested = None

    # streaming started

    # If a Range request is present, comply with 206 responses so browsers can buffer correctly.
    if range_header:
        async def _wait_for_bytes(min_size: int) -> int:
            while True:
                if os.path.exists(session.output_path):
                    current_size = os.path.getsize(session.output_path)
                    if current_size > min_size:
                        return current_size
                    if session.is_complete and current_size >= min_size:
                        return current_size
                if session.is_complete and not os.path.exists(session.output_path):
                    return 0
                await asyncio.sleep(0.1)

        available_size = await _wait_for_bytes(range_start)
        if available_size <= range_start:
            raise HTTPException(status_code=416, detail="Requested range not satisfiable")

        range_end = available_size - 1
        if range_end_requested is not None:
            range_end = min(range_end, range_end_requested)

        if range_end < range_start:
            raise HTTPException(status_code=416, detail="Requested range not satisfiable")

        bytes_to_send = range_end - range_start + 1

        def ranged_stream():
            with open(session.output_path, 'rb') as f:
                f.seek(range_start)
                remaining = bytes_to_send
                chunk_size = 64 * 1024
                while remaining > 0:
                    chunk = f.read(min(chunk_size, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        total_size_header = os.path.getsize(session.output_path) if session.is_complete and os.path.exists(session.output_path) else '*'
        headers = {
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Range": f"bytes {range_start}-{range_end}/{total_size_header}",
            "Content-Length": str(bytes_to_send),
        }

        return StreamingResponse(
            ranged_stream(),
            media_type="video/mp4",
            status_code=206,
            headers=headers,
        )

    # Fallback: progressive streaming without Range; keep connection open for live growth.
    def generate_realtime_stream():
        import time as _time
        last_size = 0
        while not os.path.exists(session.output_path) or os.path.getsize(session.output_path) < 1024:
            _time.sleep(0.1)
        while True:
            try:
                current_size = os.path.getsize(session.output_path)
                if current_size > last_size:
                    with open(session.output_path, 'rb') as f:
                        f.seek(last_size)
                        chunk = f.read(current_size - last_size)
                        if chunk:
                            yield chunk
                            last_size = current_size
                if session.is_complete:
                    with open(session.output_path, 'rb') as f:
                        f.seek(last_size)
                        tail = f.read()
                        if tail:
                            yield tail
                    break
                _time.sleep(0.1)
            except Exception as e:
                # swallow transient streaming errors; client may close early
                break
                break

    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }

    return StreamingResponse(
        generate_realtime_stream(),
        media_type="video/mp4",
        headers=headers,
    )

@app.get("/api/download_simple/{session_id}")
async def download_simple(session_id: str):
    """Download generated audio+video file"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    # Prefer the remuxed final file if available; fallback to the live fragmented file
    final_path = getattr(session, 'final_path', None)
    serve_path = final_path if final_path and os.path.exists(final_path) else session.output_path

    if not os.path.exists(serve_path):
        raise HTTPException(status_code=404, detail="File not ready")

    return FileResponse(
        serve_path,
        media_type="video/mp4",
        filename=f"generated_{session_id}.mp4"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "simple_audio_video_api:app",
        host="0.0.0.0",
        port=8010,
        reload=False,
        workers=1
    )
