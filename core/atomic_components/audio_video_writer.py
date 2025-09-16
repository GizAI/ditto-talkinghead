import os
import subprocess
import threading
import time
from typing import Optional

import numpy as np


class AudioVideoWriter:
    """Stream frames into FFmpeg and mux audio while generation is running."""

    def __init__(self, video_path: str, audio_path: str, fps: int = 25, ffmpeg_binary: Optional[str] = None, **_: object) -> None:
        self.video_path = video_path
        self.audio_path = audio_path
        self.fps = fps
        self.frame_count = 0
        self.closed = False

        os.makedirs(os.path.dirname(self.video_path) or ".", exist_ok=True)

        if not os.path.exists(self.audio_path):
            raise FileNotFoundError(f"Audio source not found: {self.audio_path}")

        self.ffmpeg_binary = ffmpeg_binary or os.environ.get("FFMPEG_BINARY", "ffmpeg")
        self.proc: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_buffer: list[str] = []
        self._frame_shape: Optional[tuple[int, int]] = None

    def _start_ffmpeg(self, height: int, width: int) -> None:
        """Start FFmpeg once the first frame arrives so dimensions are known."""
        def _codec_supported(codec: str) -> bool:
            test_cmd = [
                self.ffmpeg_binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=16x16:r=1",
                "-frames:v",
                "1",
                "-c:v",
                codec,
                "-f",
                "null",
                "-",
            ]
            try:
                result = subprocess.run(
                    test_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=5,
                )
            except Exception:
                return False
            if result.returncode == 0:
                return True
            return False

        base_cmd = [
            self.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-fflags",
            "+genpts",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",
            "-thread_queue_size",
            "256",
            "-i",
            self.audio_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
        ]

        codec_candidates = ["libx264", "libopenh264", "h264", "mpeg4"]
        last_error = ""
        selected_codec: Optional[str] = None

        for codec in codec_candidates:
            if not _codec_supported(codec):
                continue
            cmd = base_cmd + [
                "-c:v",
                codec,
                "-pix_fmt",
                "yuv420p",
                "-g",
                str(max(int(self.fps), 1)),
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+frag_keyframe+empty_moov+default_base_moof",
                "-f",
                "mp4",
                self.video_path,
            ]

            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
            )

            # Give FFmpeg a moment to report codec errors before we proceed
            for _ in range(10):
                time.sleep(0.05)
                if proc.poll() is not None:
                    break

            if proc.poll() is None:
                self.proc = proc
                selected_codec = codec
                break

            try:
                _, stderr = proc.communicate(timeout=0.5)
            except subprocess.TimeoutExpired:
                proc.kill()
                _, stderr = proc.communicate()
            last_error = stderr.decode("utf-8", errors="ignore").strip()
            # keep last_error for raising later, but avoid noisy prints

        if not self.proc or not selected_codec:
            raise RuntimeError(f"Failed to start FFmpeg with available codecs. Last error: {last_error}")

        def _drain_stderr() -> None:
            assert self.proc and self.proc.stderr
            for line in iter(self.proc.stderr.readline, b""):
                text = line.decode("utf-8", errors="ignore").strip()
                if text:
                    self._stderr_buffer.append(text)
            # swallow stderr; available for debugging via self._stderr_buffer

        self._stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        self._stderr_thread.start()

    def __call__(self, img: np.ndarray, fmt: str = "bgr") -> None:
        if self.closed:
            return

        frame = img
        if fmt == "bgr":
            frame = frame[..., ::-1]

        if frame.dtype != np.uint8:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected frame with shape (H, W, 3)")

        frame = np.ascontiguousarray(frame)

        height, width = frame.shape[:2]
        if self._frame_shape is None:
            self._frame_shape = (height, width)
            self._start_ffmpeg(height, width)
        elif self._frame_shape != (height, width):
            raise ValueError(
                f"Frame shape changed from {self._frame_shape} to {(height, width)}"
            )

        if not self.proc or not self.proc.stdin or self.proc.stdin.closed:
            raise RuntimeError("FFmpeg process is not available")

        if self.proc.poll() is not None:
            self.closed = True
            stderr_tail = " | ".join(self._stderr_buffer[-5:]) if self._stderr_buffer else ""
            raise RuntimeError(f"FFmpeg exited early. Stderr: {stderr_tail}")

        try:
            self.proc.stdin.write(frame.tobytes())
            self.frame_count += 1
            # no periodic logs
        except BrokenPipeError as exc:
            self.closed = True
            raise RuntimeError("FFmpeg pipe closed unexpectedly") from exc

    def close(self) -> None:
        if self.closed:
            return

        try:
            if self.proc:
                if self.proc.stdin and not self.proc.stdin.closed:
                    self.proc.stdin.close()
                self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            if self.proc:
                self.proc.terminate()
        finally:
            if self._stderr_thread and self._stderr_thread.is_alive():
                self._stderr_thread.join(timeout=1)
            self.closed = True


class VideoWriterByImageIO:
    """Original video-only writer (kept for compatibility)"""

    def __init__(self, video_path, fps=25, **kwargs):
        video_format = kwargs.get("format", "mp4")
        codec = kwargs.get("vcodec", "libx264")
        quality = kwargs.get("quality")
        pixelformat = kwargs.get("pixelformat", "yuv420p")
        macro_block_size = kwargs.get("macro_block_size", 2)
        ffmpeg_params = ["-crf", str(kwargs.get("crf", 18))]

        os.makedirs(os.path.dirname(video_path), exist_ok=True)

        import imageio

        writer = imageio.get_writer(
            video_path,
            fps=fps,
            format=video_format,
            codec=codec,
            quality=quality,
            ffmpeg_params=ffmpeg_params,
            pixelformat=pixelformat,
            macro_block_size=macro_block_size,
        )
        self.writer = writer

    def __call__(self, img, fmt="bgr"):
        if fmt == "bgr":
            frame = img[..., ::-1]
        else:
            frame = img
        self.writer.append_data(frame)

    def close(self):
        self.writer.close()
