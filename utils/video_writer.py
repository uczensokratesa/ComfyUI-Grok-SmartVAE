"""
ltx_chunking_suite/utils/video_writer.py  v2
─────────────────────────────────────────────────────────────────────────────
StreamingVideoWriter and StreamingVideoConfig — async ffmpeg-pipe edition.

Architecture change vs v1
──────────────────────────
v1 used imageio.get_writer().append_data() which is synchronous:
    GPU decode → [stall] → imageio encode → [stall] → GPU decode → …

The stall causes two problems:
  1.  Memory spike: decoded tensor + imageio internal buffer coexist.
  2.  Throughput: GPU idles while CPU/disk finishes encoding.

v2 uses a direct ffmpeg stdin rawvideo pipe with a background writer thread:

    GPU decode ──→ Queue(maxsize=32) ──→ [writer thread] ──→ ffmpeg pipe ──→ disk

The Queue(maxsize=32) is the key: it decouples GPU and disk IO entirely.
  •  GPU fills the queue at full speed.
  •  Writer thread drains the queue asynchronously.
  •  Queue.put() blocks only when the queue is full — which provides natural
     backpressure if ffmpeg is slower than decode (rare for fast SSDs).
  •  Peak memory is bounded: at most 32 frames live in the queue at once.

Frame conversion
────────────────
  frame.mul(255.0).clamp_(0, 255).to(torch.uint8)
is done on the GPU (or CPU if already there) in-place, avoiding the extra
float32 copy that .byte() would create after .clamp(0,255).
Then .cpu().numpy() transfers to a contiguous uint8 numpy array.

For pinned-memory transfers (CUDA only), the caller can optionally pass
frames already on CPU with non_blocking=True — the writer accepts both
torch.Tensor and numpy arrays.

Codec support
─────────────
All four codecs (h264, h265, prores, ffv1) are piped through ffmpeg.
ProRes and FFV1 use rawvideo → ffmpeg with appropriate codec flags.
The ffmpeg binary is resolved via imageio_ffmpeg if available, else PATH.
"""

import json
import logging
import os
import queue
import subprocess
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

class StreamingVideoConfig:
    """
    Codec definitions for direct ffmpeg rawvideo-pipe encoding.

    output_params are appended to the ffmpeg command AFTER the codec flag.
    pixel_format is the OUTPUT pixel format (not the pipe input which is
    always rgb24).
    """
    CODECS = {
        "h264": {
            "codec":         "libx264",
            "ext":           "mp4",
            "output_params": ["-crf", "23", "-preset", "fast", "-movflags", "+faststart"],
            "pixel_format":  "yuv420p",
            "description":   "H.264 (best compatibility, fast encode)",
        },
        "h265": {
            "codec":         "libx265",
            "ext":           "mp4",
            "output_params": ["-crf", "28", "-preset", "fast", "-movflags", "+faststart"],
            "pixel_format":  "yuv420p",
            "description":   "H.265/HEVC (smaller files, slightly slower)",
        },
        "prores": {
            "codec":         "prores_ks",
            "ext":           "mov",
            "output_params": ["-profile:v", "3"],
            "pixel_format":  "yuv422p10le",
            "description":   "ProRes 422 (professional, large files)",
        },
        "ffv1": {
            "codec":         "ffv1",
            "ext":           "mkv",
            "output_params": ["-level", "3", "-g", "1", "-slices", "4", "-slicecrc", "1"],
            "pixel_format":  "yuv444p",
            "description":   "FFV1 (lossless archival)",
        },
    }

    PREVIEW_INTERVAL      = 50    # generate a thumbnail every N frames
    METADATA_SAVE_INTERVAL = 100  # save JSON checkpoint every N frames
    QUEUE_SIZE            = 32    # max frames buffered between decode and write


# ─────────────────────────────────────────────────────────────────────────────
#  Writer
# ─────────────────────────────────────────────────────────────────────────────

class StreamingVideoWriter:
    """
    Async frame-by-frame video encoder backed by a direct ffmpeg stdin pipe.

    Usage
    ─────
        writer = StreamingVideoWriter(path, fps, codec, width, height)
        writer.set_decode_params(time_scale, total_latent_frames)  # for resume
        for frame in frames:
            preview = writer.write_frame(frame)   # non-blocking (queue)
        final_path = writer.finalize(audio_path)  # blocks until ffmpeg done

    write_frame() is effectively non-blocking: it puts the frame numpy array
    into a Queue and returns immediately.  The writer thread dequeues and pipes
    raw bytes to ffmpeg in the background.

    Memory bound: at most QUEUE_SIZE (32) uint8 frames live in the queue.
    For 1080p that is 32 × 1920 × 1080 × 3 ≈ 190 MB — negligible.
    """

    _SENTINEL = None   # poison pill that tells the writer thread to stop

    def __init__(
        self,
        output_path: str,
        fps:         int,
        codec:       str,
        width:       int,
        height:      int,
        resume:      bool = False,
    ):
        self.output_path  = output_path
        self.fps          = fps
        self.codec        = codec
        self.width        = width
        self.height       = height
        self.resume       = resume

        self.metadata_path = output_path + ".metadata.json"
        base, ext          = os.path.splitext(output_path)
        self.temp_path     = f"{base}.tmp{ext}" if ext else output_path + ".tmp"

        self.frames_written         = 0
        self.session_frames_written = 0
        self.last_preview: Optional[np.ndarray] = None
        self.resume_prefix_path: Optional[str]  = None

        # Decode params stored for resume metadata
        self._time_scale:          Optional[int] = None
        self._total_latent_frames: Optional[int] = None

        # Async IO
        self._frame_queue  = queue.Queue(maxsize=StreamingVideoConfig.QUEUE_SIZE)
        self._error_queue  = queue.Queue()
        self._process:      Optional[subprocess.Popen] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._lock          = threading.Lock()
        self._is_running    = False

        if resume and os.path.exists(self.metadata_path):
            self._load_metadata()

        self._start()

    # ── Public API ────────────────────────────────────────────────────

    def set_decode_params(self, time_scale: int, total_latent_frames: int) -> None:
        """
        Store decode parameters so metadata captures them for safe resume.
        Must be called before the first write_frame().
        """
        self._time_scale          = time_scale
        self._total_latent_frames = total_latent_frames

    def write_frame(self, frame) -> Optional[np.ndarray]:
        """
        Queue one frame for async writing.

        Accepts a torch.Tensor [H, W, 3] float32 in [0, 1]  OR  a uint8
        numpy array [H, W, 3].  Conversion to uint8 happens here on the
        calling thread (GPU→CPU for tensors) so the writer thread only
        handles raw bytes.

        Returns a downscaled preview numpy array every PREVIEW_INTERVAL
        frames, None otherwise.

        Blocks only if the queue is full (backpressure).
        """
        # Check writer health before queuing
        if not self._error_queue.empty():
            raise RuntimeError(f"Writer error: {self._error_queue.get()}")

        # Convert to uint8 numpy — do this on the caller's thread to keep
        # the writer thread lean (pure bytes → pipe, no tensor ops)
        if hasattr(frame, "mul"):   # torch.Tensor
            frame_np = (
                frame.mul(255.0)
                     .clamp_(0, 255)
                     .to(dtype=_UINT8, non_blocking=False)
                     .cpu()
                     .numpy()
            )
        else:
            frame_np = np.asarray(frame, dtype=np.uint8)

        # Ensure C-contiguous for tobytes()
        if not frame_np.flags["C_CONTIGUOUS"]:
            frame_np = np.ascontiguousarray(frame_np)

        # Queue (blocks only when full — natural backpressure)
        try:
            self._frame_queue.put(frame_np, timeout=30.0)
        except queue.Full:
            raise RuntimeError(
                "Writer queue full after 30 s — ffmpeg may have stalled."
            )

        # Update counter under lock (writer thread also touches frames_written)
        with self._lock:
            self.frames_written         += 1
            self.session_frames_written += 1
            n = self.frames_written

        # Periodic preview thumbnail (cheap: resize one frame)
        preview = None
        if n % StreamingVideoConfig.PREVIEW_INTERVAL == 0:
            scale = min(1.0, 512.0 / max(self.width, self.height))
            if scale < 1.0 and CV2_AVAILABLE:
                nw = int(self.width  * scale)
                nh = int(self.height * scale)
                preview = cv2.resize(frame_np, (nw, nh), interpolation=cv2.INTER_AREA)
            else:
                preview = frame_np.copy()
            self.last_preview = preview

        # Periodic metadata checkpoint
        if n % StreamingVideoConfig.METADATA_SAVE_INTERVAL == 0:
            self._save_metadata()

        return preview

    def finalize(self, audio_path: Optional[str] = None) -> str:
        """
        Flush the queue, wait for ffmpeg to finish, rename temp → output,
        optionally mux audio.  Returns the final video path.
        """
        self._stop()

        if not os.path.exists(self.temp_path):
            logger.error(f"Temp file missing after encode: {self.temp_path}")
            return self.output_path

        # Merge resume prefix if present
        base_video = self._merge_resume_prefix()

        # audio mux
        if audio_path and os.path.exists(audio_path):
            return self._mux_audio(base_video, audio_path)

        return base_video

    # ── Internal: start / stop ────────────────────────────────────────

    @staticmethod
    def _get_ffmpeg() -> str:
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            return "ffmpeg"

    def _start(self) -> None:
        cfg = StreamingVideoConfig.CODECS.get(
            self.codec, StreamingVideoConfig.CODECS["h264"]
        )

        # Handle resume prefix — same logic as v1
        if self.resume and self.frames_written > 0:
            if os.path.exists(self.output_path):
                self.resume_prefix_path = self.output_path
            elif os.path.exists(self.temp_path):
                base, ext  = os.path.splitext(self.temp_path)
                prefix_bak = f"{base}.resume_prefix{ext}" if ext else self.temp_path + ".resume_prefix"
                try:
                    os.replace(self.temp_path, prefix_bak)
                    self.resume_prefix_path = prefix_bak
                except Exception as e:
                    logger.warning(f"Resume prefix backup failed: {e}; restarting from 0")
                    self.frames_written = 0
            else:
                logger.warning("Resume metadata found but no video on disk — restarting")
                self.frames_written = 0

        # Build ffmpeg command
        # Input: rawvideo rgb24 via stdin pipe
        # Output: codec-specific container
        cmd = [
            self._get_ffmpeg(), "-y",
            # Input specification
            "-f",        "rawvideo",
            "-vcodec",   "rawvideo",
            "-pix_fmt",  "rgb24",
            "-s",        f"{self.width}x{self.height}",
            "-framerate", str(self.fps),
            "-i",        "-",                 # stdin
            # Output
            "-c:v",      cfg["codec"],
            "-pix_fmt",  cfg["pixel_format"],
            # Colour metadata (ensures players don't guess wrong)
            "-vf",       "scale=out_color_matrix=bt709",
            "-color_range", "tv",
            "-colorspace",  "bt709",
            "-color_primaries", "bt709",
            "-color_trc",   "bt709",
        ] + cfg["output_params"] + [self.temp_path]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdin  = subprocess.PIPE,
                stderr = subprocess.PIPE,
                bufsize = 10 * 1024 * 1024,   # 10 MB write buffer
            )
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found.  Install ffmpeg and ensure it is on PATH."
            )

        self._is_running = True

        # Background writer thread
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="ltx-video-writer"
        )
        self._writer_thread.start()

        # Background stderr monitor (logs ffmpeg errors without blocking)
        threading.Thread(
            target=self._stderr_monitor, daemon=True, name="ltx-stderr"
        ).start()

        logger.info(
            f"ffmpeg pipe open: {cfg['description']}  "
            f"{self.width}×{self.height} @ {self.fps} fps → {self.temp_path}"
        )

    def _stop(self) -> None:
        """Drain queue, send sentinel, wait for writer thread and ffmpeg."""
        if not self._is_running:
            return

        # Wait for all queued frames to be dequeued
        self._frame_queue.join()

        # Send sentinel to stop writer thread
        self._frame_queue.put(self._SENTINEL)
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=30)

        self._is_running = False

        # Close stdin so ffmpeg knows the stream is over
        if self._process and self._process.stdin:
            try:
                self._process.stdin.close()
            except Exception:
                pass

        # Wait for ffmpeg to finish encoding
        if self._process:
            try:
                self._process.wait(timeout=120)
                if self._process.returncode not in (0, None):
                    logger.warning(
                        f"ffmpeg exited with code {self._process.returncode}"
                    )
            except subprocess.TimeoutExpired:
                logger.error("ffmpeg timed out — killing")
                self._process.kill()
                self._process.wait()

    def _writer_loop(self) -> None:
        """
        Background thread: dequeues frames and writes raw bytes to ffmpeg stdin.

        Writing raw bytes (frame_np.tobytes()) is the fastest possible path:
        no Python-level encoding, no copy — just a single write() syscall per frame.
        """
        while True:
            try:
                frame_np = self._frame_queue.get(timeout=2.0)
            except queue.Empty:
                # Check if we should keep waiting
                if not self._is_running and self._frame_queue.empty():
                    break
                continue

            if frame_np is self._SENTINEL:
                self._frame_queue.task_done()
                break

            if self._process and self._process.poll() is None:
                try:
                    self._process.stdin.write(frame_np.tobytes())
                except (BrokenPipeError, OSError) as e:
                    self._error_queue.put(f"ffmpeg pipe broken: {e}")
                    self._is_running = False
                    self._frame_queue.task_done()
                    break
            else:
                self._error_queue.put("ffmpeg process died unexpectedly")
                self._is_running = False
                self._frame_queue.task_done()
                break

            self._frame_queue.task_done()

    def _stderr_monitor(self) -> None:
        """Log ffmpeg stderr without blocking the main pipeline."""
        if not self._process:
            return
        try:
            for line in iter(self._process.stderr.readline, b""):
                text = line.decode("utf-8", errors="replace").strip()
                if text and any(k in text.lower() for k in ("error", "warning", "invalid")):
                    logger.debug(f"[ffmpeg] {text}")
        except Exception:
            pass

    # ── Internal: resume / metadata ───────────────────────────────────

    def _save_metadata(self) -> None:
        with self._lock:
            n = self.frames_written
        metadata = {
            "frames_written":      n,
            "output_path":         self.output_path,
            "temp_path":           self.temp_path,
            "fps":                 self.fps,
            "codec":               self.codec,
            "resolution":          [self.width, self.height],
            "timestamp":           time.time(),
            "time_scale":          self._time_scale,
            "total_latent_frames": self._total_latent_frames,
        }
        try:
            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.debug(f"Metadata save failed: {e}")

    def _load_metadata(self) -> None:
        try:
            with open(self.metadata_path) as f:
                meta = json.load(f)
            self.frames_written = meta.get("frames_written", 0)
            logger.info(f"Resume: checkpoint at frame {self.frames_written}")
        except Exception as e:
            logger.warning(f"Metadata load failed: {e}")
            self.frames_written = 0

    def _merge_resume_prefix(self) -> str:
        """Merge the resume prefix video with the new segment if needed."""
        temp_exists = os.path.exists(self.temp_path)

        if self.resume_prefix_path and os.path.exists(self.resume_prefix_path):
            if self.session_frames_written > 0 and temp_exists:
                merged = self._concat_videos(self.resume_prefix_path, self.temp_path)
                if merged:
                    try:
                        os.remove(self.temp_path)
                    except Exception:
                        pass
                    return merged
                return self.resume_prefix_path
            else:
                if temp_exists:
                    try:
                        os.remove(self.temp_path)
                    except Exception:
                        pass
                return self.resume_prefix_path

        if temp_exists:
            try:
                os.replace(self.temp_path, self.output_path)
                return self.output_path
            except Exception as e:
                logger.error(f"Failed to rename temp file: {e}")
                return self.temp_path

        return self.output_path

    # ── Internal: ffmpeg helpers ──────────────────────────────────────

    @staticmethod
    def _escape_concat_path(p: str) -> str:
        return p.replace("'", "'\\''")

    def _concat_videos(self, first: str, second: str) -> Optional[str]:
        if not (os.path.exists(first) and os.path.exists(second)):
            return None
        ffmpeg = self._get_ffmpeg()
        base, ext   = os.path.splitext(self.output_path)
        concat_out  = f"{base}.concat{ext}"
        concat_list = f"{base}.concat.txt"
        try:
            with open(concat_list, "w", encoding="utf-8") as f:
                f.write(f"file '{self._escape_concat_path(first)}'\n")
                f.write(f"file '{self._escape_concat_path(second)}'\n")
            subprocess.run(
                [ffmpeg, "-y", "-f", "concat", "-safe", "0",
                 "-i", concat_list, "-c", "copy", concat_out],
                check=True, capture_output=True, timeout=600,
            )
            if os.path.exists(concat_out):
                os.replace(concat_out, self.output_path)
                return self.output_path
        except Exception as e:
            logger.error(f"Concat failed: {e}")
        finally:
            for p in (concat_out, concat_list):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
        return None

    def _mux_audio(self, video_path: str, audio_path: str) -> str:
        if not os.path.exists(video_path):
            return video_path
        ffmpeg   = self._get_ffmpeg()
        cfg      = StreamingVideoConfig.CODECS.get(self.codec, StreamingVideoConfig.CODECS["h264"])
        base, ext = os.path.splitext(self.output_path)
        out       = f"{base}_audio{ext}"
        cmd = (
            [ffmpeg, "-y", "-i", video_path, "-i", audio_path,
             "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy",              # video already encoded — just copy stream
             "-c:a", "aac", "-b:a", "192k",
             "-shortest", out]
        )
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            if os.path.exists(out):
                if video_path != self.output_path and os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                    except Exception:
                        pass
                logger.info(f"Audio muxed → {out}")
                return out
        except Exception as e:
            logger.error(f"Audio mux failed: {e}")
        return video_path

    # ── Cleanup ───────────────────────────────────────────────────────

    def __del__(self):
        try:
            if self._is_running:
                self._stop()
        except Exception:
            pass


# ── lazy import of torch.uint8 alias to avoid import-time torch dependency ──
try:
    import torch as _torch
    _UINT8 = _torch.uint8
except ImportError:
    _UINT8 = None   # write_frame handles this path via numpy asarray
