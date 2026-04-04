"""
ltx_chunking_suite/ltx_video_decoder.py
─────────────────────────────────────────────────────────────────────────────
LTX Video Decoder — one node to decode LTX video latents to MP4.

Replaces both SmartVAE_StreamingDecoder and SmartVAEAdvancedDecoder with a
single node whose advanced features (colour correction, calibration image)
are optional and default to off.

Design principles
─────────────────
• Single decode loop — no subclassing, no code duplication between "base"
  and "advanced".  All fixes (resume drift guard, safety margin, break on
  empty chunk) are present in one place.
• Colour correction is injected naturally inside the loop when enabled;
  it is a zero-cost no-op when disabled.
• All safety features from the original node are preserved:
    - VRAM auto-reduction for long sequences
    - CPU fallback for mega-sequences (>2500 latent frames)
    - OOM retry with tiling → batch halving → tile reduction
    - Crash-resume with metadata checkpoint
    - FPS float→int rounding (for 23.976, 29.97, etc.)
    - ignore_warnings mode
    - AUDIO dict input + audio_path string input

Dependencies
────────────
    pip install imageio imageio-ffmpeg numpy psutil
    (cv2 optional — used only for preview thumbnail generation)
"""

import gc
import logging
import os
import subprocess
import time
import warnings
from typing import Optional

import torch
import comfy.utils
import folder_paths
from comfy.model_management import throw_exception_if_processing_interrupted

from .utils import (
    StreamingVideoWriter,     # v2: async ffmpeg-pipe + Queue (no imageio)
    StreamingVideoConfig,
    extract_image_stats,
    apply_temporal_color_match,
)

logger = logging.getLogger(__name__)

# ── Optional dependencies ──────────────────────────────────────────────────
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("numpy not available — required for LTX Video Decoder")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torchaudio        # noqa: F401  (used indirectly for audio tensor ops)
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False


# ═════════════════════════════════════════════════════════════════════════════
#  LTX Video Decoder
# ═════════════════════════════════════════════════════════════════════════════

class LTX_VideoDecoder:
    """
    Streaming VAE decode for LTX 2.3 video latents.

    Decodes in small batches (frames_per_batch) to stay within VRAM limits,
    writing directly to disk frame-by-frame rather than building a full
    [T, H, W, 3] tensor in memory.

    Colour correction (anti_color_bleed)
    ─────────────────────────────────────
    LTX's streaming decode can produce visible colour steps at batch
    boundaries because each batch is decoded independently and the VAE's
    internal statistics reset between calls.  With anti_color_bleed=True,
    a lightweight Reinhard-style correction is applied to the first
    cross_fade_frames frames of each batch, anchoring them to the colour
    statistics of the previous batch's tail.  An optional calibration_image
    seeds the initial target so the whole video is graded consistently.

    Resume on crash
    ───────────────
    When resume_on_crash=True, the node saves a JSON metadata checkpoint
    every METADATA_SAVE_INTERVAL frames.  If the run is interrupted, the
    next execution picks up from that checkpoint.  The resume index
    calculation includes a 2-frame safety margin (re-decodes up to 16
    pixel frames) to avoid corrupted tail frames, and a stitched-latent
    drift guard for latents assembled by LTX_Latent_Stitcher.
    """

    MAX_OOM_RETRIES = 5

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae":     ("VAE",),
                "frames_per_batch": ("INT", {
                    "default": 8, "min": 1, "max": 128, "step": 1,
                    "tooltip": "Latent frames per decode call. Auto-reduces on OOM.",
                }),
            },
            "optional": {
                # ── Streaming / quality ───────────────────────────────
                "overlap_frames": ("INT", {
                    "default": 2, "min": 0, "max": 16, "step": 1,
                    "tooltip": "Temporal overlap between batches for seamless stitching.",
                }),
                "force_time_scale": ("INT", {
                    "default": 0, "min": 0, "max": 16, "step": 1,
                    "tooltip": "Override auto-detected time scale (0=auto, 8=LTX).",
                }),
                "enable_tiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force spatial tiling. Auto-enables on OOM.",
                }),
                "tile_size": ("INT", {
                    "default": 512, "min": 256, "max": 2048, "step": 64,
                }),
                # ── Output ────────────────────────────────────────────
                "video_output_path": ("STRING", {
                    "default": "",
                    "tooltip": "Full output path. Blank = auto-named in ComfyUI output dir.",
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0,
                    "tooltip": "Frame rate. Supports fractional values (23.976, 29.97 …).",
                }),
                "codec": (list(StreamingVideoConfig.CODECS.keys()), {
                    "default": "h264",
                    "tooltip": "h264=compatible | h265=smaller | prores/ffv1=professional",
                }),
                # ── Audio ─────────────────────────────────────────────
                "audio": ("AUDIO", {
                    "tooltip": "ComfyUI AUDIO dict. Takes priority over audio_path.",
                }),
                "audio_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to an audio file to mux into the output.",
                }),
                # ── Colour correction ─────────────────────────────────
                "anti_color_bleed": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Temporal colour matching at batch boundaries.  "
                        "Eliminates visible colour steps in streaming decode.  "
                        "Safe to enable — negligible performance cost."
                    ),
                }),
                "calibration_image": ("IMAGE", {
                    "tooltip": (
                        "Optional reference image that seeds the colour grade.  "
                        "When supplied, the whole video is anchored to its statistics."
                    ),
                }),
                "correction_strength": ("FLOAT", {
                    "default": 0.18, "min": 0.0, "max": 0.4, "step": 0.02,
                    "tooltip": "Blend weight of the colour correction (0=off, 0.18=default).",
                }),
                "ema_momentum": ("FLOAT", {
                    "default": 0.93, "min": 0.7, "max": 0.98, "step": 0.01,
                    "tooltip": "EMA decay for the running colour reference (higher=slower).",
                }),
                "cross_fade_frames": ("INT", {
                    "default": 12, "min": 4, "max": 24, "step": 2,
                    "tooltip": "Number of frames at each batch boundary to colour-correct.",
                }),
                # ── Recovery / safety ─────────────────────────────────
                "resume_on_crash": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Resume from last checkpoint if a previous run crashed.",
                }),
                "ignore_warnings": (["none", "minor", "all"], {
                    "default": "none",
                    "tooltip": (
                        "none  = abort on NaN/corruption  |  "
                        "minor = continue if <10% corrupt  |  "
                        "all   = force decode (high risk of black frames)"
                    ),
                }),
                "verbose": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES  = ("IMAGE", "STRING")
    RETURN_NAMES  = ("preview_thumbs", "video_path")
    FUNCTION      = "decode"
    OUTPUT_NODE   = True
    CATEGORY      = "RTX Director Suite / LTX"

    def __init__(self):
        self._time_scale_cache  = {}
        self._force_scale_cache = {}
        self._verbose           = False

    # ── Public entry point ────────────────────────────────────────────

    def decode(
        self,
        samples:            dict,
        vae,
        frames_per_batch:   int,
        overlap_frames:     int   = 2,
        force_time_scale:   int   = 0,
        enable_tiling:      bool  = False,
        tile_size:          int   = 512,
        video_output_path:  str   = "",
        fps:                float = 24.0,
        codec:              str   = "h264",
        audio                     = None,
        audio_path:         str   = "",
        anti_color_bleed:   bool  = False,
        calibration_image         = None,
        correction_strength:float = 0.18,
        ema_momentum:       float = 0.93,
        cross_fade_frames:  int   = 12,
        resume_on_crash:    bool  = True,
        ignore_warnings:    str   = "none",
        verbose:            bool  = False,
    ):
        self._verbose = verbose

        if not NUMPY_AVAILABLE:
            raise RuntimeError(
                "numpy is required.\n"
                "Install: pip install imageio imageio-ffmpeg numpy"
            )

        # ── FPS normalisation (handles 23.976, 29.97, …) ──────────────
        final_fps = int(round(fps))
        if verbose and abs(fps - final_fps) > 0.01:
            logger.info(f"🎬 fps {fps:.3f} → rounded to {final_fps}")
        if not 1 <= final_fps <= 240:
            raise ValueError(f"fps out of range: {final_fps}")

        # ── Latent unpacking ──────────────────────────────────────────
        latents = samples["samples"]
        if latents.dim() == 4:
            raise ValueError(
                "5D tensor expected (video latent). "
                "For images use the standard VAE Decode node."
            )

        _, latent_ch, total_frames, h_lat, w_lat = latents.shape
        if total_frames <= 0:
            raise ValueError("Latent has no frames.")

        # ── NaN / Inf validation ───────────────────────────────────────
        if torch.isnan(latents).any() or torch.isinf(latents).any():
            nan_pct = (torch.isnan(latents).sum().item() / latents.numel()) * 100
            if ignore_warnings == "none":
                raise ValueError(
                    f"Latent contains {nan_pct:.1f}% NaN/Inf. "
                    f"Set ignore_warnings='minor' or 'all' to proceed."
                )
            if ignore_warnings == "minor" and nan_pct > 10.0:
                raise ValueError(f"Too corrupt to continue: {nan_pct:.1f}% NaN.")
            logger.warning(f"⚠️  Continuing with {nan_pct:.1f}% NaN (cleaned to 0)")
            latents = torch.nan_to_num(latents, nan=0.0, posinf=1.0, neginf=0.0)

        # ── Time scale detection ──────────────────────────────────────
        time_scale     = self._detect_time_scale(vae, latents, force_time_scale, verbose)
        expected_frames = 1 + (total_frames - 1) * time_scale
        aspect_ratio   = h_lat / float(w_lat)

        if verbose:
            logger.info(
                f"Latent: {total_frames} frames × time_scale={time_scale} "
                f"→ ~{expected_frames} px frames"
            )

        # ── Output size detection ─────────────────────────────────────
        output_h, output_w = self._detect_output_size(vae, latents, h_lat, w_lat, tile_size)

        # ── Long-sequence safety ──────────────────────────────────────
        if total_frames > 1000:
            safe = 4 if total_frames > 2000 else (6 if total_frames > 1500 else 8)
            if frames_per_batch > safe:
                if verbose:
                    logger.info(
                        f"🛡️  Long sequence ({total_frames} frames): "
                        f"batch {frames_per_batch} → {safe}"
                    )
                frames_per_batch = safe

        if total_frames > 2500 and latents.device.type == "cuda":
            logger.warning(
                f"🚨 Mega-sequence ({total_frames} frames) — "
                f"forcing CPU decode (will be slow)"
            )
            latents = latents.cpu()
            vae.first_stage_model.to("cpu")
            frames_per_batch = min(frames_per_batch, 4)
            gc.collect()
            torch.cuda.empty_cache()

        # ── Output path ───────────────────────────────────────────────
        if not video_output_path:
            ts       = int(time.time())
            ext      = StreamingVideoConfig.CODECS[codec]["ext"]
            video_output_path = os.path.join(
                folder_paths.get_output_directory(),
                f"ltx_video_{ts}.{ext}",
            )
        os.makedirs(os.path.dirname(os.path.abspath(video_output_path)), exist_ok=True)

        # ── Audio preparation ─────────────────────────────────────────
        resolved_audio_path = self._prepare_audio(audio, audio_path, folder_paths.get_temp_directory())

        # ── Writer ────────────────────────────────────────────────────
        writer = StreamingVideoWriter(
            video_output_path, final_fps, codec, output_w, output_h,
            resume=resume_on_crash,
        )
        writer.set_decode_params(time_scale, total_frames)

        # ── VRAM auto-reduction ───────────────────────────────────────
        frames_per_batch = max(1, min(frames_per_batch, total_frames))
        overlap_frames   = max(0, min(overlap_frames, frames_per_batch - 1))
        initial_overlap  = overlap_frames

        vram = self._get_available_vram()
        if vram is not None:
            est = self._estimate_chunk_vram(
                frames_per_batch + 2 * overlap_frames,
                latent_ch, h_lat, w_lat, time_scale,
            )
            if est > vram * 0.55:
                frames_per_batch = max(1, int(frames_per_batch * (vram * 0.45 / est)))
                overlap_frames   = min(initial_overlap, frames_per_batch - 1)
                if verbose:
                    logger.info(f"📉 VRAM reduction → batch={frames_per_batch}")

        # ── Resume index ──────────────────────────────────────────────
        current_batch = frames_per_batch
        start_idx = (
            self._resume_to_latent_index(
                writer.frames_written, time_scale, total_frames
            )
            if resume_on_crash else 0
        )
        if verbose and writer.frames_written > 0:
            logger.info(
                f"▶  Resuming: {writer.frames_written} px frames written → "
                f"latent start_idx={start_idx}"
            )

        # ── Colour correction state ───────────────────────────────────
        calib_stats = extract_image_stats(calibration_image) if calibration_image is not None else None
        ema_stats   = None
        prev_stats  = None

        # ── Decode loop ───────────────────────────────────────────────
        pbar            = comfy.utils.ProgressBar(total_frames)
        preview_frames  = []
        frames_processed = 0
        last_processed   = -1
        stagnation       = 0
        oom_retry        = 0

        while start_idx < total_frames:
            throw_exception_if_processing_interrupted()

            if oom_retry >= self.MAX_OOM_RETRIES:
                raise RuntimeError(f"Exceeded {self.MAX_OOM_RETRIES} OOM retries.")

            # Stagnation guard
            if frames_processed == last_processed:
                stagnation += 1
                if stagnation >= 3:
                    raise RuntimeError(f"Decode stalled at latent frame {start_idx}.")
            else:
                stagnation = 0
            last_processed = frames_processed

            end_idx   = min(start_idx + current_batch, total_frames)
            ctx_start = max(0, start_idx - overlap_frames)
            ctx_end   = min(total_frames, end_idx + overlap_frames)

            # Empty chunk — normal at tail of stitched latent with drift guard
            if ctx_end <= ctx_start:
                break

            latent_chunk = latents[:, :, ctx_start:ctx_end, :, :]

            # Decode batch
            try:
                with torch.no_grad():
                    if enable_tiling and hasattr(vae, "decode_tiled"):
                        raw = vae.decode_tiled(latent_chunk, tile_x=tile_size, tile_y=tile_size)
                    else:
                        raw = vae.decode(latent_chunk)
                    decoded = self._extract_output(raw).cpu()
                oom_retry = 0

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom_retry += 1
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    if not enable_tiling:
                        enable_tiling = True
                        continue
                    if current_batch > 1:
                        current_batch = max(1, current_batch // 2)
                        overlap_frames = min(initial_overlap, current_batch - 1)
                        continue
                    if tile_size > 256:
                        tile_size = max(256, tile_size - 128)
                        continue
                    continue
                raise

            # Normalise and trim overlap
            decoded      = self._normalise(decoded, aspect_ratio)
            front_trim   = (start_idx - ctx_start) * time_scale

            if end_idx == total_frames:
                valid = decoded[front_trim:]
            else:
                core_len = (end_idx - start_idx) * time_scale
                valid    = decoded[front_trim: front_trim + core_len]

            valid = self._center_crop(valid, output_h, output_w)

            # Colour correction (no-op when anti_color_bleed=False)
            if anti_color_bleed and valid.shape[0] > 0:
                valid, ema_stats, prev_stats = apply_temporal_color_match(
                    valid,
                    prev_stats       = prev_stats,
                    ema_stats        = ema_stats,
                    calibration_stats= calib_stats,
                    correction_strength = correction_strength,
                    ema_momentum     = ema_momentum,
                    fade_frames      = cross_fade_frames,
                )
            elif anti_color_bleed is False and valid.shape[0] > 0:
                # Track prev_stats even without correction — used on first-enabled batch
                from .utils.color_utils import extract_stats
                prev_stats = extract_stats(valid, cross_fade_frames)

            # Write frames
            for frame in valid:
                preview = writer.write_frame(frame)
                if preview is not None:
                    preview_frames.append(
                        torch.from_numpy(preview).float() / 255.0
                    )
                    if len(preview_frames) > 5:
                        preview_frames.pop(0)

            processed = end_idx - start_idx
            frames_processed += processed
            pbar.update(processed)

            if verbose:
                pct = writer.frames_written / max(expected_frames, 1) * 100
                logger.info(
                    f"  {writer.frames_written}/{expected_frames} frames "
                    f"({pct:.1f}%)"
                )

            start_idx = end_idx

            del latent_chunk, decoded, valid
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── Finalise ──────────────────────────────────────────────────
        final_path = writer.finalize(resolved_audio_path or audio_path or None)

        if ignore_warnings != "none":
            logger.warning(
                f"⚠️  Video produced with ignore_warnings='{ignore_warnings}'. "
                f"Some frames may be black or corrupt."
            )

        if verbose:
            size_mb = os.path.getsize(final_path) / 1_048_576
            logger.info(
                f"✅ Done → {final_path}  "
                f"({writer.frames_written} frames, {size_mb:.1f} MB)"
            )

        logger.info(
            "📸 preview_thumbs: last 5 thumbnails only — full video on disk."
        )

        preview_tensor = (
            torch.stack(preview_frames) if preview_frames
            else torch.empty(0, dtype=torch.float32)
        )
        return (preview_tensor, final_path)

    # ── Time scale detection ──────────────────────────────────────────

    def _detect_time_scale(self, vae, latents: torch.Tensor,
                            force: int, verbose: bool) -> int:
        vae_id = id(vae)

        if force > 0:
            self._force_scale_cache[vae_id] = force
            self._time_scale_cache[vae_id]  = force
            if verbose:
                logger.info(f"🔧 time_scale forced: {force}×")
            return force

        self._force_scale_cache.pop(vae_id, None)

        if vae_id in self._time_scale_cache:
            return self._time_scale_cache[vae_id]

        # Try VAE metadata
        if hasattr(vae, "downscale_index_formula") and vae.downscale_index_formula:
            try:
                ts = int(vae.downscale_index_formula[0])
                self._time_scale_cache[vae_id] = ts
                if verbose:
                    logger.info(f"🔍 time_scale from VAE metadata: {ts}×")
                return ts
            except Exception:
                pass

        # Probe decode
        try:
            n_test = min(5, latents.shape[2])
            if n_test <= 1:
                self._time_scale_cache[vae_id] = 1
                return 1

            probe = latents[:, :, :n_test, :16, :16]
            with torch.no_grad():
                out = vae.decode(probe)
            out = self._normalise(out, aspect_ratio=1.0)
            ts  = max(1, (out.shape[0] - 1) // (n_test - 1))

            del out, probe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._time_scale_cache[vae_id] = ts
            if verbose:
                logger.info(f"🔍 time_scale auto-detected: {ts}×")
            return ts

        except Exception as e:
            if verbose:
                logger.warning(f"time_scale detection failed ({e}), fallback=1")
            self._time_scale_cache[vae_id] = 1
            return 1

    # ── Output size detection ─────────────────────────────────────────

    def _detect_output_size(self, vae, latents, h_lat, w_lat, tile_size):
        probe = latents[:, :, :1, :, :]
        try:
            with torch.no_grad():
                if hasattr(vae, "decode_tiled"):
                    out = vae.decode_tiled(probe, tile_x=tile_size, tile_y=tile_size)
                else:
                    out = vae.decode(probe)
            out = self._normalise(out, aspect_ratio=None)
            return out.shape[1], out.shape[2]
        except Exception:
            return h_lat * 8, w_lat * 8

    # ── Resume index (drift-safe) ─────────────────────────────────────

    @staticmethod
    def _resume_to_latent_index(
        output_frames: int,
        time_scale:    int,
        total_frames:  int,
        safety_margin: int = 2,
    ) -> int:
        """
        Convert saved frame count back to a latent start index.

        safety_margin backs up by 2 latent frames so any corrupt tail
        frames are cleanly re-decoded.

        drift guard: stitched latents may produce fewer pixel frames than
        the formula `1 + (T-1)*S` predicts.  When output_frames already
        meets or exceeds the formula's expected count, we return
        total_frames-1 (force one last batch) rather than total_frames
        (which would silently skip the tail).
        """
        if output_frames <= 0:
            return 0
        if time_scale <= 1:
            return min(output_frames, total_frames)

        formula_expected = 1 + (total_frames - 1) * time_scale
        if output_frames >= formula_expected:
            # Drift guard for stitched latents
            return max(0, total_frames - 1)

        idx = 1 + max(0, output_frames - 1) // time_scale
        return min(max(0, idx - safety_margin), total_frames)

    # ── Tensor helpers ────────────────────────────────────────────────

    @staticmethod
    def _extract_output(tensor):
        if isinstance(tensor, (list, tuple)):
            if not tensor:
                raise ValueError("VAE returned empty output")
            tensor = tensor[0]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected Tensor, got {type(tensor)}")
        return tensor

    @staticmethod
    def _normalise(tensor: torch.Tensor,
                   aspect_ratio: Optional[float]) -> torch.Tensor:
        if isinstance(tensor, (list, tuple)):
            tensor = tensor[0]

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected Tensor, got {type(tensor)}")
        if tensor.dtype != torch.float32:
            tensor = tensor.float()

        dim = tensor.dim()

        if dim == 4:
            if tensor.shape[1] in (3, 4):
                tensor = tensor.permute(0, 2, 3, 1)

        elif dim == 5:
            shape = list(tensor.shape)
            if shape[0] == 1:
                tensor = tensor.squeeze(0)
                shape  = list(tensor.shape)

            try:
                c_idx = next(i for i, s in enumerate(shape) if s in (3, 4))
            except StopIteration:
                raise ValueError(f"Cannot find channel dim in {shape}")

            others  = [i for i in range(4) if i != c_idx]
            by_size = sorted(zip(others, [shape[i] for i in others]), key=lambda x: x[1])
            f_idx   = by_size[0][0]
            s_big   = by_size[2][0]
            s_sml   = by_size[1][0]

            if aspect_ratio is not None:
                h_idx = s_big if aspect_ratio > 1.0 else s_sml
                w_idx = s_sml if aspect_ratio > 1.0 else s_big
            else:
                h_idx, w_idx = s_big, s_sml

            tensor = tensor.permute(f_idx, h_idx, w_idx, c_idx)

        else:
            raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

        if tensor.min().item() < 0:
            tensor = (tensor + 1.0) / 2.0

        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)
        tensor = torch.clamp(tensor, 0.0, 1.0)

        if tensor.shape[-1] > 3:
            tensor = tensor[..., :3]

        return tensor.contiguous()

    @staticmethod
    def _center_crop(tensor: torch.Tensor, h_ref: int, w_ref: int) -> torch.Tensor:
        _, h, w, _ = tensor.shape
        if h == h_ref and w == w_ref:
            return tensor
        ho = max(0, (h - h_ref) // 2)
        wo = max(0, (w - w_ref) // 2)
        return tensor[:, ho: ho + h_ref, wo: wo + w_ref, :]

    # ── VRAM / RAM estimators ─────────────────────────────────────────

    @staticmethod
    def _get_available_vram() -> Optional[float]:
        try:
            if not torch.cuda.is_available():
                return None
            free, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
            return free / (1024 ** 3)
        except Exception:
            return None

    @staticmethod
    def _estimate_chunk_vram(frames: int, channels: int,
                              h: int, w: int, time_scale: int) -> float:
        lat_b  = frames * channels * h * w * 4
        out_b  = frames * time_scale * 3 * (h * 8) * (w * 8) * 4
        return (lat_b + out_b) * 3.5 * 1.1 / (1024 ** 3)

    # ── Audio helper ──────────────────────────────────────────────────

    @staticmethod
    @staticmethod
    def _prepare_audio(audio_dict, audio_path_str: str, tmp_dir: str) -> str:
        """
        Convert a ComfyUI AUDIO dict to a temporary WAV file path.

        Uses torchaudio.save() which handles the planar→interleaved conversion
        correctly and supports all sample rates and channel counts natively.
        Falls back to the ffmpeg s16le pipe approach when torchaudio is absent.

        Returns a WAV file path, "" if no audio was provided, or audio_path_str
        if conversion fails.
        """
        if audio_dict is None:
            return ""

        if not isinstance(audio_dict, dict) or "waveform" not in audio_dict:
            logger.warning("Invalid AUDIO input format — ignoring")
            return audio_path_str

        waveform    = audio_dict["waveform"]
        sample_rate = int(audio_dict.get("sample_rate", 44100))

        if waveform is None or (hasattr(waveform, "numel") and waveform.numel() == 0):
            logger.warning("AUDIO waveform is empty — ignoring")
            return audio_path_str

        # Normalise to (channels, samples) float32 in [-1, 1]
        waveform = waveform.cpu().float()
        if waveform.dim() == 1:      # (samples,) → (1, samples)
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 3:    # (1, channels, samples) → (channels, samples)
            waveform = waveform.squeeze(0)
        # dim == 2: already (channels, samples)

        # Clamp to valid range
        waveform = waveform.clamp_(-1.0, 1.0)

        ts       = int(time.time() * 1000)
        tmp_path = os.path.join(tmp_dir, f"ltx_audio_{ts}.wav")

        # ── Path 1: torchaudio (preferred — correct interleaving, no subprocess) ──
        try:
            import torchaudio
            torchaudio.save(tmp_path, waveform, sample_rate)
            logger.info(
                f"Audio saved via torchaudio: {waveform.shape} @ {sample_rate} Hz"
            )
            return tmp_path
        except ImportError:
            pass   # fall through to ffmpeg path
        except Exception as e:
            logger.warning(f"torchaudio.save failed: {e} — trying ffmpeg fallback")

        # ── Path 2: ffmpeg s16le pipe fallback (interleaved, corrected) ──
        # waveform is (C, S) planar. ffmpeg -f s16le expects interleaved (S, C).
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg = "ffmpeg"

        n_channels = waveform.shape[0]
        wav_np     = (waveform.numpy() * 32767).astype("int16")   # (C, S)
        if n_channels > 1:
            wav_np = wav_np.T   # (C, S) → (S, C) interleaved
        # .tobytes() on C-contiguous (S, C) array = correct interleaved PCM

        try:
            result = subprocess.run(
                [ffmpeg, "-y",
                 "-f",   "s16le",
                 "-ar",  str(sample_rate),
                 "-ac",  str(n_channels),
                 "-i",   "pipe:0",
                 "-c:a", "pcm_s16le", tmp_path],
                input           = wav_np.tobytes(),
                capture_output  = True,
                timeout         = 120,
            )
            if result.returncode == 0 and os.path.exists(tmp_path):
                logger.info(f"Audio saved via ffmpeg fallback: {n_channels}ch @ {sample_rate} Hz")
                return tmp_path
            logger.warning(f"ffmpeg audio fallback failed: {result.stderr[-200:]}")
        except Exception as e:
            logger.warning(f"Audio preparation error: {e}")

        return audio_path_str


# ── ComfyUI registration ──────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "LTX_VideoDecoder": LTX_VideoDecoder,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX_VideoDecoder": "🎬 LTX Video Decoder",
}
