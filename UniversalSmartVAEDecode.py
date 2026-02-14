"""
Universal Smart VAE Decode - v11.3 (Ignore Warnings + NaN safety)
Production Hardened VAE decoder with disk offloading and user override for corrupted latents

CRITICAL FIXES in v11.3:
- Ignore warnings mode for NaN/corrupted latents
- Detailed corruption detection (percent, affected frames)
- 3-tier handling: stop / minor / force
- Final warning in logs if override used

Credits: Grok (disk offload, orientation), Claude (ignore warnings + safety), Kimi (memory), Gemini (math), GPT (structure)
Version: 11.3.0
License: MIT
GitHub: https://github.com/uczensokratesa/ComfyUI-Grok-SmartVAE
"""

import torch
import comfy.utils
from comfy.model_management import throw_exception_if_processing_interrupted
import gc
import math
import os
import tempfile
import warnings
import atexit
import logging
from typing import Optional, Tuple

# Setup logging
logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn(
        "psutil not available. Install with 'pip install psutil' for automatic RAM detection.\n"
        "Disk offload will still work based on max_ram_frames threshold."
    )


class UniversalSmartVAEDecode:
    """
    Production-grade VAE decoder with disk offloading and safety overrides.
    
    Features:
    - Handles 2000+ frames with automatic memory management
    - Disk offload when RAM insufficient
    - Frame-perfect audio sync (if integrated downstream)
    - Multi-stage OOM recovery
    - NEW: Ignore warnings for corrupted latents (risk of black frames)
    """
    
    MAX_OOM_RETRIES = 5
    DISK_MERGE_CHUNK_SIZE = 10
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "frames_per_batch": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 128,
                    "step": 1,
                    "tooltip": "Frames to decode per batch. Auto-reduces on OOM."
                }),
            },
            "optional": {
                "overlap_frames": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Temporal overlap for seamless stitching."
                }),
                "force_time_scale": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Manual time scale override (0=auto). E.g., 8 for LTX-Video."
                }),
                "enable_tiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force spatial tiling. Auto-enables on OOM."
                }),
                "tile_size": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Tile size in pixels for spatial tiling."
                }),
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show detailed progress logs."
                }),
                "max_ram_frames": ("INT", {
                    "default": 500,
                    "min": 100,
                    "max": 10000,
                    "step": 100,
                    "tooltip": "Threshold for disk offload. Tune to system RAM: 16GB=300, 32GB=600, 64GB=1000."
                }),
                "ignore_warnings": (["none", "minor", "all"], {
                    "default": "none",
                    "tooltip": (
                        "Corrupted latent handling:\n"
                        "• none = Stop on NaN/corruption (safest)\n"
                        "• minor = Try if <10% corrupt (may produce black frames)\n"
                        "• all = Force decode anyway (high crash/black frames risk)"
                    )
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "latent/video"

    def __init__(self):
        self._time_scale_cache = {}
        self._force_scale_cache = {}
        self.temp_dir = None
        self._verbose = False
        
        atexit.register(self._cleanup_temp_dir)

    def _cleanup_temp_dir(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                for file in os.listdir(self.temp_dir):
                    try:
                        os.remove(os.path.join(self.temp_dir, file))
                    except Exception as e:
                        logger.debug(f"Failed to remove temp file: {e}")
                os.rmdir(self.temp_dir)
                logger.debug(f"Cleaned up temp dir: {self.temp_dir}")
            except Exception as e:
                logger.debug(f"Temp dir cleanup error: {e}")

    def __del__(self):
        self._cleanup_temp_dir()

    def _get_available_vram(self) -> Optional[float]:
        try:
            if not torch.cuda.is_available():
                return None
            free_vram, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
            return free_vram / (1024 ** 3)
        except Exception as e:
            logger.warning(f"Failed to get VRAM info: {e}")
            return None

    def _get_available_ram(self) -> Optional[float]:
        if PSUTIL_AVAILABLE:
            try:
                return psutil.virtual_memory().available / (1024 ** 3)
            except Exception as e:
                logger.warning(f"Failed to get RAM info: {e}")
        return None

    def detect_time_scale(self, vae, latents: torch.Tensor, force_scale: int = 0, 
                          verbose: bool = True) -> int:
        vae_id = id(vae)
        
        if force_scale > 0:
            if self._force_scale_cache.get(vae_id) != force_scale:
                self._time_scale_cache.pop(vae_id, None)
                self._force_scale_cache[vae_id] = force_scale
            
            if verbose:
                logger.info(f"🔧 Forced time scale: {force_scale}x")
            
            self._time_scale_cache[vae_id] = force_scale
            return force_scale
        
        self._force_scale_cache.pop(vae_id, None)
        
        if vae_id in self._time_scale_cache:
            return self._time_scale_cache[vae_id]
        
        if hasattr(vae, 'downscale_index_formula') and vae.downscale_index_formula:
            try:
                time_scale = int(vae.downscale_index_formula[0])
                if verbose:
                    logger.info(f"🔍 VAE metadata time scale: {time_scale}x")
                self._time_scale_cache[vae_id] = time_scale
                return time_scale
            except Exception as e:
                logger.debug(f"Metadata parse failed: {e}")
        
        try:
            total_latent_frames = latents.shape[2]
            test_frames = min(5, total_latent_frames)
            
            if test_frames <= 1:
                if verbose:
                    logger.warning(f"Insufficient frames for detection. Fallback: 1x")
                self._time_scale_cache[vae_id] = 1
                return 1
            
            test_sample = latents[:, :, 0:test_frames, :16, :16]
            
            with torch.no_grad():
                test_output = vae.decode(test_sample)
            
            test_output = self._normalize_output(test_output, aspect_ratio=1.0)
            output_frames = test_output.shape[0]
            
            time_scale = max(1, (output_frames - 1) // (test_frames - 1))
            
            if verbose:
                logger.info(f"🔍 Auto-detected: {time_scale}x ({test_frames}→{output_frames})")
            
            self._time_scale_cache[vae_id] = time_scale
            
            del test_output, test_sample
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return time_scale
        
        except Exception as e:
            if verbose:
                logger.warning(f"Detection failed: {e}. Fallback: 1x")
            
            self._time_scale_cache[vae_id] = 1
            return 1

    def detect_output_size(self, vae, latents: torch.Tensor, h_latent: int, w_latent: int, 
                           tile_size: int = 512, verbose: bool = True) -> Tuple[int, int]:
        aspect_ratio = h_latent / float(w_latent)
        test_sample = latents[:, :, 0:1, :, :]
        
        try:
            with torch.no_grad():
                test_out = vae.decode(test_sample)
            if verbose:
                logger.info("🔍 Output size detected (standard)")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                try:
                    with torch.no_grad():
                        test_out = vae.decode_tiled(test_sample, tile_x=tile_size, tile_y=tile_size)
                    if verbose:
                        logger.info("🔍 Output size detected (tiled)")
                except Exception as inner_e:
                    raise RuntimeError("Failed to detect output size even with tiling.") from inner_e
            else:
                raise
        
        test_out = self._normalize_output(test_out, aspect_ratio)
        output_h, output_w = test_out.shape[1:3]
        
        del test_out, test_sample
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return output_h, output_w

    def _estimate_chunk_vram(self, frames: int, channels: int, h: int, w: int, 
                            time_scale: int = 1) -> float:
        spatial_scale = 8
        latent_bytes = frames * channels * h * w * 4
        output_frames = frames * time_scale
        output_bytes = output_frames * 3 * (h * spatial_scale) * (w * spatial_scale) * 4
        total_bytes = (latent_bytes + output_bytes) * 3.5 * 1.1
        return total_bytes / (1024 ** 3)

    def _estimate_output_ram(self, expected_frames: int, output_h: int, output_w: int) -> float:
        bytes_per_frame = output_h * output_w * 3 * 4
        return (expected_frames * bytes_per_frame) / (1024 ** 3)

    def _normalize_output(self, tensor, aspect_ratio: Optional[float] = None) -> torch.Tensor:
        if isinstance(tensor, (list, tuple)):
            if not tensor:
                raise ValueError("VAE returned empty output")
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
                shape = list(tensor.shape)
            
            try:
                c_idx = next(i for i, s in enumerate(shape) if s in (3, 4))
            except StopIteration:
                raise ValueError(f"Cannot find channel dim in {shape}")
            
            remaining_idxs = [i for i in range(4) if i != c_idx]
            remaining_sizes = [shape[i] for i in remaining_idxs]
            
            sorted_remaining = sorted(zip(remaining_idxs, remaining_sizes), key=lambda x: x[1])
            f_idx = sorted_remaining[0][0]
            spatial_large_idx = sorted_remaining[2][0]
            spatial_small_idx = sorted_remaining[1][0]
            
            if aspect_ratio is not None:
                if aspect_ratio > 1.0:
                    h_idx = spatial_large_idx
                    w_idx = spatial_small_idx
                else:
                    w_idx = spatial_large_idx
                    h_idx = spatial_small_idx
            else:
                h_idx = spatial_large_idx
                w_idx = spatial_small_idx
            
            perm = [f_idx, h_idx, w_idx, c_idx]
            tensor = tensor.permute(*perm)
        
        else:
            raise ValueError(f"Unsupported: {dim}D, shape {tensor.shape}")
        
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        if min_val < 0.0:
            tensor = (tensor + 1.0) / 2.0
        
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)
        tensor = torch.clamp(tensor, 0.0, 1.0)
        
        if self._verbose:
            if min_val < 0.0 or max_val > 1.0:
                logger.warning(f"Normalize: auto-scaled and clamped [{min_val:.4f}, {max_val:.4f}] → [0,1]")
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                logger.warning("NaN/Inf detected and cleaned in output!")
        
        if tensor.shape[-1] > 3:
            tensor = tensor[..., :3]
        
        return tensor.contiguous()

    def _center_crop_to_reference(self, tensor: torch.Tensor, h_ref: int, w_ref: int) -> torch.Tensor:
        _, h, w, _ = tensor.shape
        if h == h_ref and w == w_ref:
            return tensor
        
        h_offset = max(0, (h - h_ref) // 2)
        w_offset = max(0, (w - w_ref) // 2)
        
        return tensor[:, h_offset:h_offset + h_ref, w_offset:w_offset + w_ref, :]

    def decode(self, vae, samples, frames_per_batch, overlap_frames=2, force_time_scale=0, 
               enable_tiling=False, tile_size=512, verbose=True, max_ram_frames=500,
               ignore_warnings="none"):
        """
        Main decode with disk offloading, OOM protection and corrupted latent override.
        """
        self._verbose = verbose
        
        latents = samples["samples"]
        
        # ======== IMAGE PATH ========
        if latents.dim() == 4:
            if verbose:
                logger.info("🖼️  Image decode")
            
            with torch.no_grad():
                if enable_tiling and hasattr(vae, 'decode_tiled'):
                    output = vae.decode_tiled(latents, tile_x=tile_size, tile_y=tile_size)
                else:
                    output = vae.decode(latents)
            
            return (self._normalize_output(output, aspect_ratio=1.0),)
        
        # ======== VIDEO PATH ========
        batch, channels, total_frames, h_latent, w_latent = latents.shape
        
        if total_frames <= 0:
            raise ValueError("Latent has no frames")
        
        if verbose:
            logger.info("🔍 Validating latent...")
        
        # ============= LATENT VALIDATION + IGNORE WARNINGS =============
        if torch.isnan(latents).any():
            nan_count = torch.isnan(latents).sum().item()
            total_elements = latents.numel()
            nan_percent = (nan_count / total_elements) * 100
            
            logger.error("=" * 70)
            logger.error("🚨 CORRUPTED LATENT DETECTED!")
            logger.error(f"   NaN values: {nan_count:,} / {total_elements:,} ({nan_percent:.2f}%)")
            logger.error(f"   Shape: {latents.shape}")
            
            nan_frames = torch.isnan(latents).any(dim=(0,1,3,4)).nonzero(as_tuple=True)[0]
            if len(nan_frames) > 0:
                first_bad = nan_frames[0].item()
                last_bad = nan_frames[-1].item()
                logger.error(f"   Affected frames: {first_bad} to {last_bad}")
                logger.error(f"   → Corruption likely started around frame {first_bad}")
            
            logger.error("")
            logger.error("💊 RECOMMENDED FIXES:")
            logger.error("   1. REDUCE VIDEO LENGTH (most effective)")
            logger.error(f"      Current: {total_frames} frames → try {int(total_frames * 0.6)}")
            logger.error("   2. REDUCE RESOLUTION")
            logger.error("   3. Lower CFG scale (e.g. 3.5 instead of 7.0)")
            logger.error("   4. Use TensorParallel or Kijaji nodes")
            logger.error("   5. Clear VRAM before sampling")
            logger.error("=" * 70)
            
            if ignore_warnings == "none":
                raise ValueError(
                    f"Latent contains {nan_percent:.1f}% NaN values - cannot decode safely.\n"
                    f"The sampler ran out of VRAM.\n\n"
                    f"To force decode anyway (may produce black frames):\n"
                    f"  Set 'ignore_warnings' to 'minor' (<10%) or 'all' (risky)"
                )
            
            elif ignore_warnings == "minor":
                if nan_percent > 10.0:
                    raise ValueError(
                        f"Latent is {nan_percent:.1f}% corrupted (limit: 10% for 'minor' mode).\n"
                        f"This is too severe - output would be mostly black.\n\n"
                        f"Options:\n"
                        f"  1. Reduce video length/resolution and re-sample\n"
                        f"  2. Set 'ignore_warnings' to 'all' (not recommended)"
                    )
                
                logger.warning("=" * 70)
                logger.warning("⚠️ CONTINUING WITH CORRUPTED LATENT (user override)")
                logger.warning(f"   Corruption: {nan_percent:.2f}% (under 10% threshold)")
                logger.warning(f"   Frames {first_bad}-{last_bad} will likely be BLACK")
                logger.warning("   Rest of video may be OK")
                logger.warning("=" * 70)
                
                latents = torch.nan_to_num(latents, nan=0.0, posinf=1.0, neginf=0.0)
            
            elif ignore_warnings == "all":
                logger.error("=" * 70)
                logger.error("🚨 FORCING DECODE WITH SEVERELY CORRUPTED LATENT!")
                logger.error(f"   Corruption: {nan_percent:.2f}%")
                logger.error("   ⚠️ WARNING: Output will likely be:")
                logger.error("      • Mostly/entirely BLACK")
                logger.error("      • May CRASH during decode")
                logger.error("      • Waste time and produce unusable file")
                logger.error("")
                logger.error("   You have been warned. Proceeding anyway...")
                logger.error("=" * 70)
                
                latents = torch.nan_to_num(latents, nan=0.0, posinf=1.0, neginf=0.0)
        
        if torch.isinf(latents).any():
            raise ValueError("Latent contains Inf values - upstream overflow!")
        
        if verbose:
            logger.info("✓ Latent validation passed")
        # ===============================================================

        # Detect scales
        time_scale = self.detect_time_scale(vae, latents, force_time_scale, verbose)
        expected_frames = 1 + (total_frames - 1) * time_scale
        aspect_ratio = h_latent / float(w_latent)
        
        # Detect output size
        output_h, output_w = self.detect_output_size(vae, latents, h_latent, w_latent, tile_size, verbose)
        
        # RAM estimation
        est_ram = self._estimate_output_ram(expected_frames, output_h, output_w)
        available_ram = self._get_available_ram()
        
        use_disk_offload = (
            expected_frames > max_ram_frames or
            (available_ram is not None and est_ram > available_ram * 0.5)
        )
        
        if use_disk_offload:
            self.temp_dir = tempfile.mkdtemp(prefix="comfy_smartvae_")
            if verbose:
                logger.info(f"💾 Disk offload enabled ({expected_frames} frames, est {est_ram:.2f}GB RAM)")
        
        if verbose:
            logger.info(f"🎬 Video decode:")
            logger.info(f"   Input: {total_frames} latent frames")
            logger.info(f"   Time scale: {time_scale}x")
            logger.info(f"   Expected output: ~{expected_frames} frames")
            logger.info(f"   Output resolution: {output_h}x{output_w}")
            logger.info(f"   Estimated RAM: {est_ram:.2f}GB")
            
            if available_ram:
                ram_pct = (est_ram / available_ram) * 100
                logger.info(f"   RAM usage: {ram_pct:.1f}% of {available_ram:.1f}GB available")
            
            orientation = 'portrait' if aspect_ratio < 1.0 else 'landscape'
            logger.info(f"   Aspect ratio: {aspect_ratio:.3f} ({orientation})")
            logger.info(f"   Mode: {'💾 Disk offload' if use_disk_offload else '🚀 In-memory (pinned)'}")
            
            if est_ram > 16:
                warnings.warn(
                    f"High RAM usage ({est_ram:.1f}GB). "
                    f"Consider lowering resolution or increasing max_ram_frames."
                )
        
        frames_per_batch = max(1, min(frames_per_batch, total_frames))
        overlap_frames = max(0, min(overlap_frames, frames_per_batch - 1))
        initial_overlap = overlap_frames
        
        available_vram = self._get_available_vram()
        if available_vram is not None:
            chunk_frames = frames_per_batch + 2 * overlap_frames
            est_vram = self._estimate_chunk_vram(chunk_frames, channels, h_latent, w_latent, time_scale)
            
            if est_vram > available_vram * 0.55:
                reduction = (available_vram * 0.45) / est_vram
                old_batch = frames_per_batch
                frames_per_batch = max(1, int(frames_per_batch * reduction))
                overlap_frames = min(overlap_frames, frames_per_batch - 1)
                
                if verbose:
                    logger.info(f"📉 Predictive VRAM reduction:")
                    logger.info(f"   Batch: {old_batch} → {frames_per_batch}")
                    logger.info(f"   Est: {est_vram:.2f}GB / Avail: {available_vram:.2f}GB")
        
        if verbose:
            logger.info(f"   Batch size: {frames_per_batch}")
            logger.info(f"   Overlap: {overlap_frames} frames")
        
        temp_files = [] if use_disk_offload else None
        final_output = None
        current_write_idx = 0
        
        if not use_disk_offload:
            try:
                final_output = torch.empty(
                    expected_frames, output_h, output_w, 3,
                    dtype=torch.float32,
                    device='cpu',
                    pin_memory=torch.cuda.is_available()
                )
                if verbose:
                    logger.info("   ✓ Pre-allocated pinned memory")
            except Exception as e:
                if verbose:
                    logger.warning(f"Pre-allocation failed: {e}")
                    logger.info("→ Falling back to disk offload")
                use_disk_offload = True
                self.temp_dir = tempfile.mkdtemp(prefix="comfy_smartvae_")
                temp_files = []
        
        current_batch = frames_per_batch
        start_idx = 0
        frames_processed = 0
        last_frames_processed = -1
        stagnation_count = 0
        MAX_STAGNATION = 3
        cumulative_output_frames = 0
        oom_retry_count = 0
        
        pbar = comfy.utils.ProgressBar(total_frames)
        
        est_total_chunks = math.ceil(total_frames / frames_per_batch)
        log_every_n = 5 if est_total_chunks > 20 else 1
        chunk_count = 0
        
        while start_idx < total_frames:
            throw_exception_if_processing_interrupted()
            
            if oom_retry_count >= self.MAX_OOM_RETRIES:
                raise RuntimeError(
                    f"Exceeded maximum OOM retries ({self.MAX_OOM_RETRIES}).\n"
                    f"Current settings: batch={current_batch}, tile={tile_size}, tiling={enable_tiling}\n"
                    f"This indicates insufficient VRAM for this workflow.\n\n"
                    f"Suggestions:\n"
                    f"  1. Reduce latent resolution\n"
                    f"  2. Close other GPU applications\n"
                    f"  3. Process video in smaller segments"
                )
            
            if frames_processed == last_frames_processed:
                stagnation_count += 1
                if stagnation_count >= MAX_STAGNATION:
                    raise RuntimeError(
                        f"Processing stalled at frame {start_idx}/{total_frames}.\n"
                        f"Settings: batch={current_batch}, overlap={overlap_frames}, "
                        f"tiling={enable_tiling}, tile={tile_size}"
                    )
            else:
                stagnation_count = 0
            last_frames_processed = frames_processed
            
            end_idx = min(start_idx + current_batch, total_frames)
            ctx_start = max(0, start_idx - overlap_frames)
            ctx_end = min(total_frames, end_idx + overlap_frames)
            
            if ctx_end <= ctx_start:
                raise RuntimeError(f"Empty chunk (ctx {ctx_start}→{ctx_end})")
            
            latent_chunk = latents[:, :, ctx_start:ctx_end, :, :]
            
            should_log = verbose and (chunk_count % log_every_n == 0 or end_idx == total_frames)
            if should_log:
                mem_gb = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
                logger.info(
                    f"Chunk {chunk_count+1}/{est_total_chunks}: "
                    f"latent {start_idx}→{end_idx} | ctx {ctx_start}→{ctx_end} | "
                    f"VRAM: {mem_gb:.2f}GB"
                )
            
            try:
                with torch.no_grad():
                    if enable_tiling and hasattr(vae, 'decode_tiled'):
                        decoded_chunk = vae.decode_tiled(latent_chunk, tile_x=tile_size, tile_y=tile_size)
                    else:
                        decoded_chunk = vae.decode(latent_chunk)
                    decoded_chunk = decoded_chunk.cpu()
                
                decoded_chunk = torch.nan_to_num(decoded_chunk, nan=0.0, posinf=1.0, neginf=0.0)
                decoded_chunk = torch.clamp(decoded_chunk, min=0.0, max=1.0)
                
                if verbose:
                    min_val = decoded_chunk.min().item()
                    max_val = decoded_chunk.max().item()
                    if min_val < 0.0 or max_val > 1.0:
                        logger.warning(f"   Chunk clamped: [{min_val:.4f}, {max_val:.4f}] → [0,1]")
                
                oom_retry_count = 0
            
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom_retry_count += 1
                    
                    if verbose:
                        logger.warning(f"OOM at frame {start_idx} (retry {oom_retry_count}/{self.MAX_OOM_RETRIES})")
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    gc.collect()
                    
                    if not enable_tiling:
                        logger.info("→ Stage 1: Enabling tiling")
                        enable_tiling = True
                        output_h, output_w = self.detect_output_size(
                            vae, latents, h_latent, w_latent, tile_size, verbose=False
                        )
                        continue
                    
                    if current_batch > 1:
                        old_batch = current_batch
                        current_batch = max(1, current_batch // 2)
                        overlap_frames = min(initial_overlap, current_batch - 1)
                        logger.info(f"→ Stage 2: Batch {old_batch} → {current_batch}")
                        continue
                    
                    if tile_size > 256:
                        old_tile = tile_size
                        tile_size = max(256, tile_size - 128)
                        logger.info(f"→ Stage 3: Tile {old_tile} → {tile_size}px")
                        continue
                    
                    continue
                else:
                    raise
            
            decoded_chunk = self._normalize_output(decoded_chunk, aspect_ratio)
            
            front_trim = (start_idx - ctx_start) * time_scale
            
            if end_idx == total_frames:
                valid_frames = decoded_chunk[front_trim:]
            else:
                core_length = (end_idx - start_idx) * time_scale
                valid_frames = decoded_chunk[front_trim:front_trim + core_length]
            
            actual = valid_frames.shape[0]
            if end_idx < total_frames:
                expected = (end_idx - start_idx) * time_scale
            else:
                expected = expected_frames - cumulative_output_frames
            
            if should_log and abs(actual - expected) > 0:
                logger.info(f"  Chunk frames: expected {expected}, got {actual}")
            
            valid_frames = self._center_crop_to_reference(valid_frames, output_h, output_w)
            
            if verbose and chunk_count == 0:
                logger.info(f"✓ First chunk: shape {valid_frames.shape}")
                if (aspect_ratio > 1.0 and valid_frames.shape[2] > valid_frames.shape[1]) or \
                   (aspect_ratio < 1.0 and valid_frames.shape[1] > valid_frames.shape[2]):
                    warnings.warn("Orientation mismatch! Output may be rotated.")
            
            if use_disk_offload:
                temp_path = os.path.join(self.temp_dir, f"chunk_{chunk_count:06d}.pt")
                torch.save(valid_frames, temp_path)
                temp_files.append(temp_path)
            else:
                final_output[current_write_idx:current_write_idx + actual] = valid_frames
                current_write_idx += actual
            
            processed_this_chunk = end_idx - start_idx
            frames_processed += processed_this_chunk
            cumulative_output_frames += actual
            pbar.update(processed_this_chunk)
            start_idx = end_idx
            chunk_count += 1
            
            del latent_chunk, decoded_chunk, valid_frames
            
            if chunk_count % 3 == 0 or current_batch <= 4:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
        
        if use_disk_offload:
            if verbose:
                logger.info(f"🛡️  Assembling {len(temp_files)} chunks from disk...")
            
            chunks = []
            for i, temp_path in enumerate(temp_files):
                chunk = torch.load(temp_path, map_location='cpu')
                chunks.append(chunk)
                
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.debug(f"Failed to remove temp file: {e}")
                
                if len(chunks) >= self.DISK_MERGE_CHUNK_SIZE or i == len(temp_files) - 1:
                    merged = torch.cat(chunks, dim=0)
                    chunks = [merged]
                    gc.collect()
            
            final_output = chunks[0] if chunks else torch.empty((0, output_h, output_w, 3), dtype=torch.float32)
            
            self._cleanup_temp_dir()
            self.temp_dir = None
        
        if final_output.shape[0] > expected_frames:
            final_output = final_output[:expected_frames]
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        actual_frames = final_output.shape[0]
        
        # ============= FINAL WARNING =============
        if ignore_warnings != "none":
            logger.warning("=" * 70)
            logger.warning("⚠️ VIDEO DECODED WITH WARNINGS IGNORED")
            logger.warning(f"   Mode: '{ignore_warnings}'")
            logger.warning(f"   Some frames MAY BE BLACK or corrupted")
            logger.warning("   Recommend: Check output before sharing")
            logger.warning("=" * 70)
        # =========================================
        
        if verbose:
            logger.info(f"✅ Decode complete!")
            logger.info(f"   Output frames: {actual_frames}")
            logger.info(f"   Final shape: {final_output.shape}")
        
        frame_diff = abs(actual_frames - expected_frames)
        if frame_diff == 0:
            if verbose:
                logger.info(f"   🎵 Audio sync: PERFECT ✓")
        elif frame_diff <= time_scale:
            if verbose:
                logger.info(f"   ⚠️  Minor diff: {frame_diff} (rounding, OK)")
        else:
            if verbose:
                logger.warning(f"Audio sync warning:")
                logger.warning(f"   Expected: {expected_frames}, Got: {actual_frames}")
                logger.warning(f"   Diff: {frame_diff} frames")
                logger.warning(f"   Try: force_time_scale={time_scale-1} or {time_scale+1}")
        
        return (final_output,)


NODE_CLASS_MAPPINGS = {
    "UniversalSmartVAEDecode": UniversalSmartVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalSmartVAEDecode": "🎬 Universal VAE Decode (v11.3 + Ignore Warnings)",
}
