"""
SmartVAE Advanced Streaming Decoder
Extends base SmartVAE_StreamingDecoder with experimental features.

NEW Features in Advanced:
- Anti-color bleed correction (reduces color jumps on chunk boundaries)
- Color space normalization between chunks
- Experimental temporal consistency improvements

Based on: SmartVAE_StreamingDecoder v11.4 (Codex fixes)
Version: 1.0.0 (Advanced)
License: MIT
"""

import torch
import logging
from typing import Optional, Dict

from .universal_smart_vae_video_decode import (
    SmartVAE_StreamingDecoder,
    StreamingVideoConfig,
    StreamingVideoWriter,
    IMAGEIO_AVAILABLE,
    NUMPY_AVAILABLE,
    CV2_AVAILABLE,
    PSUTIL_AVAILABLE
)

import folder_paths
import time
import os
import gc
import comfy.utils
from comfy.model_management import throw_exception_if_processing_interrupted
import subprocess

logger = logging.getLogger(__name__)


class SmartVAEAdvancedDecoder(SmartVAE_StreamingDecoder):
    """
    Advanced streaming decoder with anti-color bleed.
    
    Inherits ALL functionality from SmartVAE_StreamingDecoder,
    only overrides _streaming_decode to add color correction.
    
    Key differences from base:
    - Tracks color statistics between chunks
    - Normalizes overlap zones to prevent color jumps
    - Experimental - test before production use!
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get base inputs from parent
        base_inputs = super().INPUT_TYPES()
        
        # Add new experimental parameter
        base_inputs["optional"]["anti_color_bleed"] = ("BOOLEAN", {
            "default": False,
            "tooltip": (
                "🧪 EXPERIMENTAL: Normalize color between chunks.\n"
                "✓ Reduces color jumps at chunk boundaries\n"
                "⚠ May affect artistic color grading\n"
                "⚠ Adds ~5-10% processing time\n"
                "📝 Test on short clips first (30-60s)!"
            )
        })
        
        return base_inputs
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview_thumbs", "video_path")
    FUNCTION = "decode"
    OUTPUT_NODE = True
    CATEGORY = "latent/video"
    
    def decode(self, vae, samples, frames_per_batch, overlap_frames=2, force_time_scale=0, 
               enable_tiling=False, tile_size=512, verbose=False,
               video_output_path="", fps=24, codec="h264", audio_path="", audio=None, 
               resume_on_crash=True, ignore_warnings="none", anti_color_bleed=False):
        """
        Main decode - adds anti_color_bleed parameter to base.
        """
        self._verbose = verbose
        self._anti_color_bleed = anti_color_bleed
        
        if anti_color_bleed and verbose:
            logger.info("🧪 Anti-color bleed ENABLED (experimental)")
        
        # Delegate to parent's decode (which calls our overridden _streaming_decode)
        return super().decode(
            vae, samples, frames_per_batch, overlap_frames, force_time_scale,
            enable_tiling, tile_size, verbose, video_output_path, fps, codec,
            audio_path, audio, resume_on_crash, ignore_warnings
        )
    
    def _streaming_decode(self, vae, latents, video_output_path, fps, codec, audio_path,
                          resume_on_crash, frames_per_batch, overlap_frames,
                          force_time_scale, enable_tiling, tile_size, verbose,
                          time_scale, expected_frames, aspect_ratio, output_h, output_w,
                          h_latent, w_latent, latent_channels, ignore_warnings="none"):
        """
        OVERRIDE: Adds anti-color bleed logic to parent's streaming decode.
        
        This method is 95% identical to parent - we only add color correction.
        All validation, OOM handling, resume logic inherited from parent.
        """
        
        batch, _, total_frames, _, _ = latents.shape
        
        # ===== VALIDATION (from parent) =====
        if verbose:
            logger.info("🔍 Validating latent before decode...")
        
        try:
            _ = latents[0, 0, 0, 0, 0].item()
            
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
                
                if ignore_warnings == "none":
                    raise ValueError(
                        f"Latent contains {nan_percent:.1f}% NaN values - cannot decode safely.\n"
                        "Set 'ignore_warnings' to 'minor' or 'all' to force decode."
                    )
                
                elif ignore_warnings == "minor":
                    if nan_percent > 10.0:
                        raise ValueError(
                            f"Latent is {nan_percent:.1f}% corrupted (limit: 10% for 'minor' mode)."
                        )
                    logger.warning("⚠️ CONTINUING WITH CORRUPTED LATENT (user override)")
                    latents = torch.nan_to_num(latents, nan=0.0, posinf=1.0, neginf=0.0)
                
                elif ignore_warnings == "all":
                    logger.error("🚨 FORCING DECODE WITH SEVERELY CORRUPTED LATENT!")
                    latents = torch.nan_to_num(latents, nan=0.0, posinf=1.0, neginf=0.0)
            
            if verbose:
                min_val = latents.min().item()
                max_val = latents.max().item()
                logger.info(f"   ✓ Latent range: [{min_val:.4f}, {max_val:.4f}]")
        
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "cuda" in error_msg or "memory" in error_msg:
                logger.error("🚨 SILENT OOM DETECTED IN UPSTREAM NODE!")
                logger.warning("⚠️ Attempting emergency CPU recovery...")
                try:
                    latents = latents.cpu()
                    vae.first_stage_model.to('cpu')
                except Exception:
                    raise
        
        # Adaptive batch size for long sequences
        if total_frames > 1000:
            original_batch = frames_per_batch
            if total_frames > 2000:
                safe_batch = 4
            elif total_frames > 1500:
                safe_batch = 6
            else:
                safe_batch = 8
            
            if frames_per_batch > safe_batch:
                frames_per_batch = safe_batch
                if verbose:
                    logger.info(f"🛡️ Auto-reducing batch: {original_batch} → {frames_per_batch}")
        
        if verbose:
            logger.info("✓ Latent validation complete")
        
        # ===== WRITER SETUP =====
        writer = StreamingVideoWriter(
            video_output_path, fps, codec, output_h, output_w, resume=resume_on_crash
        )
        
        frames_per_batch = max(1, min(frames_per_batch, total_frames))
        overlap_frames = max(0, min(overlap_frames, frames_per_batch - 1))
        initial_overlap = overlap_frames
        
        # VRAM adaptive sizing
        available_vram = self._get_available_vram()
        if available_vram is not None:
            chunk_frames = frames_per_batch + 2 * overlap_frames
            est_vram = self._estimate_chunk_vram(chunk_frames, latent_channels, h_latent, w_latent, time_scale)
            
            if est_vram > available_vram * 0.55:
                reduction = (available_vram * 0.45) / est_vram
                old_batch = frames_per_batch
                frames_per_batch = max(1, int(frames_per_batch * reduction))
                overlap_frames = min(overlap_frames, frames_per_batch - 1)
                
                if verbose:
                    logger.info(f"📉 VRAM reduction: {old_batch} → {frames_per_batch}")
        
        if verbose:
            logger.info(f"   Batch: {frames_per_batch}, Overlap: {overlap_frames}")
        
        # ===== DECODE LOOP SETUP =====
        current_batch = frames_per_batch
        resume_output_frames = writer.frames_written if resume_on_crash else 0
        start_idx = (
            self._output_frames_to_latent_index(resume_output_frames, time_scale, total_frames)
            if resume_on_crash else 0
        )
        
        frames_processed = 0
        last_frames_processed = -1
        stagnation_count = 0
        MAX_STAGNATION = 3
        oom_retry_count = 0
        
        pbar = comfy.utils.ProgressBar(total_frames)
        preview_frames = []
        
        # ===== NEW: Anti-color bleed state =====
        prev_chunk_stats = None
        
        # ===== MAIN DECODE LOOP =====
        while start_idx < total_frames:
            throw_exception_if_processing_interrupted()
            
            if oom_retry_count >= self.MAX_OOM_RETRIES:
                raise RuntimeError(f"Exceeded {self.MAX_OOM_RETRIES} OOM retries")
            
            if frames_processed == last_frames_processed:
                stagnation_count += 1
                if stagnation_count >= MAX_STAGNATION:
                    raise RuntimeError(f"Stalled at frame {start_idx}")
            else:
                stagnation_count = 0
            last_frames_processed = frames_processed
            
            end_idx = min(start_idx + current_batch, total_frames)
            ctx_start = max(0, start_idx - overlap_frames)
            ctx_end = min(total_frames, end_idx + overlap_frames)
            
            if ctx_end <= ctx_start:
                raise RuntimeError(f"Empty chunk")
            
            latent_chunk = latents[:, :, ctx_start:ctx_end, :, :]
            
            # Decode chunk
            try:
                with torch.no_grad():
                    if enable_tiling and hasattr(vae, 'decode_tiled'):
                        decoded_raw = vae.decode_tiled(latent_chunk, tile_x=tile_size, tile_y=tile_size)
                    else:
                        decoded_raw = vae.decode(latent_chunk)
                    decoded_chunk = self._extract_tensor_output(decoded_raw).cpu()
                
                oom_retry_count = 0
            
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom_retry_count += 1
                    if verbose:
                        logger.warning(f"OOM at {start_idx} (retry {oom_retry_count}/{self.MAX_OOM_RETRIES})")
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
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
                else:
                    raise
            
            # Normalize output
            decoded_chunk = self._normalize_output(decoded_chunk, aspect_ratio)
            
            # Extract valid frames
            front_trim = (start_idx - ctx_start) * time_scale
            
            if end_idx == total_frames:
                valid_frames = decoded_chunk[front_trim:]
            else:
                core_length = (end_idx - start_idx) * time_scale
                valid_frames = decoded_chunk[front_trim:front_trim + core_length]
            
            valid_frames = self._center_crop_to_reference(valid_frames, output_h, output_w)
            
            # ===== NEW: ANTI-COLOR BLEED CORRECTION =====
            if self._anti_color_bleed and prev_chunk_stats is not None and overlap_frames > 0:
                if verbose:
                    logger.info(f"🎨 Applying color correction (chunk at latent frame {start_idx})")
                
                valid_frames = self._apply_color_correction(
                    valid_frames, prev_chunk_stats, overlap_frames * time_scale, verbose
                )
            
            # Store color stats for next chunk
            if self._anti_color_bleed and valid_frames.shape[0] > 0:
                prev_chunk_stats = self._extract_color_stats(
                    valid_frames, overlap_frames * time_scale
                )
            
            # ===== WRITE FRAMES =====
            for frame_idx in range(valid_frames.shape[0]):
                frame = valid_frames[frame_idx]
                
                preview = writer.write_frame(frame)
                
                if preview is not None:
                    preview_frames.append(torch.from_numpy(preview).float() / 255.0)
                    if len(preview_frames) > 5:
                        preview_frames.pop(0)
            
            processed_this_chunk = end_idx - start_idx
            frames_processed += processed_this_chunk
            pbar.update(processed_this_chunk)
            
            progress_pct = (writer.frames_written / expected_frames) * 100
            if verbose:
                logger.info(f"Progress: {writer.frames_written}/{expected_frames} ({progress_pct:.1f}%)")
            
            start_idx = end_idx
            
            del latent_chunk, decoded_chunk, valid_frames
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # ===== FINALIZE =====
        final_path = writer.finalize(audio_path if audio_path else None)
        
        if ignore_warnings != "none":
            logger.warning("⚠️ VIDEO CREATED WITH WARNINGS IGNORED")
        
        if verbose:
            logger.info(f"✅ Streaming complete!")
            logger.info(f"   File: {final_path}")
            logger.info(f"   Frames: {writer.frames_written}")
            file_size_mb = os.path.getsize(final_path) / (1024 * 1024)
            logger.info(f"   Size: {file_size_mb:.2f} MB")
        
        if preview_frames:
            preview_tensor = torch.stack(preview_frames, dim=0)
        else:
            preview_tensor = torch.empty(0, dtype=torch.float32)
        
        return (preview_tensor, final_path)
    
    # ===== NEW METHODS: Color correction =====
    
    def _extract_color_stats(self, frames: torch.Tensor, overlap_size: int) -> Dict:
        """
        Extract color statistics from end of chunk for next chunk's correction.
        
        Args:
            frames: [F, H, W, 3] decoded frames
            overlap_size: Number of frames to analyze at end
        
        Returns:
            dict with 'mean' and 'std' per RGB channel
        """
        if frames.shape[0] < overlap_size:
            overlap_size = frames.shape[0]
        
        if overlap_size == 0:
            return {'mean': [0.5, 0.5, 0.5], 'std': [0.1, 0.1, 0.1]}
        
        # Analyze last N frames (will overlap with next chunk's start)
        overlap_zone = frames[-overlap_size:]
        
        stats = {
            'mean': [],
            'std': []
        }
        
        for c in range(3):  # RGB channels
            channel_data = overlap_zone[..., c]
            stats['mean'].append(channel_data.mean().item())
            stats['std'].append(max(channel_data.std().item(), 1e-6))  # Avoid div by zero
        
        return stats
    
    def _apply_color_correction(self, frames: torch.Tensor, prev_stats: Dict, 
                                overlap_size: int, verbose: bool) -> torch.Tensor:
        """
        Normalize color of current chunk's start to match previous chunk's end.
        
        Uses histogram matching / color transfer approach:
        1. Extract overlap zone at start of current chunk
        2. Normalize each channel: (x - μ_curr) * (σ_prev / σ_curr) + μ_prev
        3. Clamp to [0, 1]
        
        Args:
            frames: [F, H, W, 3] current chunk
            prev_stats: Color stats from previous chunk's end
            overlap_size: Number of frames to correct at start
            verbose: Logging
        
        Returns:
            Color-corrected frames
        """
        if frames.shape[0] < overlap_size:
            overlap_size = frames.shape[0]
        
        if overlap_size == 0:
            return frames
        
        corrected = frames.clone()
        overlap_zone = corrected[:overlap_size]  # First N frames
        
        color_shifts = []
        
        for c in range(3):  # RGB
            curr_mean = overlap_zone[..., c].mean()
            curr_std = max(overlap_zone[..., c].std(), 1e-6)
            
            target_mean = prev_stats['mean'][c]
            target_std = prev_stats['std'][c]
            
            # Histogram matching normalization
            normalized = (overlap_zone[..., c] - curr_mean) * (target_std / curr_std) + target_mean
            overlap_zone[..., c] = torch.clamp(normalized, 0.0, 1.0)
            
            shift = abs(target_mean - curr_mean.item())
            color_shifts.append(shift)
            
            if verbose and shift > 0.05:
                channel_name = ['R', 'G', 'B'][c]
                logger.debug(
                    f"   {channel_name}: {curr_mean:.3f}→{target_mean:.3f} "
                    f"(shift: {shift:.3f}, std: {curr_std:.3f}→{target_std:.3f})"
                )
        
        corrected[:overlap_size] = overlap_zone
        
        avg_shift = sum(color_shifts) / 3
        if verbose and avg_shift > 0.03:
            logger.info(f"   💡 Color correction applied (avg shift: {avg_shift:.3f})")
        
        return corrected


NODE_CLASS_MAPPINGS = {
    "SmartVAE_AdvancedDecoder": SmartVAEAdvancedDecoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartVAE_AdvancedDecoder": "🎞️ SmartVAE Advanced Decoder (🧪 Experimental)",
}
