"""
SmartVAE Advanced Streaming Decoder v2.1
Fixes: Brightness flicker in EMA mode with soft blending correction.

Based on: SmartVAE_StreamingDecoder v11.4 (Codex fixes)
Version: 2.1.0 (Flicker Fix)
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

logger = logging.getLogger(__name__)


class SmartVAEAdvancedDecoder(SmartVAE_StreamingDecoder):
    """
    Advanced streaming decoder with flicker-free color correction.
    
    v2.1 Fix: Soft blending instead of hard correction in EMA mode.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        
        base_inputs["optional"]["anti_color_bleed"] = ("BOOLEAN", {
            "default": False,
            "tooltip": (
                "🧪 EXPERIMENTAL: Temporal color correction.\n"
                "✓ Reduces color jumps at chunk boundaries\n"
                "💡 Enable with calibration_image for best results!"
            )
        })
        
        base_inputs["optional"]["calibration_image"] = ("IMAGE", {
            "tooltip": (
                "🎨 Optional reference image for color grading.\n"
                "• Alone: Matches first chunk to reference\n"
                "• With anti_color_bleed: Smooth temporal evolution\n"
            )
        })
        
        base_inputs["optional"]["ema_momentum"] = ("FLOAT", {
            "default": 0.85,
            "min": 0.5,
            "max": 0.95,
            "step": 0.05,
            "tooltip": (
                "⚙️ EMA smoothing (higher = smoother, slower adaptation).\n"
                "• 0.5-0.7 = responsive\n"
                "• 0.85 = balanced (default)\n"
                "• 0.9-0.95 = very smooth"
            )
        })
        
        base_inputs["optional"]["correction_strength"] = ("FLOAT", {
            "default": 0.4,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "tooltip": (
                "🎚️ How strongly to apply EMA correction.\n"
                "• 0.0 = no correction (bypass)\n"
                "• 0.3-0.5 = subtle (best for avoiding flicker)\n"
                "• 0.7-1.0 = strong (may flicker)\n"
                "\n"
                "💡 Lower = less flicker, but weaker color matching"
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
               resume_on_crash=True, ignore_warnings="none", anti_color_bleed=False,
               calibration_image=None, ema_momentum=0.85, correction_strength=0.4):
        """
        Main decode with flicker-free color correction.
        """
        self._verbose = verbose
        self._anti_color_bleed = anti_color_bleed
        self._ema_momentum = ema_momentum
        self._correction_strength = correction_strength
        
        # Extract reference stats
        self._calibration_stats = None
        if calibration_image is not None:
            self._calibration_stats = self._extract_reference_image_stats(calibration_image)
            if self._calibration_stats is None:
                logger.warning("Calibration image invalid - ignoring")
        
        # Determine mode
        has_ref = self._calibration_stats is not None
        has_bleed = anti_color_bleed
        
        if has_ref and has_bleed:
            self._correction_mode = "ema"
            if verbose:
                logger.info(
                    f"🎬 EMA mode: momentum={ema_momentum:.2f}, "
                    f"strength={correction_strength:.2f}"
                )
        elif has_bleed:
            self._correction_mode = "temporal"
            if verbose:
                logger.info("🎨 Temporal correction only")
        elif has_ref:
            self._correction_mode = "reference"
            if verbose:
                logger.info("🎨 Reference matching (first chunk)")
        else:
            self._correction_mode = "none"
        
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
        """OVERRIDE: Adds flicker-free color correction."""
        
        batch, _, total_frames, _, _ = latents.shape
        
        # ===== VALIDATION =====
        if verbose:
            logger.info("🔍 Validating latent...")
        
        try:
            _ = latents[0, 0, 0, 0, 0].item()
            
            if torch.isnan(latents).any():
                nan_pct = (torch.isnan(latents).sum().item() / latents.numel()) * 100
                
                if ignore_warnings == "none":
                    raise ValueError(f"Latent {nan_pct:.1f}% NaN - set ignore_warnings to continue")
                elif ignore_warnings == "minor" and nan_pct > 10.0:
                    raise ValueError(f"Too corrupt: {nan_pct:.1f}% (limit: 10%)")
                
                logger.warning(f"⚠️ Continuing with {nan_pct:.1f}% NaN (user override)")
                latents = torch.nan_to_num(latents, nan=0.0)
            
            if verbose:
                logger.info(f"   ✓ Range: [{latents.min().item():.4f}, {latents.max().item():.4f}]")
        
        except RuntimeError as e:
            if "memory" in str(e).lower():
                logger.error("🚨 OOM - CPU recovery...")
                latents = latents.cpu()
                vae.first_stage_model.to('cpu')
        
        # Adaptive batch
        if total_frames > 1000:
            orig = frames_per_batch
            safe = 4 if total_frames > 2000 else (6 if total_frames > 1500 else 8)
            if frames_per_batch > safe:
                frames_per_batch = safe
                if verbose:
                    logger.info(f"🛡️ Batch: {orig} → {frames_per_batch}")
        
        # ===== WRITER SETUP =====
        writer = StreamingVideoWriter(video_output_path, fps, codec, output_h, output_w, resume=resume_on_crash)
        
        frames_per_batch = max(1, min(frames_per_batch, total_frames))
        overlap_frames = max(0, min(overlap_frames, frames_per_batch - 1))
        initial_overlap = overlap_frames
        
        # VRAM adaptive
        vram = self._get_available_vram()
        if vram:
            est = self._estimate_chunk_vram(frames_per_batch + 2*overlap_frames, latent_channels, h_latent, w_latent, time_scale)
            if est > vram * 0.55:
                old = frames_per_batch
                frames_per_batch = max(1, int(frames_per_batch * (vram * 0.45 / est)))
                overlap_frames = min(overlap_frames, frames_per_batch - 1)
                if verbose:
                    logger.info(f"📉 VRAM: {old} → {frames_per_batch}")
        
        # ===== DECODE LOOP SETUP =====
        current_batch = frames_per_batch
        resume_frames = writer.frames_written if resume_on_crash else 0
        start_idx = self._output_frames_to_latent_index(resume_frames, time_scale, total_frames) if resume_on_crash else 0
        
        frames_processed = 0
        last_processed = -1
        stagnation = 0
        oom_retry = 0
        
        pbar = comfy.utils.ProgressBar(total_frames)
        preview_frames = []
        
        # Color correction state
        ema_stats = None
        prev_stats = None
        
        # ===== MAIN LOOP =====
        while start_idx < total_frames:
            throw_exception_if_processing_interrupted()
            
            if oom_retry >= self.MAX_OOM_RETRIES:
                raise RuntimeError("Max OOM retries exceeded")
            
            if frames_processed == last_processed:
                stagnation += 1
                if stagnation >= 3:
                    raise RuntimeError(f"Stalled at {start_idx}")
            else:
                stagnation = 0
            last_processed = frames_processed
            
            end_idx = min(start_idx + current_batch, total_frames)
            ctx_start = max(0, start_idx - overlap_frames)
            ctx_end = min(total_frames, end_idx + overlap_frames)
            
            latent_chunk = latents[:, :, ctx_start:ctx_end, :, :]
            
            # Decode
            try:
                with torch.no_grad():
                    if enable_tiling and hasattr(vae, 'decode_tiled'):
                        decoded_raw = vae.decode_tiled(latent_chunk, tile_x=tile_size, tile_y=tile_size)
                    else:
                        decoded_raw = vae.decode(latent_chunk)
                    decoded_chunk = self._extract_tensor_output(decoded_raw).cpu()
                
                oom_retry = 0
            
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom_retry += 1
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
                raise
            
            decoded_chunk = self._normalize_output(decoded_chunk, aspect_ratio)
            
            # Extract valid
            front_trim = (start_idx - ctx_start) * time_scale
            if end_idx == total_frames:
                valid_frames = decoded_chunk[front_trim:]
            else:
                core_len = (end_idx - start_idx) * time_scale
                valid_frames = decoded_chunk[front_trim:front_trim + core_len]
            
            valid_frames = self._center_crop_to_reference(valid_frames, output_h, output_w)
            
            # ===== COLOR CORRECTION =====
            if valid_frames.shape[0] > 0:
                is_first = (frames_processed == 0)
                
                if self._correction_mode == "ema":
                    valid_frames, ema_stats = self._apply_ema_soft(
                        valid_frames, ema_stats, prev_stats,
                        overlap_frames * time_scale, verbose, is_first
                    )
                    prev_stats = self._extract_stats(valid_frames, overlap_frames * time_scale)
                
                elif self._correction_mode == "temporal":
                    if prev_stats and overlap_frames > 0:
                        valid_frames = self._apply_temporal(
                            valid_frames, prev_stats, overlap_frames * time_scale
                        )
                    prev_stats = self._extract_stats(valid_frames, overlap_frames * time_scale)
                
                elif self._correction_mode == "reference" and is_first:
                    if verbose:
                        logger.info("🎨 Reference anchor")
                    valid_frames = self._apply_reference(valid_frames, self._calibration_stats)
            
            # Write
            for frame in valid_frames:
                preview = writer.write_frame(frame)
                if preview is not None:
                    preview_frames.append(torch.from_numpy(preview).float() / 255.0)
                    if len(preview_frames) > 5:
                        preview_frames.pop(0)
            
            frames_processed += end_idx - start_idx
            pbar.update(end_idx - start_idx)
            
            if verbose:
                pct = (writer.frames_written / expected_frames) * 100
                logger.info(f"Progress: {writer.frames_written}/{expected_frames} ({pct:.1f}%)")
            
            start_idx = end_idx
            
            del latent_chunk, decoded_chunk, valid_frames
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Finalize
        final_path = writer.finalize(audio_path if audio_path else None)
        
        if verbose:
            logger.info(f"✅ Done: {final_path}")
            logger.info(f"   Frames: {writer.frames_written}, Size: {os.path.getsize(final_path)/(1024**2):.2f}MB")
        
        preview_tensor = torch.stack(preview_frames) if preview_frames else torch.empty(0)
        return (preview_tensor, final_path)
    
    # ===== HELPER METHODS =====
    
    def _extract_reference_image_stats(self, img) -> Optional[Dict]:
        """Extract stats from reference image."""
        if img is None:
            return None
        
        if isinstance(img, (list, tuple)):
            if not img:
                return None
            img = img[0]
        
        if not isinstance(img, torch.Tensor):
            return None
        
        img = img.detach().float().cpu()
        
        # Normalize to [B,H,W,C]
        if img.dim() == 4:
            if img.shape[-1] in (3,4):
                img = img[..., :3]
            elif img.shape[1] in (3,4):
                img = img.permute(0,2,3,1)[..., :3]
            else:
                return None
        elif img.dim() == 3:
            if img.shape[-1] in (3,4):
                img = img[..., :3].unsqueeze(0)
            elif img.shape[0] in (3,4):
                img = img.permute(1,2,0)[..., :3].unsqueeze(0)
            else:
                return None
        else:
            return None
        
        if img.min().item() < 0:
            img = (img + 1.0) / 2.0
        img = torch.clamp(torch.nan_to_num(img), 0.0, 1.0)
        
        stats = {"mean": [], "std": []}
        for c in range(3):
            stats["mean"].append(img[..., c].mean().item())
            stats["std"].append(max(img[..., c].std().item(), 1e-6))
        
        return stats
    
    def _extract_stats(self, frames: torch.Tensor, overlap: int) -> Dict:
        """Extract from end of chunk."""
        if frames.shape[0] < overlap:
            overlap = frames.shape[0]
        if overlap == 0:
            return {'mean': [0.5]*3, 'std': [0.1]*3}
        
        zone = frames[-overlap:]
        stats = {'mean': [], 'std': []}
        for c in range(3):
            stats['mean'].append(zone[..., c].mean().item())
            stats['std'].append(max(zone[..., c].std().item(), 1e-6))
        return stats
    
    def _apply_temporal(self, frames: torch.Tensor, prev: Dict, overlap: int) -> torch.Tensor:
        """Temporal overlap correction."""
        if overlap == 0 or frames.shape[0] < overlap:
            return frames
        
        corr = frames.clone()
        zone = corr[:overlap]
        
        for c in range(3):
            curr_m = zone[..., c].mean()
            curr_s = max(zone[..., c].std().item(), 1e-6)
            tgt_m = prev['mean'][c]
            tgt_s = prev['std'][c]
            
            norm = (zone[..., c] - curr_m) * (tgt_s / curr_s) + tgt_m
            zone[..., c] = torch.clamp(norm, 0.0, 1.0)
        
        corr[:overlap] = zone
        return corr
    
    def _apply_reference(self, frames: torch.Tensor, ref: Dict) -> torch.Tensor:
        """Reference calibration."""
        if frames.shape[0] == 0:
            return frames
        
        calib = frames.clone()
        for c in range(3):
            curr_m = calib[..., c].mean()
            curr_s = max(calib[..., c].std().item(), 1e-6)
            tgt_m = ref["mean"][c]
            tgt_s = ref["std"][c]
            
            norm = (calib[..., c] - curr_m) * (tgt_s / curr_s) + tgt_m
            calib[..., c] = torch.clamp(norm, 0.0, 1.0)
        
        return calib
    
    def _apply_ema_soft(self, frames: torch.Tensor, ema: Optional[Dict],
                       prev: Optional[Dict], overlap: int,
                       verbose: bool, is_first: bool) -> tuple:
        """
        EMA with SOFT BLENDING (flicker fix).
        
        Key change: Instead of 100% snap to EMA target,
        blend original with corrected (strength param).
        """
        
        # Init EMA from reference
        if ema is None:
            if self._calibration_stats:
                ema = {
                    'mean': self._calibration_stats['mean'].copy(),
                    'std': self._calibration_stats['std'].copy()
                }
                if verbose:
                    logger.info("🎬 EMA from reference")
            else:
                ema = self._extract_stats(frames, len(frames))
        
        # Apply soft correction to overlap
        if not is_first and prev and overlap > 0:
            if overlap > frames.shape[0]:
                overlap = frames.shape[0]
            
            original = frames.clone()
            corrected = frames.clone()
            zone = corrected[:overlap]
            
            for c in range(3):
                curr_m = zone[..., c].mean()
                curr_s = max(zone[..., c].std().item(), 1e-6)
                
                tgt_m = ema['mean'][c]
                tgt_s = ema['std'][c]
                
                # Full correction
                norm = (zone[..., c] - curr_m) * (tgt_s / curr_s) + tgt_m
                zone[..., c] = torch.clamp(norm, 0.0, 1.0)
            
            corrected[:overlap] = zone
            
            # ===== KEY FIX: SOFT BLEND =====
            strength = self._correction_strength
            frames = (1 - strength) * original + strength * corrected
            # This eliminates hard snap → no flicker!
        
        # Update EMA
        curr = self._extract_stats(frames, overlap if overlap > 0 else len(frames))
        
        mom = self._ema_momentum
        for c in range(3):
            ema['mean'][c] = mom * ema['mean'][c] + (1 - mom) * curr['mean'][c]
            ema['std'][c] = mom * ema['std'][c] + (1 - mom) * curr['std'][c]
        
        if verbose and not is_first:
            shift = sum(abs(ema['mean'][c] - curr['mean'][c]) for c in range(3)) / 3
            if shift > 0.03:
                logger.info(f"   🎬 EMA shift: {shift:.3f}")
        
        return frames, ema


NODE_CLASS_MAPPINGS = {
    "SmartVAE_AdvancedDecoder": SmartVAEAdvancedDecoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartVAE_AdvancedDecoder": "🎞️ SmartVAE Advanced Decoder (🧪 v2.1 Flicker Fix)",
}
