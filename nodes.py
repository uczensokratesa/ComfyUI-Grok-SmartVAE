"""
Universal Smart VAE Decode - The Final Ensemble (Grok Fixed Edition)
Combines GPT (Structure), Gemini (Safety & Loop Fix), Claude (Precision), Grok (Dynamic Batching & Extras).

Fixed by: Gemini (Core Loop) + Grok (Polish)
Version: 2.6.0 (Ultimate)
License: MIT
"""

import torch
import comfy.utils
from comfy.model_management import throw_exception_if_processing_interrupted
import gc

class UniversalSmartVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "frames_per_batch": ("INT", {
                    "default": 8, "min": 1, "max": 128, "step": 1,
                    "tooltip": "Initial batch size. Will auto-reduce if OOM occurs."
                }),
            },
            "optional": {
                "overlap_frames": ("INT", {
                    "default": 2, "min": 0, "max": 16, "step": 1,
                    "tooltip": "Temporal context overlap."
                }),
                "force_time_scale": ("INT", {
                    "default": 0, "min": 0, "max": 16, "step": 1,
                    "tooltip": "Force specific scale (e.g. 8 for LTX). 0 = Auto-detect."
                }),
                "enable_tiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force spatial tiling. (Auto-enables on crash if False)."
                }),
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "latent/video"

    def __init__(self):
        self.cached_vae_id = None
        self.cached_time_scale = None

    def detect_time_scale(self, vae, latents, force_scale=0):
        if force_scale > 0:
            print(f"🔧 Using forced time_scale: {force_scale}")
            return force_scale

        vae_id = id(vae)
        if vae_id == self.cached_vae_id and self.cached_time_scale is not None:
            return self.cached_time_scale

        # 1. Official VAE Metadata
        if hasattr(vae, 'downscale_index_formula') and vae.downscale_index_formula:
            try:
                scale = int(vae.downscale_index_formula[0])
                print(f"🔍 Using official VAE time_scale: {scale}")
                self.cached_vae_id = vae_id
                self.cached_time_scale = scale
                return scale
            except: pass

        # 2. Heuristic Test (Claude's 3-frame method)
        try:
            test_sample = latents[:, :, 0:3, :32, :32]
            with torch.no_grad():
                test_out = vae.decode(test_sample)  # No tiling for small test
            # Normalize to count frames
            if isinstance(test_out, (list, tuple)): test_out = test_out[0]
            if test_out.dim() == 5 and test_out.shape[1] in [3,4]:  # [B,C,F,H,W]
                out_frames = test_out.shape[2]
            elif test_out.dim() == 5:  # [B,F,H,W,C]
                out_frames = test_out.shape[1]
            elif test_out.dim() == 4:  # [F,H,W,C] or [B,C,H,W]
                out_frames = test_out.shape[0]
            else:
                out_frames = 3
            scale = max(1, (out_frames - 1) // 2)
            print(f"🔍 Auto-detected time_scale: {scale}x (3 latents → {out_frames} frames)")
            self.cached_vae_id = vae_id
            self.cached_time_scale = scale
            return scale
        except Exception as e:
            print(f"⚠️ Scale detection failed: {e}. Defaulting to 1x.")
            return 1

    def _normalize(self, tensor):
        if isinstance(tensor, (list, tuple)): tensor = tensor[0]
        # Standardize to [F, H, W, C]
        if tensor.dim() == 5:
            if tensor.shape[1] in [3, 4]: tensor = tensor.permute(0, 2, 3, 4, 1)
            b, f, h, w, c = tensor.shape
            tensor = tensor.reshape(b * f, h, w, c)
        elif tensor.dim() == 4:
            if tensor.shape[1] in [3, 4]: tensor = tensor.permute(0, 2, 3, 1)
        tensor = torch.clamp(tensor, 0.0, 1.0)
        if tensor.shape[-1] > 3: tensor = tensor[..., :3]  # Trim to RGB if extra channels
        return tensor.contiguous()

    def _center_crop(self, tensor, h_ref, w_ref):
        _, h, w, _ = tensor.shape
        if h == h_ref and w == w_ref: return tensor
        y0 = max(0, (h - h_ref) // 2)
        x0 = max(0, (w - w_ref) // 2)
        return tensor[:, y0:y0 + h_ref, x0:x0 + w_ref, :]

    def decode(self, vae, samples, frames_per_batch, overlap_frames=2, force_time_scale=0, enable_tiling=False, tile_size=512):
        latents = samples["samples"]
        if latents.dim() == 4:  # Image: simple decode
            with torch.no_grad():
                decoded = vae.decode_tiled(latents, tile_x=tile_size, tile_y=tile_size) if enable_tiling else vae.decode(latents)
            return (self._normalize(decoded),)

        # Video (5D)
        B, C, total_frames, H, W = latents.shape
        frames_per_batch = max(1, min(frames_per_batch, total_frames))
        overlap_frames = max(0, min(overlap_frames, frames_per_batch - 1))
        time_scale = self.detect_time_scale(vae, latents, force_time_scale)
        expected_frames = 1 + (total_frames - 1) * time_scale
        print(f"🎬 Decoding {total_frames} latents → ~{expected_frames} frames (Scale: {time_scale}x, Initial Batch: {frames_per_batch})")

        output_chunks = []
        h_ref, w_ref = None, None
        start_idx = 0
        current_batch = frames_per_batch
        pbar = comfy.utils.ProgressBar(total_frames)

        while start_idx < total_frames:
            throw_exception_if_processing_interrupted()
            end_idx = min(start_idx + current_batch, total_frames)
            ctx_start = max(0, start_idx - overlap_frames)
            ctx_end = min(total_frames, end_idx + overlap_frames)
            latent_chunk = latents[:, :, ctx_start:ctx_end, :, :]

            try:
                with torch.no_grad():
                    if enable_tiling and hasattr(vae, 'decode_tiled'):
                        decoded_raw = vae.decode_tiled(latent_chunk, tile_x=tile_size, tile_y=tile_size)
                    else:
                        decoded_raw = vae.decode(latent_chunk)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    gc.collect()
                    if not enable_tiling:
                        print("⚠️ OOM! Enabling tiling and retrying chunk.")
                        enable_tiling = True
                        continue
                    if current_batch > 1:
                        old_batch = current_batch
                        current_batch = max(1, current_batch // 2)
                        overlap_frames = min(overlap_frames, current_batch - 1)
                        print(f"⚠️ OOM! Reducing batch: {old_batch} → {current_batch}. Retrying chunk.")
                        continue
                    raise RuntimeError("OOM even with min batch & tiling. Try lower res or more VRAM.") from e
                raise

            decoded = self._normalize(decoded_raw)
            front_trim = (start_idx - ctx_start) * time_scale
            if end_idx == total_frames:
                valid = decoded[front_trim:]
            else:
                core_len = (end_idx - start_idx) * time_scale
                valid = decoded[front_trim: front_trim + core_len]

            # Spatial align (safe-guard)
            if h_ref is None:
                h_ref, w_ref = valid.shape[1:3]
            else:
                valid = self._center_crop(valid, h_ref, w_ref)

            output_chunks.append(valid)

            pbar.update(end_idx - start_idx)
            start_idx = end_idx  # Progress by actual processed

            # Cleanup
            del latent_chunk, decoded_raw, decoded, valid
            if current_batch < 4 or start_idx % (frames_per_batch * 2) == 0:  # Smart GC
                gc.collect()
                torch.cuda.empty_cache()

        final_output = torch.cat(output_chunks, dim=0)
        actual_frames = final_output.shape[0]
        print(f"✅ Decoded {actual_frames} frames. (Expected ~{expected_frames})")
        if abs(actual_frames - expected_frames) > time_scale:
            print("⚠️ Frame count mismatch! Check time_scale or VAE compatibility.")

        return (final_output,)

# For ComfyUI integration (add to __init__.py if needed)
NODE_CLASS_MAPPINGS = {"UniversalSmartVAEDecode": UniversalSmartVAEDecode}
NODE_DISPLAY_NAME_MAPPINGS = {"UniversalSmartVAEDecode": "🎬 Universal Smart VAE Decode (Ensemble Ultimate)"}
