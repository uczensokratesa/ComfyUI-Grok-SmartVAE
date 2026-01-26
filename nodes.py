""" Universal Smart VAE Decode - Audio Sync God Mode
Fixes Fencepost Error for perfect audio sync. Integrates Gemini math fix with Grok resilience.
Author: Grok (xAI) with credits to Gemini (math), Claude (stability), GPT (structure)
Version: 6.0.0
License: MIT
GitHub: https://github.com/uczensokratesa/ComfyUI-Grok-SmartVAE

Key Fix: Middle chunks use (end-start)*scale, last takes full remainder.
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
                "frames_per_batch": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),
            },
            "optional": {
                "overlap_frames": ("INT", {"default": 2, "min": 0, "max": 16, "step": 1}),
                "force_time_scale": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}),
                "enable_tiling": ("BOOLEAN", {"default": False}),
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "debug_level": ("INT", {"default": 1, "min": 0, "max": 2, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "latent/video"

    def __init__(self):
        self.cached_vae_id = None
        self.cached_time_scale = None
        self.cached_force_scale = None
        self.cached_spatial_scale = None

    def _get_available_vram(self):
        try:
            if not torch.cuda.is_available():
                return None
            device = torch.cuda.current_device()
            free_vram, _ = torch.cuda.mem_get_info(device)
            return free_vram / (1024 ** 3)
        except:
            return None

    def detect_scales(self, vae, latents, force_time=0, debug_level=1):
        vae_id = id(vae)
        if force_time > 0:
            if self.cached_force_scale != force_time:
                self.cached_time_scale = None
                self.cached_force_scale = force_time
            if debug_level > 0:
                print(f"🔧 Forced time: {force_time}x")
            self.cached_time_scale = force_time
            time_scale = force_time
        else:
            self.cached_force_scale = None
            if vae_id == self.cached_vae_id and self.cached_time_scale is not None:
                time_scale = self.cached_time_scale
            else:
                if hasattr(vae, 'downscale_index_formula') and vae.downscale_index_formula:
                    try:
                        time_scale = int(vae.downscale_index_formula[0])
                        if debug_level > 0:
                            print(f"🔍 Metadata time: {time_scale}x")
                    except:
                        pass
                else:
                    try:
                        test_sample = latents[:, :, 0:5, :16, :16]
                        with torch.no_grad():
                            test_out = vae.decode(test_sample)
                        test_out = self._normalize_output(test_out)
                        output_frames = test_out.shape[0]
                        time_scale = max(1, (output_frames - 1) // 4)
                        if debug_level > 0:
                            print(f"🔍 Detected time: {time_scale}x (5→{output_frames})")
                    except Exception as e:
                        if debug_level > 0:
                            print(f"⚠️ Time detect failed: {e}. Default 1x.")
                        time_scale = 1
                self.cached_vae_id = vae_id
                self.cached_time_scale = time_scale
        
        if self.cached_spatial_scale is None:
            try:
                test_sample = latents[:, :, 0:1, :, :]
                with torch.no_grad():
                    test_out = vae.decode(test_sample)
                test_out = self._normalize_output(test_out)
                h_out, w_out = test_out.shape[1:3]
                h_in, w_in = latents.shape[3:5]
                spatial_scale = h_out // h_in
                if debug_level > 0:
                    print(f"🔍 Detected spatial: {spatial_scale}x")
                self.cached_spatial_scale = spatial_scale
            except:
                self.cached_spatial_scale = 8
        spatial_scale = self.cached_spatial_scale
        
        return time_scale, spatial_scale

    def _estimate_chunk_vram(self, frames, channels, h, w, time_scale=1, spatial_scale=8):
        latent_bytes = frames * channels * h * w * 4
        output_frames = frames * time_scale  # Simplified for mid chunks
        output_bytes = output_frames * 3 * (h * spatial_scale) * (w * spatial_scale) * 4
        total_bytes = (latent_bytes + output_bytes) * 3.5 * 1.1
        return total_bytes / (1024 ** 3)

    def _normalize_output(self, tensor):
        if isinstance(tensor, (list, tuple)):
            if not tensor or len(tensor) == 0:
                raise ValueError("VAE returned empty output")
            tensor = tensor[0]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected Tensor, got {type(tensor)}")
        
        if tensor.dim() == 5:
            if tensor.shape[1] in [3, 4]:
                tensor = tensor.permute(0, 2, 3, 4, 1)
            b, f, h, w, c = tensor.shape
            tensor = tensor.reshape(b * f, h, w, c)
        elif tensor.dim() == 4:
            if tensor.shape[1] in [3, 4]:
                tensor = tensor.permute(0, 2, 3, 1)
        
        tensor = torch.clamp(tensor, 0.0, 1.0)
        if tensor.shape[-1] > 3:
            tensor = tensor[..., :3]
        return tensor.contiguous()

    def _center_crop_to_reference(self, tensor, h_ref, w_ref):
        _, h, w, _ = tensor.shape
        if h == h_ref and w == w_ref:
            return tensor
        h_offset = max(0, (h - h_ref) // 2)
        w_offset = max(0, (w - w_ref) // 2)
        return tensor[:, h_offset:h_offset + h_ref, w_offset:w_offset + w_ref, :]

    def decode(self, vae, samples, frames_per_batch, overlap_frames=2, force_time_scale=0, 
               enable_tiling=False, tile_size=512, debug_level=1):
        latents = samples["samples"]
        
        if latents.dim() == 4:
            if debug_level > 0:
                print("🖼️ Image decode")
            with torch.no_grad():
                output = vae.decode_tiled(latents, tile_x=tile_size, tile_y=tile_size) if enable_tiling else vae.decode(latents)
            return (self._normalize_output(output),)
        
        batch, channels, total_frames, h_latent, w_latent = latents.shape
        frames_per_batch = max(1, min(frames_per_batch, total_frames))
        overlap_frames = max(0, min(overlap_frames, frames_per_batch - 1))
        initial_batch = frames_per_batch
        initial_tile = tile_size
        fixed_overlap = overlap_frames
        
        time_scale, spatial_scale = self.detect_scales(vae, latents, force_time_scale, debug_level)
        expected_frames = 1 + (total_frames - 1) * time_scale
        cumulative_expected = 0
        cumulative_actual = 0
        chunk_count = 0
        
        if debug_level > 0:
            print(f"🎬 Video: {total_frames} latents → ~{expected_frames} frames (Time: {time_scale}x, Spatial: {spatial_scale}x)")
        
        available_vram = self._get_available_vram()
        if available_vram:
            est_vram = self._estimate_chunk_vram(frames_per_batch + 2*fixed_overlap, channels, h_latent, w_latent, time_scale, spatial_scale)
            if est_vram > available_vram * 0.6:
                frames_per_batch = max(1, int(frames_per_batch * (available_vram * 0.5 / est_vram)))
                if debug_level > 0:
                    print(f"📉 Preemptive: Batch → {frames_per_batch}")
        
        output_chunks = []
        h_reference = None
        w_reference = None
        current_batch = frames_per_batch
        start_idx = 0
        
        pbar = comfy.utils.ProgressBar(total_frames)
        
        while start_idx < total_frames:
            throw_exception_if_processing_interrupted()
            
            end_idx = min(start_idx + current_batch, total_frames)
            ctx_start = max(0, start_idx - fixed_overlap)
            ctx_end = min(total_frames, end_idx + fixed_overlap)
            
            latent_chunk = latents[:, :, ctx_start:ctx_end, :, :]
            
            try:
                with torch.no_grad():
                    if enable_tiling and hasattr(vae, 'decode_tiled'):
                        decoded_chunk = vae.decode_tiled(latent_chunk, tile_x=tile_size, tile_y=tile_size)
                    else:
                        decoded_chunk = vae.decode(latent_chunk)
            
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    gc.collect()
                    if debug_level > 0:
                        print(f"⚠️ OOM at {start_idx}")
                    
                    if not enable_tiling:
                        if debug_level > 0:
                            print("  → Tiling on")
                        enable_tiling = True
                        continue
                    
                    if current_batch > 1:
                        current_batch = max(1, current_batch // 2)
                        if debug_level > 0:
                            print(f"  → Batch → {current_batch}")
                        continue
                    
                    if tile_size > 256:
                        tile_size = max(256, tile_size // 2)
                        if debug_level > 0:
                            print(f"  → Tile → {tile_size}px")
                        continue
                    
                    min_vram = self._estimate_chunk_vram(1, channels, h_latent, w_latent, time_scale, spatial_scale)
                    raise RuntimeError(
                        f"Persistent OOM. Min VRAM: ~{min_vram:.1f}GB\nSuggestions: Close apps, lower res, segment, upgrade GPU."
                    ) from e
                else:
                    raise
            
            decoded_chunk = self._normalize_output(decoded_chunk)
            
            front_trim = (start_idx - ctx_start) * time_scale
            if end_idx == total_frames:
                valid_frames = decoded_chunk[front_trim:]
            else:
                core_length = (end_idx - start_idx) * time_scale
                valid_frames = decoded_chunk[front_trim:front_trim + core_length]
            
            actual_chunk = valid_frames.shape[0]
            expected_chunk = (end_idx - start_idx) * time_scale if end_idx < total_frames else expected_frames - cumulative_expected
            diff = abs(actual_chunk - expected_chunk)
            if diff > 0 and debug_level > 0:
                print(f"⚠️ Chunk {start_idx}: Diff {diff} frames (expected {expected_chunk}, got {actual_chunk})")
            
            cumulative_expected += expected_chunk
            cumulative_actual += actual_chunk
            chunk_count += 1
            
            if h_reference is None:
                h_reference, w_reference = valid_frames.shape[1:3]
            else:
                valid_frames = self._center_crop_to_reference(valid_frames, h_reference, w_reference)
            
            output_chunks.append(valid_frames)
            
            pbar.update(end_idx - start_idx)
            
            del latent_chunk, decoded_chunk
            
            if current_batch <= 4 or chunk_count % max(2, current_batch // 2) == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if available_vram and chunk_count % 5 == 0:
                est_vram = self._estimate_chunk_vram(current_batch + 2*fixed_overlap, channels, h_latent, w_latent, time_scale, spatial_scale)
                if est_vram < available_vram * 0.6:
                    if current_batch < initial_batch:
                        current_batch = min(initial_batch, current_batch + (current_batch // 2))
                        if debug_level > 1:
                            print(f"📈 Safe restore: Batch → {current_batch}")
                    if tile_size < initial_tile:
                        tile_size = min(initial_tile, tile_size + (tile_size // 2))
                        if debug_level > 1:
                            print(f"📈 Safe restore: Tile → {tile_size}px")
        
        final_output = torch.cat(output_chunks, dim=0)
        actual_frames = final_output.shape[0]
        if debug_level > 0:
            print(f"✅ Done: {actual_frames} frames (Chunks: {chunk_count}, Avg/chunk: {actual_frames / chunk_count:.1f})")
        
        frame_diff = abs(actual_frames - expected_frames)
        if frame_diff > 0:
            print(f"⚠️ Final diff {frame_diff}. Minor rounding – audio should be close. If desync, try force scale.")
        
        return (final_output,)

NODE_CLASS_MAPPINGS = {"UniversalSmartVAEDecode": UniversalSmartVAEDecode}
NODE_DISPLAY_NAME_MAPPINGS = {"UniversalSmartVAEDecode": "🎬 Universal VAE Decode (Audio Sync v6.0)"}
