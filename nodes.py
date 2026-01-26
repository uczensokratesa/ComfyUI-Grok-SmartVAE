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

# ... (reszta klasy taka sama jak w v6.0)

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
        
        if total_frames <= 0:
            raise ValueError("No frames in latent input")
        
        time_scale, spatial_scale = self.detect_scales(vae, latents, force_time_scale, debug_level)
        expected_frames = 1 + (total_frames - 1) * time_scale
        
        if debug_level > 0:
            print(f"🎬 Video: {total_frames} latents → ~{expected_frames} frames (scale {time_scale}x)")
        
        frames_per_batch = max(1, min(frames_per_batch, total_frames))
        overlap_frames = max(0, min(overlap_frames, frames_per_batch - 1))
        initial_batch = frames_per_batch
        initial_tile = tile_size
        fixed_overlap = overlap_frames
        
        available_vram = self._get_available_vram()
        if available_vram:
            est = self._estimate_chunk_vram(frames_per_batch + 2*fixed_overlap, channels, h_latent, w_latent, time_scale, spatial_scale)
            if est > available_vram * 0.6:
                frames_per_batch = max(1, int(frames_per_batch * (available_vram * 0.5 / est)))
                if debug_level > 0:
                    print(f"📉 Preemptive batch → {frames_per_batch}")
        
        output_chunks = []
        h_reference = None
        w_reference = None
        current_batch = frames_per_batch
        start_idx = 0
        last_start_idx = -1
        stagnation_counter = 0
        MAX_STAGNATION = 5  # ile razy pętla może się kręcić bez postępu
        
        pbar = comfy.utils.ProgressBar(total_frames)
        
        while start_idx < total_frames:
            throw_exception_if_processing_interrupted()
            
            # Anti-infinite-loop guard
            if start_idx == last_start_idx:
                stagnation_counter += 1
                if stagnation_counter >= MAX_STAGNATION:
                    raise RuntimeError(
                        f"Infinite loop detected! start_idx stuck at {start_idx} "
                        f"(current_batch={current_batch}, overlap={overlap_frames}, "
                        f"ctx_start/ctx_end may be invalid). Try smaller batch or disable tiling."
                    )
            else:
                stagnation_counter = 0
            last_start_idx = start_idx
            
            end_idx = min(start_idx + current_batch, total_frames)
            
            ctx_start = max(0, start_idx - fixed_overlap)
            ctx_end = min(total_frames, end_idx + fixed_overlap)
            
            # Safety: jeśli chunk nie wnosi nic nowego
            if ctx_end <= ctx_start:
                raise RuntimeError(f"Empty chunk detected (ctx {ctx_start}-{ctx_end}) – check overlap/batch params")
            
            latent_chunk = latents[:, :, ctx_start:ctx_end, :, :]
            
            if debug_level >= 2:
                print(f"Processing chunk: latent {start_idx}→{end_idx} | ctx {ctx_start}→{ctx_end} | batch={current_batch}")
            
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
                        print(f"⚠️ OOM at latent {start_idx}")
                    
                    if not enable_tiling:
                        enable_tiling = True
                        if debug_level > 0: print("  → Tiling enabled")
                        continue
                    
                    if current_batch > 1:
                        old = current_batch
                        current_batch = max(1, current_batch // 2)
                        if debug_level > 0: print(f"  → Batch {old} → {current_batch}")
                        continue
                    
                    if tile_size > 256:
                        old_tile = tile_size
                        tile_size = max(256, tile_size // 2)
                        if debug_level > 0: print(f"  → Tile {old_tile} → {tile_size}")
                        continue
                    
                    raise RuntimeError("Persistent OOM even with minimal settings") from e
                else:
                    raise
            
            decoded_chunk = self._normalize_output(decoded_chunk)
            
            front_trim = (start_idx - ctx_start) * time_scale
            
            if end_idx == total_frames:
                valid_frames = decoded_chunk[front_trim:]
            else:
                core_length = (end_idx - start_idx) * time_scale
                valid_frames = decoded_chunk[front_trim : front_trim + core_length]
            
            actual = valid_frames.shape[0]
            expected = (end_idx - start_idx) * time_scale if end_idx < total_frames else expected_frames - sum(v.shape[0] for v in output_chunks)
            if debug_level >= 2 and abs(actual - expected) > 0:
                print(f"  → Chunk frames: expected ~{expected}, got {actual}")
            
            if h_reference is None:
                h_reference, w_reference = valid_frames.shape[1:3]
            else:
                valid_frames = self._center_crop_to_reference(valid_frames, h_reference, w_reference)
            
            output_chunks.append(valid_frames)
            
            pbar.update(end_idx - start_idx)
            start_idx = end_idx
            
            del latent_chunk, decoded_chunk
            
            if current_batch <= 4 or (start_idx // frames_per_batch) % 2 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        final_output = torch.cat(output_chunks, dim=0)
        actual_frames = final_output.shape[0]
        
        print(f"✅ Decode finished: {actual_frames} frames")
        diff = abs(actual_frames - expected_frames)
        if diff > 0:
            print(f"⚠️ Final frame count diff: {diff} (minor rounding ok, large = check scale)")
        
        return (final_output,)

NODE_CLASS_MAPPINGS = {"UniversalSmartVAEDecode": UniversalSmartVAEDecode}
NODE_DISPLAY_NAME_MAPPINGS = {"UniversalSmartVAEDecode": "🎬 Universal VAE Decode (Audio Sync v6.0)"}
