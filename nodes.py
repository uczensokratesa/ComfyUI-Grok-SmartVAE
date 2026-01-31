"""
Universal Smart VAE Decode - v11.1 Final Production Edition
Grok v11.0 + Claude's refinements for ultimate stability

Key Features:
- Disk offloading for 700+ frame workflows (Kimi's contribution)
- Orientation-safe normalization (Grok's geometry fix)
- Pre-allocated pinned memory for speed
- Progress-based stagnation detection
- Frame-perfect audio sync (Gemini's math)
- Adaptive logging and cleanup

Credits: Grok (disk offload, orientation), Kimi (memory safety), Claude (stability), 
         Gemini (math), GPT (structure)
Version: 11.1.0
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

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available. Install with 'pip install psutil' for automatic RAM detection.")


class UniversalSmartVAEDecode:
    """
    Production-grade VAE decoder with disk offloading for massive workflows.
    Handles 2000+ frames with automatic memory management.
    """
    
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
                    "default": True,
                    "tooltip": "Show detailed progress logs."
                }),
                "max_ram_frames": ("INT", {
                    "default": 500,
                    "min": 100,
                    "max": 10000,
                    "step": 100,
                    "tooltip": "Threshold for disk offload. Adjust based on system RAM (16GB=300, 32GB=600, 64GB=1000)."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "latent/video"

    def __init__(self):
        self.cached_vae_id = None
        self.cached_time_scale = None
        self.cached_force_scale = None
        self.temp_dir = None
        # Register cleanup on exit
        atexit.register(self._cleanup_temp_dir)

    def _cleanup_temp_dir(self):
        """Cleanup temporary files (called on exit or error)."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                for file in os.listdir(self.temp_dir):
                    try:
                        os.remove(os.path.join(self.temp_dir, file))
                    except Exception:
                        pass
                os.rmdir(self.temp_dir)
            except Exception:
                pass  # Ignore cleanup errors

    def __del__(self):
        """Backup cleanup on object destruction."""
        self._cleanup_temp_dir()

    def _get_available_vram(self):
        """Get available VRAM in GB. Returns None on CPU or error."""
        try:
            if not torch.cuda.is_available():
                return None
            free_vram, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
            return free_vram / (1024 ** 3)
        except Exception:
            return None

    def _get_available_ram(self):
        """Get available system RAM in GB. Returns None if psutil unavailable."""
        if PSUTIL_AVAILABLE:
            return psutil.virtual_memory().available / (1024 ** 3)
        return None

    def detect_time_scale(self, vae, latents, force_scale=0, verbose=True):
        """
        Detect temporal upsampling scale.
        Priority: force > cache > metadata > empirical test > fallback
        """
        vae_id = id(vae)
        
        # User override
        if force_scale > 0:
            if self.cached_force_scale != force_scale:
                self.cached_time_scale = None
                self.cached_force_scale = force_scale
            if verbose:
                print(f"🔧 Forced time scale: {force_scale}x")
            self.cached_time_scale = force_scale
            return force_scale
        
        self.cached_force_scale = None
        
        # Cache check
        if vae_id == self.cached_vae_id and self.cached_time_scale is not None:
            return self.cached_time_scale
        
        # VAE metadata
        if hasattr(vae, 'downscale_index_formula') and vae.downscale_index_formula:
            try:
                time_scale = int(vae.downscale_index_formula[0])
                if verbose:
                    print(f"🔍 VAE metadata time scale: {time_scale}x")
                self.cached_vae_id = vae_id
                self.cached_time_scale = time_scale
                return time_scale
            except Exception:
                pass
        
        # Empirical test (5 frames for accuracy)
        try:
            test_frames = min(5, latents.shape[2])
            test_sample = latents[:, :, 0:test_frames, :16, :16]
            
            with torch.no_grad():
                test_output = vae.decode(test_sample)
            
            test_output = self._normalize_output(test_output, aspect_ratio=1.0)
            output_frames = test_output.shape[0]
            
            # Formula: output = 1 + (input - 1) * scale
            time_scale = max(1, (output_frames - 1) // (test_frames - 1))
            
            if verbose:
                print(f"🔍 Auto-detected time scale: {time_scale}x ({test_frames}→{output_frames} frames)")
            
            self.cached_vae_id = vae_id
            self.cached_time_scale = time_scale
            
            # Cleanup
            del test_output, test_sample
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return time_scale
        
        except Exception as e:
            if verbose:
                print(f"⚠️  Time scale detection failed: {e}")
                print(f"   Using safe fallback: 1x")
            self.cached_vae_id = vae_id
            self.cached_time_scale = 1
            return 1

    def detect_output_size(self, vae, latents, h_latent, w_latent, tile_size=512, verbose=True):
        """
        Detect exact output H/W from a single full-spatial frame decode.
        Falls back to tiled decode if non-tiled OOMs.
        """
        aspect_ratio = h_latent / float(w_latent)
        test_sample = latents[:, :, 0:1, :, :]  # One frame, full spatial
        
        try:
            with torch.no_grad():
                test_out = vae.decode(test_sample)
            if verbose:
                print("🔍 Output size detected (standard decode)")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                try:
                    with torch.no_grad():
                        test_out = vae.decode_tiled(test_sample, tile_x=tile_size, tile_y=tile_size)
                    if verbose:
                        print("🔍 Output size detected (tiled decode - OOM fallback)")
                except Exception as inner_e:
                    raise RuntimeError("Failed to detect output size even with tiling.") from inner_e
            else:
                raise
        
        test_out = self._normalize_output(test_out, aspect_ratio)
        output_h, output_w = test_out.shape[1:3]
        
        # Cleanup
        del test_out, test_sample
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return output_h, output_w

    def _estimate_chunk_vram(self, frames, channels, h, w, time_scale=1):
        """
        Conservative VRAM estimate assuming standard 8x spatial upscaling.
        Returns estimated GB needed for chunk decode.
        """
        spatial_scale = 8
        latent_bytes = frames * channels * h * w * 4
        output_frames = frames * time_scale
        output_bytes = output_frames * 3 * (h * spatial_scale) * (w * spatial_scale) * 4
        # 3.5x multiplier for activations + 10% safety margin
        total_bytes = (latent_bytes + output_bytes) * 3.5 * 1.1
        return total_bytes / (1024 ** 3)

    def _estimate_output_ram(self, expected_frames, output_h, output_w):
        """
        Estimate total RAM needed for output tensor.
        Returns estimated GB.
        """
        bytes_per_frame = output_h * output_w * 3 * 4  # RGB float32
        return (expected_frames * bytes_per_frame) / (1024 ** 3)

    def _normalize_output(self, tensor, aspect_ratio=None):
        """
        Normalize VAE output to [Frames, Height, Width, Channels].
        
        Args:
            tensor: VAE output tensor (4D or 5D)
            aspect_ratio: Latent aspect ratio (h/w) for orientation detection
        
        Returns:
            Normalized tensor in [F, H, W, C] format
        """
        # Handle list/tuple outputs
        if isinstance(tensor, (list, tuple)):
            if not tensor or len(tensor) == 0:
                raise ValueError("VAE returned empty output")
            tensor = tensor[0]
        
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected Tensor, got {type(tensor)}")
        
        # Ensure float32
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        
        dim = tensor.dim()
        
        if dim == 4:
            # [N, C, H, W] or [N, H, W, C]
            if tensor.shape[1] in (3, 4):
                tensor = tensor.permute(0, 2, 3, 1)  # → [N, H, W, C]
        
        elif dim == 5:
            shape = list(tensor.shape)
            
            # Squeeze batch dimension if B=1
            if shape[0] == 1:
                tensor = tensor.squeeze(0)
                shape = list(tensor.shape)
            
            # Find channel dimension (3 or 4 for RGB/RGBA)
            try:
                c_idx = next(i for i, s in enumerate(shape) if s in (3, 4))
            except StopIteration:
                raise ValueError(f"Cannot find channel dimension (3 or 4) in shape {shape}")
            
            # Remaining indices are [F, H, W] in some order
            remaining_idxs = [i for i in range(4) if i != c_idx]
            remaining_sizes = [shape[i] for i in remaining_idxs]
            
            # Sort by size: smallest is F (temporal), largest two are spatial
            sorted_remaining = sorted(zip(remaining_idxs, remaining_sizes), key=lambda x: x[1])
            f_idx = sorted_remaining[0][0]  # Smallest = frames
            
            # Spatial dimensions
            spatial_large_idx = sorted_remaining[2][0]  # Largest spatial
            spatial_small_idx = sorted_remaining[1][0]  # Second spatial
            
            # Assign H/W based on aspect ratio to prevent rotation
            if aspect_ratio is not None:
                if aspect_ratio > 1.0:
                    # Portrait (H > W): largest spatial = H
                    h_idx = spatial_large_idx
                    w_idx = spatial_small_idx
                else:
                    # Landscape (W > H): largest spatial = W
                    w_idx = spatial_large_idx
                    h_idx = spatial_small_idx
            else:
                # Default: assume portrait
                h_idx = spatial_large_idx
                w_idx = spatial_small_idx
            
            # Permute to [F, H, W, C]
            perm = [f_idx, h_idx, w_idx, c_idx]
            tensor = tensor.permute(*perm)
        
        else:
            raise ValueError(f"Unsupported tensor dimension: {dim}D with shape {tensor.shape}")
        
        # Clamp to valid pixel range
        tensor = torch.clamp(tensor, 0.0, 1.0)
        
        # Convert RGBA to RGB if needed
        if tensor.shape[-1] > 3:
            tensor = tensor[..., :3]
        
        return tensor.contiguous()

    def _center_crop_to_reference(self, tensor, h_ref, w_ref):
        """
        Center crop tensor to match reference dimensions.
        Handles minor size differences from VAE decode rounding.
        """
        _, h, w, _ = tensor.shape
        if h == h_ref and w == w_ref:
            return tensor
        
        h_offset = max(0, (h - h_ref) // 2)
        w_offset = max(0, (w - w_ref) // 2)
        
        return tensor[:, h_offset:h_offset + h_ref, w_offset:w_offset + w_ref, :]

    def decode(self, vae, samples, frames_per_batch, overlap_frames=2, force_time_scale=0, 
               enable_tiling=False, tile_size=512, verbose=True, max_ram_frames=500):
        """
        Main decode function with disk offloading and orientation safety.
        
        Automatically switches between in-memory and disk offload modes based on:
        - Expected frame count vs max_ram_frames threshold
        - Available system RAM vs estimated requirement
        """
        latents = samples["samples"]
        
        # ======== IMAGE PATH (4D) ========
        if latents.dim() == 4:
            if verbose:
                print("🖼️  Image decode")
            
            with torch.no_grad():
                if enable_tiling and hasattr(vae, 'decode_tiled'):
                    output = vae.decode_tiled(latents, tile_x=tile_size, tile_y=tile_size)
                else:
                    output = vae.decode(latents)
            
            return (self._normalize_output(output, aspect_ratio=1.0),)
        
        # ======== VIDEO PATH (5D) ========
        batch, channels, total_frames, h_latent, w_latent = latents.shape
        
        if total_frames <= 0:
            raise ValueError("Latent has no frames")
        
        # Detect temporal scale
        time_scale = self.detect_time_scale(vae, latents, force_time_scale, verbose)
        expected_frames = 1 + (total_frames - 1) * time_scale
        
        # Calculate aspect ratio for orientation detection
        aspect_ratio = h_latent / float(w_latent)
        
        # Detect exact output dimensions
        output_h, output_w = self.detect_output_size(vae, latents, h_latent, w_latent, tile_size, verbose)
        
        # Estimate RAM requirements
        est_ram = self._estimate_output_ram(expected_frames, output_h, output_w)
        available_ram = self._get_available_ram()
        
        # Decide on memory mode
        use_disk_offload = (
            expected_frames > max_ram_frames or 
            (available_ram is not None and est_ram > available_ram * 0.5)
        )
        
        # Setup temp directory if using disk offload
        if use_disk_offload:
            self.temp_dir = tempfile.mkdtemp(prefix="vae_decode_")
            if verbose:
                print(f"💾 Disk offload enabled ({expected_frames} frames, est {est_ram:.2f}GB RAM)")
        
        # Display decode info
        if verbose:
            print(f"🎬 Video decode:")
            print(f"   Input: {total_frames} latent frames")
            print(f"   Time scale: {time_scale}x")
            print(f"   Expected output: ~{expected_frames} frames")
            print(f"   Output resolution: {output_h}x{output_w}")
            print(f"   Estimated RAM: {est_ram:.2f}GB")
            
            if available_ram:
                ram_pct = (est_ram / available_ram) * 100
                print(f"   RAM usage: {ram_pct:.1f}% of {available_ram:.1f}GB available")
            
            orientation = 'portrait' if aspect_ratio < 1.0 else 'landscape'
            print(f"   Aspect ratio: {aspect_ratio:.3f} ({orientation})")
            print(f"   Mode: {'💾 Disk offload' if use_disk_offload else '🚀 In-memory (pinned)'}")
            
            if est_ram > 16:
                warnings.warn(
                    f"High RAM usage ({est_ram:.1f}GB). "
                    f"Consider lowering resolution or increasing max_ram_frames threshold."
                )
        
        # Parameter validation
        frames_per_batch = max(1, min(frames_per_batch, total_frames))
        overlap_frames = max(0, min(overlap_frames, frames_per_batch - 1))
        initial_overlap = overlap_frames
        
        # ======== PREDICTIVE VRAM OPTIMIZATION ========
        available_vram = self._get_available_vram()
        if available_vram is not None:
            chunk_frames = frames_per_batch + 2 * overlap_frames
            est_vram = self._estimate_chunk_vram(chunk_frames, channels, h_latent, w_latent, time_scale)
            
            # Conservative 55% threshold (refined from 50%)
            if est_vram > available_vram * 0.55:
                reduction = (available_vram * 0.45) / est_vram
                old_batch = frames_per_batch
                frames_per_batch = max(1, int(frames_per_batch * reduction))
                overlap_frames = min(overlap_frames, frames_per_batch - 1)
                
                if verbose:
                    print(f"📉 Predictive VRAM reduction:")
                    print(f"   Batch: {old_batch} → {frames_per_batch}")
                    print(f"   Est: {est_vram:.2f}GB / Avail: {available_vram:.2f}GB")
        
        if verbose:
            print(f"   Batch size: {frames_per_batch}")
            print(f"   Overlap: {overlap_frames} frames")
        
        # Initialize storage
        temp_files = [] if use_disk_offload else None
        final_output = None
        current_write_idx = 0
        
        # Try pre-allocation for in-memory mode
        if not use_disk_offload:
            try:
                final_output = torch.empty(
                    expected_frames, output_h, output_w, 3,
                    dtype=torch.float32,
                    device='cpu',
                    pin_memory=torch.cuda.is_available()
                )
                if verbose:
                    print("   ✓ Pre-allocated pinned memory")
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Pre-allocation failed: {e}")
                    print(f"   → Falling back to disk offload")
                use_disk_offload = True
                self.temp_dir = tempfile.mkdtemp(prefix="vae_decode_")
                temp_files = []
        
        # Processing state
        current_batch = frames_per_batch
        start_idx = 0
        
        # Progress tracking (stagnation detection)
        frames_processed = 0
        last_frames_processed = -1
        stagnation_count = 0
        MAX_STAGNATION = 3
        
        # Cumulative for O(1) validation
        cumulative_output_frames = 0
        
        pbar = comfy.utils.ProgressBar(total_frames)
        
        # Adaptive logging
        est_total_chunks = math.ceil(total_frames / frames_per_batch)
        log_every_n = 5 if est_total_chunks > 20 else 1
        chunk_count = 0
        
        # ======== SLIDING WINDOW PROCESSING ========
        while start_idx < total_frames:
            throw_exception_if_processing_interrupted()
            
            # Stagnation guard
            if frames_processed == last_frames_processed:
                stagnation_count += 1
                if stagnation_count >= MAX_STAGNATION:
                    raise RuntimeError(
                        f"Processing stalled at frame {start_idx}/{total_frames}.\n"
                        f"Settings: batch={current_batch}, overlap={overlap_frames}, "
                        f"tiling={enable_tiling}, tile_size={tile_size}\n"
                        f"Try: Lower batch size, enable tiling, or reduce resolution."
                    )
            else:
                stagnation_count = 0
            last_frames_processed = frames_processed
            
            end_idx = min(start_idx + current_batch, total_frames)
            
            # Context window (with overlap)
            ctx_start = max(0, start_idx - overlap_frames)
            ctx_end = min(total_frames, end_idx + overlap_frames)
            
            # Empty chunk guard
            if ctx_end <= ctx_start:
                raise RuntimeError(
                    f"Empty chunk detected (ctx {ctx_start}→{ctx_end}). "
                    f"Check overlap/batch parameters."
                )
            
            latent_chunk = latents[:, :, ctx_start:ctx_end, :, :]
            
            # Adaptive logging
            should_log = verbose and (chunk_count % log_every_n == 0 or end_idx == total_frames)
            if should_log:
                mem_gb = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
                print(f"Processing chunk {chunk_count+1}/{est_total_chunks}: "
                      f"latent {start_idx}→{end_idx} | ctx {ctx_start}→{ctx_end} | "
                      f"VRAM: {mem_gb:.2f}GB")
            
            # Decode with error recovery
            try:
                with torch.no_grad():
                    if enable_tiling and hasattr(vae, 'decode_tiled'):
                        decoded_chunk = vae.decode_tiled(latent_chunk, tile_x=tile_size, tile_y=tile_size)
                    else:
                        decoded_chunk = vae.decode(latent_chunk)
                    
                    # Immediate CPU move to free VRAM
                    decoded_chunk = decoded_chunk.cpu()
            
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if verbose:
                        print(f"   ⚠️  OOM at frame {start_idx}")
                    
                    # Aggressive cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    gc.collect()
                    
                    # Stage 1: Enable tiling
                    if not enable_tiling:
                        if verbose:
                            print(f"   → Stage 1: Enabling tiling")
                        enable_tiling = True
                        # Re-detect output size with tiling
                        output_h, output_w = self.detect_output_size(
                            vae, latents, h_latent, w_latent, tile_size, verbose=False
                        )
                        continue
                    
                    # Stage 2: Reduce batch
                    if current_batch > 1:
                        old_batch = current_batch
                        current_batch = max(1, current_batch // 2)
                        overlap_frames = min(initial_overlap, current_batch - 1)
                        if verbose:
                            print(f"   → Stage 2: Batch {old_batch} → {current_batch}")
                        continue
                    
                    # Stage 3: Reduce tile size
                    if tile_size > 256:
                        old_tile = tile_size
                        tile_size = max(256, tile_size - 128)
                        if verbose:
                            print(f"   → Stage 3: Tile {old_tile} → {tile_size}px")
                        continue
                    
                    # All strategies exhausted
                    min_vram = self._estimate_chunk_vram(1, channels, h_latent, w_latent, time_scale)
                    raise RuntimeError(
                        f"Persistent OOM with minimal settings.\n"
                        f"Minimum VRAM needed: ~{min_vram:.2f}GB\n\n"
                        f"Suggestions:\n"
                        f"  1. Close other GPU applications\n"
                        f"  2. Reduce latent resolution\n"
                        f"  3. Process video in segments"
                    ) from e
                else:
                    raise
            
            # Normalize with orientation awareness
            decoded_chunk = self._normalize_output(decoded_chunk, aspect_ratio)
            
            # ======== TEMPORAL TRIMMING (Audio Sync Fix) ========
            front_trim = (start_idx - ctx_start) * time_scale
            
            if end_idx == total_frames:
                # Last chunk: natural remainder (includes +1 terminal frame)
                valid_frames = decoded_chunk[front_trim:]
            else:
                # Middle chunks: LINEAR mapping (Gemini's fix)
                core_length = (end_idx - start_idx) * time_scale
                valid_frames = decoded_chunk[front_trim:front_trim + core_length]
            
            # Validation
            actual = valid_frames.shape[0]
            if end_idx < total_frames:
                expected = (end_idx - start_idx) * time_scale
            else:
                expected = expected_frames - cumulative_output_frames
            
            if should_log and abs(actual - expected) > 0:
                print(f"   Chunk frames: expected {expected}, got {actual}")
            
            # Spatial alignment
            valid_frames = self._center_crop_to_reference(valid_frames, output_h, output_w)
            
            # Orientation diagnostic (first chunk only)
            if verbose and chunk_count == 0:
                print(f"   ✓ First chunk validated: shape {valid_frames.shape}")
                if (aspect_ratio > 1.0 and valid_frames.shape[2] > valid_frames.shape[1]) or \
                   (aspect_ratio < 1.0 and valid_frames.shape[1] > valid_frames.shape[2]):
                    warnings.warn(
                        "Orientation mismatch detected! Output may be rotated. "
                        "Please report with VAE model name."
                    )
            
            # Store chunk
            if use_disk_offload:
                temp_path = os.path.join(self.temp_dir, f"chunk_{chunk_count:05d}.pt")
                torch.save(valid_frames, temp_path)
                temp_files.append(temp_path)
            else:
                final_output[current_write_idx:current_write_idx + actual] = valid_frames
                current_write_idx += actual
            
            # Update progress
            processed_this_chunk = end_idx - start_idx
            frames_processed += processed_this_chunk
            cumulative_output_frames += actual
            pbar.update(processed_this_chunk)
            start_idx = end_idx
            chunk_count += 1
            
            # Memory management
            del latent_chunk, decoded_chunk, valid_frames
            
            # Adaptive GC
            if chunk_count % 3 == 0 or current_batch <= 4:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
        
        # ======== FINAL ASSEMBLY (Disk Offload Mode) ========
        if use_disk_offload:
            if verbose:
                print(f"🛡️  Assembling {len(temp_files)} chunks from disk...")
            
            chunks = []
            for i, temp_path in enumerate(temp_files):
                chunk = torch.load(temp_path, map_location='cpu')
                chunks.append(chunk)
                
                # Immediate cleanup
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
                
                # Merge every 10 chunks to avoid RAM spike
                if len(chunks) >= 10 or i == len(temp_files) - 1:
                    merged = torch.cat(chunks, dim=0)
                    chunks = [merged]
                    gc.collect()
            
            final_output = chunks[0] if chunks else torch.empty((0, output_h, output_w, 3), dtype=torch.float32)
            
            # Cleanup temp directory
            self._cleanup_temp_dir()
            self.temp_dir = None
        
        # Trim if over-allocated
        if final_output.shape[0] > expected_frames:
            final_output = final_output[:expected_frames]
        
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        # Validation
        actual_frames = final_output.shape[0]
        
        if verbose:
            print(f"✅ Decode complete!")
            print(f"   Output frames: {actual_frames}")
            print(f"   Final shape: {final_output.shape}")
        
        # Audio sync validation
        frame_diff = abs(actual_frames - expected_frames)
        if frame_diff == 0:
            if verbose:
                print(f"   🎵 Audio sync: PERFECT ✓")
        elif frame_diff <= time_scale:
            if verbose:
                print(f"   ⚠️  Minor frame diff: {frame_diff} (likely rounding, OK)")
        else:
            if verbose:
                print(f"   ⚠️  Audio sync warning:")
                print(f"   Expected: {expected_frames}, Got: {actual_frames}")
                print(f"   Difference: {frame_diff} frames")
                print(f"   Try: force_time_scale={time_scale-1} or {time_scale+1}")
        
        return (final_output,)


# ======== NODE REGISTRATION ========
NODE_CLASS_MAPPINGS = {
    "UniversalSmartVAEDecode": UniversalSmartVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalSmartVAEDecode": "🎬 Universal VAE Decode (v11.1 Final)",
}
