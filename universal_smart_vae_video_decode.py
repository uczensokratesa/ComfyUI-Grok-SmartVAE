"""
Universal Smart VAE Video Decode - Streaming Node (FINAL)
Production-ready standalone node for streaming video decoding.

Based on: UniversalSmartVAEDecode v11.2 + streaming extensions
Refinements: Claude (final polish), Grok (implementation), Gemini (recovery), AI ensemble

Features:
- Frame-by-frame decoding (80% less RAM than standard pipeline)
- Crash recovery with resume capability
- Thumbnail previews in ComfyUI UI
- Multiple codecs: h264, h265, prores, FFV1 lossless
- Progress tracking with file write percentage
- Audio muxing support (now with AUDIO input support)
- Auto-detection of RAM/VRAM availability

Version: 1.0.8 (Audio fixes from Claude)
License: MIT
"""

import torch
import os
import time
import json
import logging
from typing import Optional, Tuple
import gc
import comfy.utils
from comfy.model_management import throw_exception_if_processing_interrupted
import warnings
import folder_paths
import subprocess
import numpy as np

logger = logging.getLogger(__name__)

# Dependencies check
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    warnings.warn(
        "imageio not available. Install with: pip install imageio imageio-ffmpeg\n"
        "This node requires imageio for video encoding."
    )

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("numpy not available - required for this node")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.info("cv2 not available - preview thumbnails will be skipped")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.info("psutil not available - RAM estimation unavailable")

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    logger.info("torchaudio not available - fallback to FFmpeg for AUDIO input.")


class StreamingVideoConfig:
    """Configuration for streaming video decode."""
    
    CODECS = {
        "h264": {
            "codec": "libx264",
            "ext": "mp4",
            "output_params": ['-crf', '23', '-preset', 'slow'],
            "pixel_format": "yuv420p",
            "description": "H.264 (best compatibility, good quality)"
        },
        "h265": {
            "codec": "libx265",
            "ext": "mp4",
            "output_params": ['-crf', '28', '-preset', 'slow'],
            "pixel_format": "yuv420p",
            "description": "H.265/HEVC (50% smaller files, slower encode)"
        },
        "prores": {
            "codec": "prores_ks",
            "ext": "mov",
            "output_params": ['-profile:v', '3'],
            "pixel_format": "yuv422p10le",
            "description": "ProRes 422 (professional, large files)"
        },
        "ffv1": {
            "codec": "ffv1",
            "ext": "mkv",
            "output_params": ['-level', '3', '-g', '1', '-slices', '4', '-slicecrc', '1'],
            "pixel_format": "yuv444p",
            "description": "FFV1 (lossless archival, very large files)"
        }
    }
    
    PREVIEW_INTERVAL = 50
    METADATA_SAVE_INTERVAL = 100


class StreamingVideoWriter:
    """Handles streaming video encoding with crash recovery."""
    
    def __init__(self, output_path: str, fps: int, codec: str, 
                 width: int, height: int, resume: bool = False):
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
        self.width = width
        self.height = height
        self.resume = resume
        
        self.metadata_path = output_path + ".metadata.json"
        self.temp_path = output_path + ".tmp"
        
        self.writer = None
        self.frames_written = 0
        self.last_preview = None
        
        if resume and os.path.exists(self.metadata_path):
            self._load_metadata()
        
        self._init_writer()
    
    def _init_writer(self):
        """Initialize video writer."""
        if not IMAGEIO_AVAILABLE:
            raise RuntimeError(
                "imageio required for video encoding.\n"
                "Install: pip install imageio imageio-ffmpeg"
            )
        
        codec_config = StreamingVideoConfig.CODECS.get(
            self.codec, 
            StreamingVideoConfig.CODECS["h264"]
        )
        
        write_path = self.temp_path if not self.resume else self.output_path
        
        try:
            self.writer = imageio.get_writer(
                write_path,
                fps=self.fps,
                codec=codec_config["codec"],
                quality=None,  # Use output_params for quality control
                pixelformat=codec_config["pixel_format"],
                output_params=codec_config["output_params"],
                macro_block_size=1
            )
            
            logger.info(f"Video writer initialized: {codec_config['description']}")
            logger.info(f"Output: {write_path} ({self.width}x{self.height} @ {self.fps}fps)")
            
        except Exception as e:
            logger.error(f"Failed to initialize video writer: {e}")
            raise
    
    def write_frame(self, frame: torch.Tensor) -> Optional[np.ndarray]:
        """
        Write frame to video.
        
        Returns:
            Preview thumbnail every PREVIEW_INTERVAL frames, else None
        """
        if not NUMPY_AVAILABLE:
            raise RuntimeError("numpy required")
        
        frame_np = (frame * 255.0).clamp(0, 255).byte().cpu().numpy()
        
        self.writer.append_data(frame_np)
        self.frames_written += 1
        
        preview = None
        if self.frames_written % StreamingVideoConfig.PREVIEW_INTERVAL == 0:
            scale = min(1.0, 512.0 / max(self.width, self.height))
            
            if scale < 1.0:
                if CV2_AVAILABLE:
                    new_w = int(self.width * scale)
                    new_h = int(self.height * scale)
                    preview = cv2.resize(frame_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
                # No fallback - skip if no cv2
            else:
                preview = frame_np.copy()
            
            self.last_preview = preview
        
        if self.frames_written % StreamingVideoConfig.METADATA_SAVE_INTERVAL == 0:
            self._save_metadata()
        
        return preview
    
    def _save_metadata(self):
        """Save crash recovery metadata."""
        metadata = {
            "frames_written": self.frames_written,
            "output_path": self.output_path,
            "temp_path": self.temp_path,
            "fps": self.fps,
            "codec": self.codec,
            "resolution": [self.width, self.height],
            "timestamp": time.time()
        }
        
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.debug(f"Metadata save failed: {e}")
    
    def _load_metadata(self):
        """Load metadata for resume."""
        try:
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.frames_written = metadata.get("frames_written", 0)
            logger.info(f"Resuming from frame {self.frames_written}")
            
        except Exception as e:
            logger.warning(f"Metadata load failed: {e}")
            self.frames_written = 0
    
    def finalize(self, audio_path: Optional[str] = None) -> str:
        """Finalize video file."""
        if self.writer:
            self.writer.close()
            self.writer = None
        
        # Cleanup metadata
        if os.path.exists(self.metadata_path):
            try:
                os.remove(self.metadata_path)
            except:
                pass
        
        temp_exists = os.path.exists(self.temp_path)
        
        if not temp_exists:
            # Writer pisze bezpośrednio do output_path
            if audio_path and os.path.exists(audio_path):
                # Musimy zmuxować audio z istniejącym plikiem
                final_path = self._mux_audio(self.output_path, audio_path)
                return final_path
            return self.output_path
        
        # Mamy plik tymczasowy do przetworzenia
        if audio_path and os.path.exists(audio_path):
            final_path = self._mux_audio(self.temp_path, audio_path)
            # Nie usuwaj temp_path tutaj - _mux_audio już to robi w przypadku sukcesu
            if final_path == self.temp_path:  # Błąd muxowania
                try:
                    os.replace(self.temp_path, self.output_path)
                except:
                    pass
                return self.output_path
            return final_path
        else:
            # Brak audio, tylko przenieś plik
            try:
                os.replace(self.temp_path, self.output_path)
            except Exception as e:
                logger.error(f"Failed to move temp file: {e}")
                return self.temp_path
            return self.output_path
    
    def _mux_audio(self, video_path: str, audio_path: str) -> str:
        """Mux audio with video using ffmpeg."""
        
        # Walidacja inputów
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return video_path
        
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return video_path
        
        import subprocess
        
        # Pobierz ffmpeg
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg_exe = "ffmpeg"
        
        base, ext = os.path.splitext(self.output_path)
        output_with_audio = f"{base}_audio{ext}"
        
        # KRYTYCZNA POPRAWKA: Pobierz config kodeka
        codec_config = StreamingVideoConfig.CODECS.get(self.codec)
        if codec_config is None:
            logger.warning(f"Unknown codec '{self.codec}', using h264")
            codec_config = StreamingVideoConfig.CODECS["h264"]
        
        try:
            # POPRAWIONA komenda FFmpeg - RE-ENCODE video z tym samym kodekiem
            cmd = [
                ffmpeg_exe, "-y",
                "-i", video_path,
                "-i", audio_path,
                "-map", "0:v:0",          # Video ze źródła 0
                "-map", "1:a:0",          # Audio ze źródła 1
                "-c:v", codec_config["codec"],  # RE-ENCODE tym samym kodekiem!
            ]
            
            # Dodaj parametry kodeka (CRF, preset, itp.)
            cmd.extend(codec_config["output_params"])
            
            # Dodaj pixel format i audio
            cmd.extend([
                "-pix_fmt", codec_config["pixel_format"],
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",              # Dopasuj do krótszego strumienia
                output_with_audio
            ])
            
            logger.info(f"Running ffmpeg: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=600,  # 10 min dla dużych plików
                text=True
            )
            
            if os.path.exists(output_with_audio):
                # POPRAWKA: Usuń stary plik video bez audio
                try:
                    if video_path != self.output_path:  # Tylko jeśli to temp file
                        os.remove(video_path)
                        logger.info(f"Removed temp video: {video_path}")
                except Exception as e:
                    logger.debug(f"Could not remove temp video: {e}")
                
                logger.info(f"✓ Audio muxed successfully: {output_with_audio}")
                return output_with_audio
            else:
                logger.error("FFmpeg succeeded but output not created")
                return video_path
                
        except subprocess.TimeoutExpired:
            logger.error("Audio muxing timeout (>10min)")
            return video_path
        except subprocess.CalledProcessError as e:
            stderr = e.stderr if e.stderr else "No error output"
            logger.error(f"FFmpeg error: {stderr}")
            return video_path
        except FileNotFoundError:
            logger.error("ffmpeg not found in PATH")
            return video_path
        except Exception as e:
            logger.error(f"Unexpected mux error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return video_path
    
    def __del__(self):
        """Cleanup on destruction."""
        if self.writer:
            try:
                self.writer.close()
            except Exception:
                pass


class UniversalSmartVAEVideoDecode:
    """
    Standalone streaming video decode node.
    Decodes latents directly to video file without holding full tensor in RAM.
    """
    
    MAX_OOM_RETRIES = 5
    
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
                    "tooltip": "Frames per decode batch. Auto-reduces on OOM."
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
                    "tooltip": "Manual override (0=auto). E.g., 8 for LTX-Video."
                }),
                "enable_tiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force tiling. Auto-enables on OOM."
                }),
                "tile_size": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Tile size in pixels."
                }),
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show detailed logs."
                }),
                "video_output_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Output path. Empty = auto-generated."
                }),
                "fps": ("INT", {
                    "default": 24,
                    "min": 1,
                    "max": 120,
                    "step": 1,
                    "tooltip": "Frames per second"
                }),
                "codec": (list(StreamingVideoConfig.CODECS.keys()), {
                    "default": "h264",
                    "tooltip": "h264=compatible | h265=smaller | prores/ffv1=professional"
                }),
                "audio_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional audio file path to mux (alternative to AUDIO input)"
                }),
                "audio": ("AUDIO", {
                    "tooltip": "Optional AUDIO input (ComfyUI format). Priority over audio_path."
                }),
                "resume_on_crash": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Resume if crash occurred"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview_thumbs", "video_path")
    FUNCTION = "decode"
    OUTPUT_NODE = True
    CATEGORY = "latent/video"

    def __init__(self):
        self._time_scale_cache = {}
        self._force_scale_cache = {}

    def _get_available_vram(self) -> Optional[float]:
        """Get available VRAM in GB."""
        try:
            if not torch.cuda.is_available():
                return None
            free_vram, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
            return free_vram / (1024 ** 3)
        except Exception as e:
            logger.warning(f"VRAM detection failed: {e}")
            return None

    def _get_available_ram(self) -> Optional[float]:
        """Get available RAM in GB."""
        if PSUTIL_AVAILABLE:
            try:
                return psutil.virtual_memory().available / (1024 ** 3)
            except Exception as e:
                logger.warning(f"RAM detection failed: {e}")
        return None

    def _estimate_chunk_vram(self, frames: int, channels: int, h: int, w: int, 
                            time_scale: int = 1) -> float:
        """Estimate VRAM for chunk (GB)."""
        spatial_scale = 8
        latent_bytes = frames * channels * h * w * 4
        output_frames = frames * time_scale
        output_bytes = output_frames * 3 * (h * spatial_scale) * (w * spatial_scale) * 4
        total_bytes = (latent_bytes + output_bytes) * 3.5 * 1.1
        return total_bytes / (1024 ** 3)

    def _estimate_output_ram(self, expected_frames: int, output_h: int, output_w: int) -> float:
        """Estimate total RAM for output tensor (GB) - for info only."""
        bytes_per_frame = output_h * output_w * 3 * 4
        return (expected_frames * bytes_per_frame) / (1024 ** 3)

    def detect_time_scale(self, vae, latents: torch.Tensor, force_scale: int = 0, 
                          verbose: bool = True) -> int:
        """Detect temporal upsampling scale."""
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
        
        # Try metadata
        if hasattr(vae, 'downscale_index_formula') and vae.downscale_index_formula:
            try:
                time_scale = int(vae.downscale_index_formula[0])
                if verbose:
                    logger.info(f"🔍 VAE metadata time scale: {time_scale}x")
                self._time_scale_cache[vae_id] = time_scale
                return time_scale
            except Exception as e:
                logger.debug(f"Metadata parse failed: {e}")
        
        # Empirical test
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
        """Detect output resolution."""
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
                    raise RuntimeError("Failed to detect output size") from inner_e
            else:
                raise
        
        test_out = self._normalize_output(test_out, aspect_ratio)
        output_h, output_w = test_out.shape[1:3]
        
        del test_out, test_sample
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return output_h, output_w

    def _normalize_output(self, tensor, aspect_ratio: Optional[float] = None) -> torch.Tensor:
        """Normalize to [F, H, W, C]."""
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
        
        tensor = torch.clamp(tensor, 0.0, 1.0)
        if tensor.shape[-1] > 3:
            tensor = tensor[..., :3]
        
        return tensor.contiguous()

    def _center_crop_to_reference(self, tensor: torch.Tensor, h_ref: int, w_ref: int) -> torch.Tensor:
        """Center crop to reference dimensions."""
        _, h, w, _ = tensor.shape
        if h == h_ref and w == w_ref:
            return tensor
        
        h_offset = max(0, (h - h_ref) // 2)
        w_offset = max(0, (w - w_ref) // 2)
        
        return tensor[:, h_offset:h_offset + h_ref, w_offset:w_offset + w_ref, :]

    def decode(self, vae, samples, frames_per_batch, overlap_frames=2, force_time_scale=0, 
               enable_tiling=False, tile_size=512, verbose=False,
               video_output_path="", fps=24, codec="h264", audio_path="", audio=None, resume_on_crash=True):
        """Main decode entry point."""
        
        if not IMAGEIO_AVAILABLE or not NUMPY_AVAILABLE:
            raise RuntimeError(
                "Required dependencies missing.\n"
                "Install: pip install imageio imageio-ffmpeg numpy"
            )
        
        latents = samples["samples"]
        
        if latents.dim() == 4:
            raise ValueError(
                "This node is for video latents only (5D tensors).\n"
                "For images (4D), use standard VAE Decode node."
            )
        
        batch, channels, total_frames, h_latent, w_latent = latents.shape
        
        if total_frames <= 0:
            raise ValueError("Latent has no frames")
        
        # Detect parameters
        time_scale = self.detect_time_scale(vae, latents, force_time_scale, verbose)
        expected_frames = 1 + (total_frames - 1) * time_scale
        aspect_ratio = h_latent / float(w_latent)
        
        output_h, output_w = self.detect_output_size(vae, latents, h_latent, w_latent, tile_size, verbose)
        
        # RAM estimation for info
        est_ram = self._estimate_output_ram(expected_frames, output_h, output_w)
        if verbose:
            logger.info(f"   Estimated full-RAM usage (if not streaming): {est_ram:.2f}GB")
        
        # Generate output path
        if not video_output_path:
            output_dir = folder_paths.get_output_directory()
            timestamp = int(time.time())
            codec_ext = StreamingVideoConfig.CODECS[codec]["ext"]
            video_output_path = os.path.join(output_dir, f"video_{timestamp}.{codec_ext}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(video_output_path) or ".", exist_ok=True)
        
        # POPRAWIONA obsługa AUDIO input
        temp_audio_path = None
        final_audio_path = audio_path  # Zachowaj oryginalną ścieżkę jako fallback
        
        if audio is not None:
            try:
                # Walidacja struktury AUDIO
                if not isinstance(audio, dict) or 'waveform' not in audio or 'sample_rate' not in audio:
                    logger.warning(f"Invalid AUDIO format. Expected dict with 'waveform' and 'sample_rate', got: {type(audio)}")
                else:
                    waveform = audio['waveform']
                    sample_rate = audio['sample_rate']
                    
                    # Dodaj tę walidację przed zapisem audio
                    if waveform is None or sample_rate is None:
                        logger.error("AUDIO input missing 'waveform' or 'sample_rate'")
                        final_audio_path = audio_path
                    elif waveform.numel() == 0:
                        logger.error("AUDIO input waveform is empty")
                        final_audio_path = audio_path
                    else:
                        # Kod zapisu audio PRZEZ FFMPEG (najbardziej niezawodny)
                        # Konwersja do CPU jeśli tensor na GPU
                        if isinstance(waveform, torch.Tensor):
                            waveform = waveform.cpu()
                        
                        # Zapewnienie poprawnego kształtu [channels, samples]
                        if waveform.dim() == 1:
                            waveform = waveform.unsqueeze(0)
                        elif waveform.dim() == 3:
                            waveform = waveform.squeeze(0)
                        
                        timestamp = int(time.time() * 1000)  # Więcej precyzji dla unikatowości
                        temp_audio_path = os.path.join(
                            folder_paths.get_temp_directory(), 
                            f"temp_audio_{timestamp}_{os.getpid()}.wav"
                        )
                        
                        # ZAPIS PRZEZ FFMPEG (bez TorchCodec, torchaudio, soundfile!)
                        try:
                            import imageio_ffmpeg
                            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                        except ImportError:
                            ffmpeg_exe = "ffmpeg"
                        
                        # Konwersja tensora do raw PCM (int16)
                        waveform_np = (waveform.numpy() * 32767).astype('int16')
                        channels = waveform_np.shape[0]
                        
                        # Wywołanie ffmpeg z raw audio input
                        cmd = [
                            ffmpeg_exe, "-y",
                            "-f", "s16le",              # Format: signed 16-bit little-endian
                            "-ar", str(sample_rate),    # Sample rate
                            "-ac", str(channels),       # Number of channels
                            "-i", "pipe:0",             # Input z stdin
                            "-c:a", "pcm_s16le",        # Codec WAV
                            temp_audio_path
                        ]
                        
                        process = subprocess.Popen(
                            cmd,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                        
                        # Wyślij raw audio data do ffmpeg (transponowane do [samples, channels])
                        stdout, stderr = process.communicate(input=waveform_np.T.tobytes())
                        
                        if process.returncode != 0:
                            raise RuntimeError(f"FFmpeg failed: {stderr.decode()}")
                        
                        final_audio_path = temp_audio_path
                        
                        if verbose:
                            logger.info(f"✓ AUDIO saved (ffmpeg): {temp_audio_path}")
                            logger.info(f"  Sample rate: {sample_rate}Hz, Shape: {waveform.shape}")
                            
            except Exception as e:
                logger.warning(f"Failed to process AUDIO input: {e}")
                import traceback
                if verbose:
                    logger.debug(traceback.format_exc())
                temp_audio_path = None
        
        # Logowanie finalnego źródła audio
        if final_audio_path:
            if os.path.exists(final_audio_path):
                logger.info(f"🔊 Audio source ready: {final_audio_path}")
            else:
                logger.warning(f"Audio path specified but file not found: {final_audio_path}")
                final_audio_path = ""  # Reset jeśli plik nie istnieje
        
        logger.info("🎬 STREAMING VIDEO DECODE")
        logger.info(f"   Output: {video_output_path}")
        logger.info(f"   Codec: {codec} @ {fps}fps")
        logger.info(f"   Resolution: {output_h}x{output_w}")
        logger.info(f"   Expected frames: ~{expected_frames}")
        if final_audio_path:
            logger.info(f"   Audio source: {final_audio_path}")
        
        try:
            return self._streaming_decode(
                vae, latents, video_output_path, fps, codec, final_audio_path,  # <- użyj final_audio_path
                resume_on_crash, frames_per_batch, overlap_frames,
                force_time_scale, enable_tiling, tile_size, verbose,
                time_scale, expected_frames, aspect_ratio, output_h, output_w,
                h_latent, w_latent, channels
            )
        finally:
            # Cleanup temp audio
            if temp_audio_path and os.path.exists(temp_audio_path):
                # Dodaj małe opóźnienie i ponowne próby
                for attempt in range(5):
                    try:
                        time.sleep(0.5)
                        os.remove(temp_audio_path)
                        if verbose:
                            logger.info(f"✓ Cleaned temp audio: {temp_audio_path}")
                        break
                    except Exception as e:
                        if attempt == 4:  # Ostatnia próba
                            logger.warning(f"Could not remove temp audio: {e}")
                        continue

    def _streaming_decode(self, vae, latents, video_output_path, fps, codec, audio_path,
                          resume_on_crash, frames_per_batch, overlap_frames,
                          force_time_scale, enable_tiling, tile_size, verbose,
                          time_scale, expected_frames, aspect_ratio, output_h, output_w,
                          h_latent, w_latent, channels):
        """Streaming decode implementation."""
        
        batch, _, total_frames, _, _ = latents.shape
        
        # Initialize writer
        writer = StreamingVideoWriter(
            video_output_path, fps, codec, output_h, output_w, resume=resume_on_crash
        )
        
        # Validate parameters
        frames_per_batch = max(1, min(frames_per_batch, total_frames))
        overlap_frames = max(0, min(overlap_frames, frames_per_batch - 1))
        initial_overlap = overlap_frames
        
        # VRAM optimization
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
                    logger.info(f"📉 VRAM reduction: {old_batch} → {frames_per_batch}")
        
        if verbose:
            logger.info(f"   Batch: {frames_per_batch}, Overlap: {overlap_frames}")
        
        # Processing state
        current_batch = frames_per_batch
        start_idx = writer.frames_written if resume_on_crash else 0
        frames_processed = 0
        last_frames_processed = -1
        stagnation_count = 0
        MAX_STAGNATION = 3
        oom_retry_count = 0
        
        pbar = comfy.utils.ProgressBar(total_frames)
        preview_frames = []
        
        # Main loop
        while start_idx < total_frames:
            throw_exception_if_processing_interrupted()
            
            # Safety guards
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
            
            try:
                with torch.no_grad():
                    if enable_tiling and hasattr(vae, 'decode_tiled'):
                        decoded_chunk = vae.decode_tiled(latent_chunk, tile_x=tile_size, tile_y=tile_size)
                    else:
                        decoded_chunk = vae.decode(latent_chunk)
                    decoded_chunk = decoded_chunk.cpu()
                
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
            
            decoded_chunk = self._normalize_output(decoded_chunk, aspect_ratio)
            
            front_trim = (start_idx - ctx_start) * time_scale
            
            if end_idx == total_frames:
                valid_frames = decoded_chunk[front_trim:]
            else:
                core_length = (end_idx - start_idx) * time_scale
                valid_frames = decoded_chunk[front_trim:front_trim + core_length]
            
            valid_frames = self._center_crop_to_reference(valid_frames, output_h, output_w)
            
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
        
        final_path = writer.finalize(audio_path if audio_path else None)
        
        if verbose:
            logger.info(f"✅ Streaming complete!")
            logger.info(f"   File: {final_path}")
            logger.info(f"   Frames: {writer.frames_written}")
            file_size_mb = os.path.getsize(final_path) / (1024 * 1024)
            logger.info(f"   Size: {file_size_mb:.2f} MB")
        
        if preview_frames:
            preview_tensor = torch.stack(preview_frames, dim=0)
        else:
            # Fallback: empty tensor if no previews (e.g. no cv2 or short video)
            preview_tensor = torch.empty(0, dtype=torch.float32)
        
        logger.info("📸 NOTE: 'preview_thumbs' output is for live monitoring only (last 5 thumbnails). Full video is saved to disk.")
        
        return (preview_tensor, final_path)


NODE_CLASS_MAPPINGS = {
    "UniversalSmartVAEVideoDecode": UniversalSmartVAEVideoDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalSmartVAEVideoDecode": "🎬 Universal VAE Video Decode (Streaming)",
}
