# Changelog

All notable changes to ComfyUI-Grok-SmartVAE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] - Branch: Twonewnodes

### Added
- **Advanced Load Latent v1.3** - Load saved latents with dropdown selection
  - Auto-scan `output/latents/` directory
  - Refresh button for live updates
  - PyTorch 2.6+ compatible (safetensors primary, pickle fallback)
  - Metadata display (shape, seed, timestamp, format)
  - Manual path override support
  
- **Latent Metadata Viewer** - Quick inspection without loading tensor
  - Fast metadata-only read
  - Browse large latent collections efficiently
  - Outputs: shape, seed, timestamp, format
  
- **SmartVAE Advanced Decoder v2.3.1** - Temporal color correction
  - Ramped cross-fade algorithm (zero flicker)
  - Three correction modes: temporal, reference, temporal_match
  - Parameters: correction_strength, cross_fade_frames, ema_momentum
  - 2-5% overhead, tested on 50s+ videos
  - Compatible with all VAEs (SD 1.5, SDXL, LTX-Video 2)

### Fixed
- **Critical: Audio sync issue** (1s offset eliminated)
  - Root cause: Video re-encoding during mux caused encoder delay
  - Solution: FFmpeg stream copy + PTS normalization
  - Flags: `-c:v copy`, `-fflags +genpts`, `-vsync cfr`, `-af asetpts=PTS-STARTPTS`
  - Result: Perfect A/V sync + 10x faster muxing + lossless quality
  
- **Advanced Decoder: Edge case crashes**
  - Complete `__init__` variable initialization
  - Prevents AttributeError on first decode with cache
  
- **Advanced Decoder: Performance optimization**
  - Eliminated redundant `_extract_stats()` calls
  - ~5-10% speedup in color correction mode

### Changed
- **Documentation: Corrected LTX-2 diagnosis**
  - OLD: "Silent VRAM death" (implied decoder bug)
  - NEW: "LTX-2 architectural limit" (model constraint)
  - Clarified root cause: Model architecture, not VRAM issue
  - Added concrete workaround: Keep videos under ~1000 frames

---

## [1.0.10] - 2025-02-27

### Added
- **Ignore Warnings Mode** - User-controlled risk tolerance
  - `none` (default): Stop on corruption with detailed diagnostics
  - `minor`: Continue if <10% corrupted (partial recovery)
  - `all`: Force decode anyway (high risk)
  - Prevents "black frames without explanation" frustration

### Fixed
- **Audio sync bug** - 1s offset on long videos
  - Changed from video re-encoding to stream copy
  - Added PTS (presentation timestamp) normalization
  - Result: Zero offset even on 40s+ videos

- **NaN handling** - Corrupted latent detection
  - Pre-decode validation with percentage calculation
  - Identifies affected frame ranges
  - Provides actionable error messages with concrete solutions

### Changed
- Improved error messages with root cause analysis
- Better console output formatting
- Version display in node title

---

## [1.0.9] - 2025-02-15

### Added
- **Corrupted latent detection** - Pre-decode validation system
  - Detects NaN values from upstream OOM
  - Shows corruption percentage and affected frames
  - Suggests concrete fixes (reduce length, lower CFG, etc.)
  
- **Diagnostic logging** - Detailed error context
  - Latent shape and size estimates
  - Device and memory info
  - Suggested parameter adjustments

### Fixed
- NaN/Inf handling in `_normalize_output()`
- Improved OOM error messages
- Better silent OOM detection

---

## [1.0.8] - 2025-02-10

### Added
- **AUDIO input support** - Direct ComfyUI AUDIO node connection
  - Priority over `audio_path` parameter
  - FFmpeg PCM conversion pipeline
  - Automatic temp file cleanup

### Fixed
- Audio path validation before mux
- Temp audio file race condition (5 retry attempts)

---

## [1.0.7] - 2025-02-05

### Added
- **Resume on crash** - Checkpoint-based recovery
  - Saves metadata every 100 frames
  - Video concatenation on resume
  - Preserves previous work if interrupted

### Fixed
- Resume latent index calculation for `time_scale > 1`
- Tuple handling from VAE output (crash prevention)

---

## [11.2] - 2025-01-25

### Added
- **First ComfyUI Registry release**
- Production-ready stability
- Comprehensive README documentation

---

## [11.1] - 2025-01-20

### Added
- **Dynamic batch reduction** - True while-loop adaptation
  - Automatic OOM recovery
  - Aggressive tiling fallback
  - VRAM/RAM detection and adjustment

- **Force time scale** - Manual override for VAE detection
  - Useful for models with non-standard scaling (LTX-Video: 8x)

### Changed
- Switched from fixed for-loop to adaptive while-loop
- Improved memory estimation algorithms

---

## [11.0] - 2025-01-15

### Added
- **SmartVAE Streaming Decoder** - Frame-by-frame video encoding
  - Direct disk output (no full tensor in memory)
  - Multiple codec support (H.264, H.265, ProRes, FFV1)
  - Thumbnail preview system
  - Crash recovery with resume

- **Universal Smart VAE Decode** - Original non-streaming node
  - Automatic time scale detection
  - Dynamic batch size reduction
  - Spatial tiling on OOM
  - Disk offloading for massive sequences (700-2000+ frames)

### Technical
- Precise temporal scale detection algorithm (Claude)
- Orientation-safe normalization (no unwanted rotations)
- Intelligent chunk stitching with overlap
- Multi-batch awareness

---

## [1.0.0] - 2025-01-01

### Added
- Initial release
- Basic VAE decode with crash protection
- OOM recovery mechanisms
- Tiling support

---

## Legend

### Types of Changes
- `Added` - New features
- `Changed` - Changes in existing functionality
- `Deprecated` - Soon-to-be removed features
- `Removed` - Removed features
- `Fixed` - Bug fixes
- `Security` - Vulnerability fixes

### Version Format
- **Major.Minor.Patch** (e.g., 1.0.10)
  - Major: Breaking changes
  - Minor: New features (backward compatible)
  - Patch: Bug fixes (backward compatible)

---

## Upcoming (Roadmap)

### Planned for v11.6
- [ ] Batch load multiple latents
- [ ] Latent diff viewer (compare 2 latents)
- [ ] Thumbnail preview in Load Latent dropdown
- [ ] Advanced Decoder: Per-channel correction option

### Under Consideration
- [ ] Real-time preview streaming (WebSocket)
- [ ] GPU-accelerated color correction (CUDA kernels)
- [ ] Latent interpolation node
- [ ] Video segment merger

---

**Note:** Version numbers follow semantic versioning. The base node (Streaming Decoder) uses 1.x.x, while the original Universal VAE uses 11.x for historical continuity.
