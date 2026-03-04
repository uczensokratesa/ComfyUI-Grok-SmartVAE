ll notable changes to ComfyUI-Grok-SmartVAE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [12.0.0] - 2025-03-04 (Branch: Twonewnodes)

### 🎉 Major Release - New Node Suite

This release adds **3 new production-ready nodes** and establishes ComfyUI-Grok-SmartVAE as a complete video workflow solution.

### Added
- **Advanced Load Latent v1.3** - Load saved latents with dropdown selection
  - Auto-scan `output/latents/` directory with refresh button
  - PyTorch 2.6+ compatible (safetensors primary, pickle fallback)
  - Metadata display (shape, seed, timestamp, format)
  - Enables iterative refinement workflows (save once, iterate color grading)
  
- **Latent Metadata Viewer v1.0** - Quick inspection without loading tensor
  - Fast metadata-only read for large files (10GB+)
  - Browse latent collections by seed/date/format
  - Zero memory overhead
  
- **SmartVAE Advanced Decoder v2.3.1** - Temporal color correction
  - **Zero-flicker** ramped cross-fade algorithm (breakthrough innovation)
  - Three correction modes: temporal, reference, temporal_match
  - Configurable parameters: strength (0.0-0.4), fade (4-24 frames), EMA (0.7-0.98)
  - Minimal overhead (2-5%), tested on 50s+ videos
  - Inherits all base decoder features (streaming, OOM recovery, audio sync)

### Fixed
- **CRITICAL: Audio sync issue** (1s offset eliminated in v1.0.10)
  - Root cause: Video re-encoding during mux caused encoder delay + PTS drift
  - Solution: FFmpeg stream copy (`-c:v copy`) + PTS normalization
  - Result: Perfect A/V sync + 10x faster muxing (4s vs 45s) + lossless quality
  - Flags added: `-fflags +genpts`, `-vsync cfr`, `-af asetpts=PTS-STARTPTS`, `-itsoffset 0`
  
- **Advanced Decoder: Edge case crashes**
  - Complete `__init__` variable initialization (prevents AttributeError on cache)
  - Added safe defaults for all parameters
  
- **Advanced Decoder: Performance optimization**
  - Eliminated redundant `_extract_stats()` calls (~5-10% speedup)
  - Optimized EMA update (reuse computed stats instead of recalculating)

### Changed
- **IMPORTANT: Corrected LTX-2 diagnosis** (documentation accuracy)
  - ❌ OLD: "Silent VRAM death" → implied this was a decoder/VRAM bug
  - ✅ NEW: "LTX-2 architectural limit" → clarified this is a model constraint
  - Root cause: LTX-2 model architecture has hard limit (~1000-1200 frames)
  - Workaround: Keep videos under 40-45s, or split into segments
  - This is NOT a VRAM issue and NOT a node bug

### Migration Notes
- **No breaking changes** - all existing workflows continue to work
- New nodes are **additive** - base decoder unchanged
- Version jump (11.x → 12.0) reflects **major feature additions**

### Known Limitations
- LTX-Video 2: ~1000 frame architectural limit (model constraint, not node issue)
- Color correction: ~2-5% processing overhead (acceptable for quality gain)

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
