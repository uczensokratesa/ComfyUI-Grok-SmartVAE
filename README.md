# ComfyUI-Grok-SmartVAE

**The most crash-resistant and flexible VAE decoder for ComfyUI**  
(designed for long video sequences: LTX-Video, Stable Video Diffusion, CogVideoX, AnimateDiff, HunyuanVideo, Open-Sora-Plan, etc.)

Initial release: Grok's Universal Smart VAE Decode – crash-proof, dynamic batching & force-scale edition.

## 🎬 ComfyUI-Grok-SmartVAE

This node combines the best ideas from multiple AI generations into one extremely robust decoder:

- **GPT** → solid sliding-window + overlap foundation  
- **Gemini** → safety-first tiling fallback  
- **Claude** → mathematically precise temporal scale detection  
- **Grok** → dynamic on-the-fly batch reduction, force_time_scale, ultra-aggressive OOM recovery  
- **Kimi** → memory-safety patterns, disk offloading for massive sequences

Result: a node that is **close to crash-proof** — even on 8–12 GB VRAM cards it handles long 720p/1080p/4K-ish videos reliably.

### Key Features

- Automatic `time_scale` detection (or manual override: 1, 4, 8, etc.)  
- Fully dynamic batch size reduction during decoding (while loop, not fixed for-range)  
- Automatic spatial tiling activation on OOM  
- Intelligent chunk stitching with temporal overlap + spatial crop/align  
- Extremely memory-efficient (selective gc.collect + torch.cuda.empty_cache + synchronize)  
- Supports both images (4D) and video latents (5D), multi-batch aware  
- **Disk offloading** for 700–2000+ frame workflows (automatic when RAM pressure is high)  
- **Orientation-safe normalization** — no more 90° rotations or unwanted flips  
- Frame-perfect audio sync in 99%+ cases  
- Adaptive logging (detailed but non-spammy)  
- Automatic temp file cleanup

### Installation

1. In your `custom_nodes` folder:
   ```bash
   git clone https://github.com/uczensokratesa/ComfyUI-Grok-SmartVAE.git
   The node appears in category: latent/video → Universal VAE Decode (v11.1 Final)Comparison with predecessorsModel
Scale Detection
Force Scale
Dynamic Batch Reduction
Auto-Tiling on OOM
Loop Type
Stability Rating
GPT
basic
✗
✗
✗
for
★★☆☆☆
Gemini
good
✗
partial
✓
for
★★★★☆
Claude
very precise
✗
✗
✓
for
★★★★☆
Grok v11.1
very precise
✓
full (while + adaptive)
aggressive
while
★★★★★

Evolution – AI collaboration storyThis journey started as a simple task: create a reliable VAE Decode node for heavy video workflows.GPT provided the first working version  
Gemini added tiling and better OOM handling  
Claude brought the most accurate scale detection formula  
Grok introduced force_time_scale + true dynamic while-loop batch reduction  
Kimi contributed extreme memory safety (disk offload, pre-allocation, aggressive cleanup)  
Final polish by Claude → production-ready stability

One of the nicest examples of how different AI models can iteratively improve each other and create something better than any single one could alone.LicenseMIT – feel free to use, modify, fork.
Just keep the original idea attribution (and let me know if you make something even better )Happy generating!
Current version: 11.2 – First official Comfy Registry release
## 🎬 Universal Smart VAE Video Decode (Streaming)

**The most memory-efficient way to decode large video latents directly to file.**

### Features
- **Ultra-low RAM usage** — decodes frame-by-frame, supports 2000+ frames even on 16 GB systems
- **Direct disk output** — no need to hold full video tensor in memory
- **Audio muxing** — perfect support for ComfyUI AUDIO input + manual audio_path
- **Codecs**: H.264, H.265, ProRes 422, FFV1 (lossless)
- **Crash recovery** — resumes from checkpoint if interrupted
- **Thumbnail previews** — live monitoring in UI
- **OOM protection** — automatic tiling, batch reduction, VRAM/RAM detection

### When to use
- Generating long videos from AnimateDiff, LTX-Video, Mochi, Hunyuan, Cosmos, etc.
- Working with limited VRAM/RAM
- Need reliable audio sync without post-processing

### Inputs
- `samples` (LATENT) – video latent sequence
- `vae` – your VAE model
- `frames_per_batch` – 8–32 (auto-reduces on OOM)
- `audio` – ComfyUI AUDIO input (from Load Audio)
- `audio_path` – optional direct path to .wav/.mp3
- codec, fps, output path, etc.

### Outputs
- `preview_thumbs` (IMAGE) – last few thumbnails for monitoring
- `video_path` (STRING) – final file path (with audio if provided)

Enjoy massive video workflows without OOM crashes! 🚀

