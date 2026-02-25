"""
ComfyUI-Grok-SmartVAE
Production VAE decoders for long video sequences

Nodes:
- UniversalSmartVAEDecode: Universal decoder (images + video) with disk offload
- SmartVAE_StreamingDecoder: Streaming video decoder with file output
- SmartVAE_AdvancedDecoder: Experimental features (anti-color bleed)

GitHub: https://github.com/uczensokratesa/ComfyUI-Grok-SmartVAE
Version: 11.4.0
License: MIT
"""

from .UniversalSmartVAEDecode import UniversalSmartVAEDecode
from .universal_smart_vae_video_decode import SmartVAE_StreamingDecoder
from .smartvae_advanced_decoder import SmartVAEAdvancedDecoder

NODE_CLASS_MAPPINGS = {
    "UniversalSmartVAEDecode": UniversalSmartVAEDecode,
    "SmartVAE_StreamingDecoder": SmartVAE_StreamingDecoder,
    "SmartVAE_AdvancedDecoder": SmartVAEAdvancedDecoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalSmartVAEDecode": "🎬 Universal VAE Decode (v11.3)",
    "SmartVAE_StreamingDecoder": "🎞️ SmartVAE Streaming Decoder",
    "SmartVAE_AdvancedDecoder": "🎞️ SmartVAE Advanced Decoder (🧪 Experimental)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
