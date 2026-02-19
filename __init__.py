"""
ComfyUI-Grok-SmartVAE
Production VAE decoders for long video sequences
"""

from .UniversalSmartVAEDecode import UniversalSmartVAEDecode
from .universal_smart_vae_video_decode import UniversalSmartVAEVideoDecode

NODE_CLASS_MAPPINGS = {
    "UniversalSmartVAEDecode": UniversalSmartVAEDecode,
    "UniversalSmartVAEVideoDecode": UniversalSmartVAEVideoDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalSmartVAEDecode": "🎬 Universal VAE Decode (v11.3 + Ignore Warnings)",
    "UniversalSmartVAEVideoDecode": "🎬 Universal VAE Video Decode (Streaming)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
