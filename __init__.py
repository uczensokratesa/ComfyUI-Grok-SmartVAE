"""
ComfyUI-Grok-SmartVAE
Production VAE decoders for long video sequences
"""

from .UniversalSmartVAEDecode import UniversalSmartVAEDecode
from .universal_smart_vae_video_decode import SmartVAE_StreamingDecoder

NODE_CLASS_MAPPINGS = {
    "UniversalSmartVAEDecode": UniversalSmartVAEDecode,
    "SmartVAE_StreamingDecoder":SmartVAE_StreamingDecoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalSmartVAEDecode": "🎬 Universal VAE Decode (v11.3 + Ignore Warnings)",
    "SmartVAE_StreamingDecoder": "🎬 SmartVAE_StreamingDecoder (Streaming)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
