"""
SmartVAE Advanced Streaming Decoder v2.6
────────────────────────────────────────
Pełna integracja z ltx_video_decoder.py Claude’a (najczystsza wersja)
Zachowuje starą nazwę node’a: SmartVAE_AdvancedDecoder
"""

import torch
import logging
from typing import Optional

import folder_paths
import gc
import comfy.utils
from comfy.model_management import throw_exception_if_processing_interrupted

# Importujemy nową, czystą implementację Claude’a
from .ltx_video_decoder import LTX_VideoDecoder

logger = logging.getLogger(__name__)


class SmartVAEAdvancedDecoder(LTX_VideoDecoder):
    """
    v2.6 – alias do LTX_VideoDecoder Claude’a
    Dzięki temu zachowujemy starą nazwę node’a i wszystkie workflowy działają bez zmian.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Dokładnie te same parametry co LTX_VideoDecoder
        return LTX_VideoDecoder.INPUT_TYPES()

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview_thumbs", "video_path")
    FUNCTION = "decode"
    OUTPUT_NODE = True
    CATEGORY = "latent/video"

    def __init__(self):
        super().__init__()

    # decode jest już w pełni zaimplementowany w LTX_VideoDecoder
    # – nie musimy nic nadpisywać


# Rejestracja – ta sama nazwa co wcześniej
NODE_CLASS_MAPPINGS = {
    "SmartVAE_AdvancedDecoder": SmartVAEAdvancedDecoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartVAE_AdvancedDecoder": "🎞️ SmartVAE Advanced Decoder (v2.6 - Claude Clean)",
}
