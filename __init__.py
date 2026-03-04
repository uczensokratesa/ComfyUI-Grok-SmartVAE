"""
ComfyUI-Grok-SmartVAE
Custom nodes for advanced VAE decoding and latent handling
"""

from .UniversalSmartVAEDecode import NODE_CLASS_MAPPINGS as UniversalSmartVAEDecode_mappings, \
                                     NODE_DISPLAY_NAME_MAPPINGS as UniversalSmartVAEDecode_display

from .universal_smart_vae_video_decode import NODE_CLASS_MAPPINGS as StreamingDecoder_mappings, \
                                              NODE_DISPLAY_NAME_MAPPINGS as StreamingDecoder_display

from .smartvae_advanced_decoder import NODE_CLASS_MAPPINGS as AdvancedDecoder_mappings, \
                                       NODE_DISPLAY_NAME_MAPPINGS as AdvancedDecoder_display

from .advanced_load_latent import NODE_CLASS_MAPPINGS as LoadLatent_mappings, \
                              NODE_DISPLAY_NAME_MAPPINGS as LoadLatent_display


# Łączymy wszystkie mapowania w jeden słownik (ComfyUI tego oczekuje)
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(UniversalSmartVAEDecode_mappings)
NODE_CLASS_MAPPINGS.update(StreamingDecoder_mappings)
NODE_CLASS_MAPPINGS.update(AdvancedDecoder_mappings)
NODE_CLASS_MAPPINGS.update(LoadLatent_mappings)


NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(UniversalSmartVAEDecode_display)
NODE_DISPLAY_NAME_MAPPINGS.update(StreamingDecoder_display)
NODE_DISPLAY_NAME_MAPPINGS.update(AdvancedDecoder_display)
NODE_DISPLAY_NAME_MAPPINGS.update(LoadLatent_display)


# Opcjonalnie – jeśli chcesz mieć czytelniejszą nazwę pakietu w menu ComfyUI
WEB_DIRECTORY = "./web"   # jeśli masz jakieś rozszerzenia JS/CSS w folderze web

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
