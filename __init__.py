# __init__.py

from .UniversalSmartVAEDecode import NODE_CLASS_MAPPINGS as N1, NODE_DISPLAY_NAME_MAPPINGS as D1
from .universal_smart_vae_video_decode import NODE_CLASS_MAPPINGS as N2, NODE_DISPLAY_NAME_MAPPINGS as D2

# Łączymy mapowania – zakładając, że klucze się nie pokrywają
NODE_CLASS_MAPPINGS = {**N1, **N2}
NODE_DISPLAY_NAME_MAPPINGS = {**D1, **D2}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
