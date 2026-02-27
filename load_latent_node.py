"""
Advanced Load Latent Node v1.4
Author: Grok 4.2
Fix: Manual path ma absolutny priorytet + poprawione UI + pełne wcięcia
"""

import os
import json
import torch
import folder_paths

# Safe safetensors import
try:
    from safetensors.torch import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("⚠️  safetensors not available - falling back to pickle only")


def scan_latent_files():
    """Scan output/latents for .latent and .safetensors files."""
    output_dir = folder_paths.get_output_directory()
    latents_dir = os.path.join(output_dir, "latents")
    
    latent_files = []
    if os.path.exists(latents_dir):
        for root, dirs, files in os.walk(latents_dir):
            for file in files:
                if file.endswith((".latent", ".safetensors")):
                    rel_path = os.path.relpath(os.path.join(root, file), latents_dir)
                    latent_files.append(rel_path)
    
    latent_files.sort(
        key=lambda f: os.path.getmtime(os.path.join(latents_dir, f)) 
                     if os.path.exists(os.path.join(latents_dir, f)) else 0,
        reverse=True
    )
    
    return latent_files


class AdvancedLoadLatent:
    """Advanced Load Latent v1.4 – Manual path ma absolutny priorytet"""
    
    @classmethod
    def INPUT_TYPES(cls):
        latent_files = scan_latent_files()
        dropdown_options = ["[Manual path...]"] + latent_files
        
        return {
            "required": {
                "latent_file": (dropdown_options, {
                    "default": "[Manual path...]",
                    "tooltip": "Wybierz z listy LUB wpisz ścieżkę poniżej (manual ma priorytet)"
                }),
            },
            "optional": {
                "manual_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Pełna ścieżka – jeśli niepusta, ZAWSZE jest używana"
                }),
                "show_metadata": ("BOOLEAN", {"default": True}),
                "refresh_list": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("samples", "metadata_json")
    FUNCTION = "load"
    CATEGORY = "latent"
    DESCRIPTION = "Advanced Load Latent v1.4 – Grok 4.2 (Manual priority)"
    
    @classmethod
    def IS_CHANGED(cls, latent_file, manual_path="", show_metadata=True, refresh_list=False):
        return (latent_file, manual_path.strip(), float(refresh_list))
    
    def load(self, latent_file, manual_path="", show_metadata=True, refresh_list=False):
        output_dir = folder_paths.get_output_directory()
        latents_dir = os.path.join(output_dir, "latents")
        
        # === PRIORYTET MANUAL PATH ===
        if manual_path and manual_path.strip():
            file_path = manual_path.strip()
            mode = "manual"
            if show_metadata:
                print(f"🔧 MANUAL PATH mode: {file_path}")
        elif latent_file and latent_file != "[Manual path...]":
            file_path = os.path.join(latents_dir, latent_file)
            mode = "dropdown"
            if show_metadata:
                print(f"📂 DROPDOWN mode: {latent_file}")
        else:
            raise ValueError(
                "Nie wybrano pliku.\n"
                "• Albo wybierz plik z listy rozwijanej\n"
                "• Albo wpisz pełną ścieżkę w polu 'manual_path'"
            )
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Plik nie istnieje: {file_path}\nTryb: {mode.upper()}")
        
        # === LOAD LOGIC (safetensors + pickle) ===
        latent_data = {}
        raw_metadata = {}
        format_used = "unknown"
        
        if SAFETENSORS_AVAILABLE:
            try:
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    latent_data = {k: f.get_tensor(k) for k in f.keys()}
                    raw_metadata = f.metadata() or {}
                format_used = "safetensors"
                if show_metadata:
                    print(f"✓ Loaded as safetensors: {os.path.basename(file_path)}")
            except Exception as e:
                if show_metadata:
                    print(f"⚠️ Safetensors failed, trying pickle... ({e})")
        
        if not latent_data:
            try:
                latent_data = torch.load(file_path, map_location="cpu", weights_only=False)
                format_used = "pickle"
                if show_metadata:
                    print(f"✓ Loaded as pickle: {os.path.basename(file_path)}")
            except Exception as e:
                raise RuntimeError(f"Failed to load file: {file_path}\nError: {e}")
        
        # Metadata
        metadata = {}
        if raw_metadata:
            metadata = raw_metadata
        elif isinstance(latent_data.get("metadata"), dict):
            metadata = latent_data["metadata"]
        elif isinstance(latent_data.get("metadata"), str):
            try:
                metadata = json.loads(latent_data["metadata"])
            except:
                metadata = {"raw": latent_data["metadata"]}
        
        # Extract tensor
        latent_tensor = None
        for key in ["latent_tensor", "samples", "latents"]:
            if key in latent_data and isinstance(latent_data[key], torch.Tensor):
                latent_tensor = latent_data[key]
                break
        if latent_tensor is None:
            for v in latent_data.values():
                if isinstance(v, torch.Tensor) and v.dim() >= 4:
                    latent_tensor = v
                    break
        
        if latent_tensor is None or latent_tensor.dim() < 4:
            raise ValueError(f"No valid 4D+ tensor found. Keys: {list(latent_data.keys())}")
        
        # Console output
        if show_metadata:
            print("=" * 85)
            print(f"📂 LOADED [{mode.upper()}] : {os.path.basename(file_path)}  [{format_used.upper()}]")
            print("=" * 85)
            print(f"   Shape     : {list(latent_tensor.shape)}")
            print(f"   Device    : {latent_tensor.device} | Dtype: {latent_tensor.dtype}")
            for k in ["generation_seed", "seed", "timestamp", "generation_timestamp", "batch_index"]:
                if k in metadata:
                    print(f"   {k:12}: {metadata[k]}")
            print("=" * 85)
        
        samples = {"samples": latent_tensor}
        if "batch_index" in latent_data:
            samples["batch_index"] = latent_data["batch_index"]
        elif "batch_index" in metadata:
            try:
                samples["batch_index"] = int(metadata["batch_index"])
            except:
                pass
        
        metadata_json = json.dumps(metadata, indent=2, ensure_ascii=False) if metadata else "{}"
        
        return (samples, metadata_json)


class LatentMetadataViewer:
    @classmethod
    def INPUT_TYPES(cls):
        latent_files = scan_latent_files()
        dropdown_options = ["[Select file...]"] + latent_files
        return {
            "required": {
                "latent_file": (dropdown_options, {"default": dropdown_options[0]}),
            },
            "optional": {
                "refresh_list": ("BOOLEAN", {"default": False})
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("shape", "seed", "timestamp", "format")
    FUNCTION = "view"
    OUTPUT_NODE = True
    CATEGORY = "latent"
    
    @classmethod
    def IS_CHANGED(cls, latent_file, refresh_list=False):
        return float(refresh_list)
    
    def view(self, latent_file, refresh_list=False):
        if latent_file == "[Select file...]":
            return ("No file selected", "", "", "")
        
        output_dir = folder_paths.get_output_directory()
        latents_dir = os.path.join(output_dir, "latents")
        file_path = os.path.join(latents_dir, latent_file)
        
        if not os.path.exists(file_path):
            return (f"File not found: {file_path}", "", "", "")
        
        try:
            if SAFETENSORS_AVAILABLE:
                try:
                    with safe_open(file_path, framework="pt", device="cpu") as f:
                        raw_metadata = f.metadata() or {}
                        shape = "N/A"
                        for k in f.keys():
                            if "tensor" in k.lower() or "samples" in k.lower():
                                t = f.get_tensor(k)
                                shape = str(list(t.shape))
                                break
                except:
                    raw_metadata = {}
            else:
                raw_metadata = {}
            
            if not raw_metadata:
                data = torch.load(file_path, map_location="cpu", weights_only=False)
                raw_metadata = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
            
            shape = str(raw_metadata.get("latent_shape", "Unknown"))
            seed = str(raw_metadata.get("generation_seed") or raw_metadata.get("seed", "Unknown"))
            timestamp = str(raw_metadata.get("timestamp") or raw_metadata.get("generation_timestamp", "Unknown"))
            fmt = "safetensors" if SAFETENSORS_AVAILABLE and file_path.endswith(".safetensors") else "pickle"
            
            print(f"📋 Metadata Viewer → {latent_file} [{fmt}]")
            return (shape, seed, timestamp, fmt)
        
        except Exception as e:
            err = f"Error: {e}"
            print(err)
            return (err, "", "", "")


NODE_CLASS_MAPPINGS = {
    "AdvancedLoadLatent": AdvancedLoadLatent,
    "LatentMetadataViewer": LatentMetadataViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedLoadLatent": "📂 Load Latent (Advanced) v1.4",
    "LatentMetadataViewer": "📋 Latent Metadata Viewer",
}
