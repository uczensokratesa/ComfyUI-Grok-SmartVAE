"""
Advanced Load Latent + Metadata Viewer v1.7 + Viewer v2.3
Author: Grok 4.2
Naprawiono na zawsze: smart format detection (safe_open first → pickle fallback)
Działa na wszystkich .latent (nowe + stare) i .safetensors
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
    """Advanced Load Latent v1.7 – Smart Format Detection"""
    
    @classmethod
    def INPUT_TYPES(cls):
        latent_files = scan_latent_files()
        return {
            "required": {
                "latent_file": (["[Manual path...]"] + latent_files, {
                    "default": "[Manual path...]",
                    "tooltip": "Wybierz z listy LUB wpisz ścieżkę poniżej (manual ma priorytet)"
                }),
            },
            "optional": {
                "manual_path": ("STRING", {"default": "", "tooltip": "Pełna ścieżka – ma absolutny priorytet"}),
                "show_metadata": ("BOOLEAN", {"default": True}),
                "refresh_list": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("samples", "metadata_json")
    FUNCTION = "load"
    CATEGORY = "latent"
    DESCRIPTION = "Advanced Load Latent v1.7 – Smart safetensors/pickle detection"
    
    @classmethod
    def IS_CHANGED(cls, latent_file, manual_path="", show_metadata=True, refresh_list=False):
        return (latent_file, manual_path.strip(), float(refresh_list))
    
    def load(self, latent_file, manual_path="", show_metadata=True, refresh_list=False):
        output_dir = folder_paths.get_output_directory()
        latents_dir = os.path.join(output_dir, "latents")
        
        if manual_path and manual_path.strip():
            file_path = manual_path.strip()
            mode = "manual"
        elif latent_file and latent_file != "[Manual path...]":
            file_path = os.path.join(latents_dir, latent_file)
            mode = "dropdown"
        else:
            raise ValueError("Nie wybrano pliku latent.")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Plik nie istnieje: {file_path}")
        
        # === SMART LOAD: zawsze najpierw safetensors ===
        latent_data = {}
        raw_metadata = {}
        format_used = "unknown"
        
        if SAFETENSORS_AVAILABLE:
            try:
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    latent_data = {k: f.get_tensor(k) for k in f.keys()}
                    raw_metadata = f.metadata() or {}
                format_used = "safetensors"
            except Exception:
                pass  # to nie safetensors → idziemy do pickle
        
        if not latent_data:
            try:
                latent_data = torch.load(file_path, map_location="cpu", weights_only=False)
                format_used = "pickle"
            except Exception as e:
                raise RuntimeError(f"Nie udało się wczytać pliku (nawet jako pickle):\n{file_path}\nBłąd: {e}")
        
        # Metadata
        metadata = raw_metadata or {}
        if not metadata:
            if isinstance(latent_data.get("metadata"), dict):
                metadata = latent_data["metadata"]
            elif isinstance(latent_data.get("metadata"), str):
                try:
                    metadata = json.loads(latent_data["metadata"])
                except:
                    pass
        
        # Tensor
        latent_tensor = None
        for key in ["samples", "latent_tensor", "latents", "latent"]:
            if key in latent_data and isinstance(latent_data[key], torch.Tensor):
                latent_tensor = latent_data[key]
                break
        if latent_tensor is None:
            for v in latent_data.values():
                if isinstance(v, torch.Tensor) and v.dim() >= 4:
                    latent_tensor = v
                    break
        
        if latent_tensor is None or latent_tensor.dim() < 4:
            raise ValueError("Nie znaleziono prawidłowego tensora latent (4D+).")
        
        if show_metadata:
            print("=" * 95)
            print(f"📂 LOADED [{mode.upper()}] : {os.path.basename(file_path)}  [{format_used.upper()}]")
            print("=" * 95)
            print(f"   Shape     : {list(latent_tensor.shape)}")
            print(f"   Device    : {latent_tensor.device} | Dtype: {latent_tensor.dtype}")
            for k in ["generation_seed", "seed", "steps", "cfg", "sampler_name", "scheduler", "model_name", "positive_prompt"]:
                if k in metadata:
                    print(f"   {k:15}: {metadata[k]}")
            print("=" * 95)
        
        samples = {"samples": latent_tensor}
        if "batch_index" in latent_data:
            samples["batch_index"] = latent_data["batch_index"]
        
        metadata_json = json.dumps(metadata, indent=2, ensure_ascii=False) if metadata else "{}"
        return (samples, metadata_json)


class LatentMetadataViewer:
    """Advanced Latent Metadata Viewer v2.3 – Smart Format Detection"""
    
    @classmethod
    def INPUT_TYPES(cls):
        latent_files = scan_latent_files()
        return {
            "required": {
                "latent_file": (["[Select file...]"] + latent_files, {"default": "[Select file...]"}),
            },
            "optional": {
                "refresh_list": ("BOOLEAN", {"default": False})
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("full_report", "shape", "seed", "model", "timestamp", "format")
    FUNCTION = "view"
    OUTPUT_NODE = True
    CATEGORY = "latent"
    DESCRIPTION = "📋 Advanced Latent Metadata Viewer v2.3 (SMART LOAD)"
    
    @classmethod
    def IS_CHANGED(cls, latent_file, refresh_list=False):
        return float(refresh_list)
    
    def view(self, latent_file, refresh_list=False):
        if latent_file == "[Select file...]":
            return ("No file selected", "N/A", "N/A", "N/A", "N/A", "N/A")
        
        output_dir = folder_paths.get_output_directory()
        latents_dir = os.path.join(output_dir, "latents")
        file_path = os.path.join(latents_dir, latent_file)
        
        if not os.path.exists(file_path):
            err = f"File not found: {file_path}"
            return (err, "N/A", "N/A", "N/A", "N/A", "N/A")
        
        try:
            latent_data = {}
            metadata = {}
            format_used = "unknown"
            
            # SMART LOAD – dokładnie tak jak w Gemini + moje poprawki
            if SAFETENSORS_AVAILABLE:
                try:
                    with safe_open(file_path, framework="pt", device="cpu") as f:
                        latent_data = {k: f.get_tensor(k) for k in f.keys()}
                        metadata = f.metadata() or {}
                    format_used = "safetensors"
                except Exception:
                    pass
            
            if not metadata and not latent_data:  # fallback pickle
                latent_data = torch.load(file_path, map_location="cpu", weights_only=False)
                format_used = "pickle"
                if isinstance(latent_data.get("metadata"), dict):
                    metadata = latent_data["metadata"]
                elif isinstance(latent_data.get("metadata"), str):
                    try:
                        metadata = json.loads(latent_data["metadata"])
                    except:
                        metadata = {}
            
            # Tensor do shape
            tensor = None
            for key in ["samples", "latent_tensor", "latents", "latent"]:
                if key in latent_data and isinstance(latent_data[key], torch.Tensor):
                    tensor = latent_data[key]
                    break
            if tensor is None:
                for v in latent_data.values():
                    if isinstance(v, torch.Tensor) and v.dim() >= 3:
                        tensor = v
                        break
            
            shape = str(list(tensor.shape)) if tensor is not None else "Unknown"
            seed = str(metadata.get("generation_seed") or metadata.get("seed") or "Unknown")
            model = str(metadata.get("model_name") or metadata.get("base_model") or "Unknown")
            timestamp = str(metadata.get("timestamp") or metadata.get("generation_timestamp") or "Unknown")
            
            report = f"""📋 LATENT METADATA VIEWER v2.3
══════════════════════════════════════════════════════
Plik          : {latent_file}
Format        : {format_used.upper()}
Shape         : {shape}
Seed          : {seed}
Model         : {model}
Timestamp     : {timestamp}

Dodatkowe informacje:
"""
            for key in ["steps", "cfg", "sampler_name", "scheduler", "denoise", "positive_prompt", "negative_prompt"]:
                if key in metadata:
                    value = str(metadata[key])
                    if len(value) > 120:
                        value = value[:117] + "..."
                    report += f"   {key:14}: {value}\n"
            
            print(report)
            return (report, shape, seed, model, timestamp, format_used)
        
        except Exception as e:
            err = f"Error loading metadata: {str(e)}"
            print(err)
            return (err, "Error", "Error", "Error", "Error", "Error")


NODE_CLASS_MAPPINGS = {
    "AdvancedLoadLatent": AdvancedLoadLatent,
    "LatentMetadataViewer": LatentMetadataViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedLoadLatent": "📂 Load Latent (Advanced) v1.7",
    "LatentMetadataViewer": "📋 Advanced Latent Metadata Viewer v2.3",
}
