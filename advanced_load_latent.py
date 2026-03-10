"""
Advanced Load Latent + Metadata Viewer v2.4 STABLE
Based on proven CustomLoadLatent core - simple and reliable
Trust the flag, no heuristics, just works
Author: Claude (after debugging session with real-world latents)
"""

import os
import json
import torch
import hashlib
import folder_paths

# Safe safetensors import
try:
    import safetensors.torch
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("⚠️  safetensors not available - install with: pip install safetensors")


def scan_latent_files():
    """Scan output/latents directory for .latent and .safetensors files"""
    output_dir = folder_paths.get_output_directory()
    latents_dir = os.path.join(output_dir, "latents")
    
    latent_files = []
    if os.path.exists(latents_dir):
        for root, dirs, files in os.walk(latents_dir):
            for file in files:
                if file.lower().endswith((".latent", ".safetensors")):
                    rel_path = os.path.relpath(os.path.join(root, file), latents_dir)
                    latent_files.append(rel_path)
    
    # Sort by modification time (newest first)
    latent_files.sort(
        key=lambda f: os.path.getmtime(os.path.join(latents_dir, f)) 
                     if os.path.exists(os.path.join(latents_dir, f)) else 0,
        reverse=True
    )
    return latent_files


class AdvancedLoadLatent:
    """
    Advanced Load Latent v2.4 STABLE
    Core: Exact CustomLoadLatent logic (trust the flag, no overrides)
    Added: Dropdown selector, info display, pickle fallback, metadata extraction
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        latent_files = scan_latent_files()
        return {
            "required": {
                "latent_file": (["[Manual path...]"] + latent_files, {
                    "default": "[Manual path...]",
                }),
            },
            "optional": {
                "manual_path": ("STRING", {
                    "default": "",
                    "tooltip": "Full file path (overrides dropdown)"
                }),
                "show_info": ("BOOLEAN", {"default": True}),
                "refresh_list": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("samples", "info")
    FUNCTION = "load"
    CATEGORY = "latent"
    
    @classmethod
    def IS_CHANGED(cls, latent_file, manual_path="", show_info=True, refresh_list=False):
        output_dir = folder_paths.get_output_directory()
        latents_dir = os.path.join(output_dir, "latents")
        
        if manual_path and manual_path.strip():
            file_path = manual_path.strip()
        elif latent_file and latent_file != "[Manual path...]":
            file_path = os.path.join(latents_dir, latent_file)
        else:
            return float(refresh_list)
        
        if not os.path.exists(file_path):
            return float(refresh_list)
        
        try:
            m = hashlib.sha256()
            with open(file_path, 'rb') as f:
                m.update(f.read())
            return m.digest().hex()
        except:
            return float(refresh_list)
    
    @classmethod
    def VALIDATE_INPUTS(cls, latent_file, manual_path="", **kwargs):
        output_dir = folder_paths.get_output_directory()
        latents_dir = os.path.join(output_dir, "latents")
        
        if manual_path and manual_path.strip():
            file_path = manual_path.strip()
        elif latent_file and latent_file != "[Manual path...]":
            file_path = os.path.join(latents_dir, latent_file)
        else:
            return "No file selected"
        
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"
        
        return True
    
    def load(self, latent_file, manual_path="", show_info=True, refresh_list=False):
        output_dir = folder_paths.get_output_directory()
        latents_dir = os.path.join(output_dir, "latents")
        
        # Determine file path - MANUAL PATH HAS PRIORITY!
        if manual_path and manual_path.strip():
            file_path = manual_path.strip()
            mode = "MANUAL"
        elif latent_file and latent_file != "[Manual path...]":
            file_path = os.path.join(latents_dir, latent_file)
            mode = "DROPDOWN"
        else:
            raise ValueError("No latent file selected")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")
        
        # ===== LOAD FILE (exactly like CustomLoadLatent) =====
        latent = None
        format_used = "unknown"
        
        if SAFETENSORS_AVAILABLE:
            try:
                # Load using safetensors (returns ONLY tensor dict, no metadata)
                latent = safetensors.torch.load_file(file_path, device="cpu")
                format_used = "SAFETENSORS"
            except Exception as e:
                if show_info:
                    print(f"  Safetensors load failed: {e}")
        
        # Fallback to pickle
        if latent is None:
            try:
                latent = torch.load(file_path, map_location="cpu", weights_only=False)
                format_used = "PICKLE"
            except Exception as e:
                raise RuntimeError(f"Failed to load file: {e}")
        
        # ===== APPLY SCALING (exactly like CustomLoadLatent) =====
        # SIMPLE: Trust the flag, no heuristics, no overrides
        multiplier = 1.0
        if "latent_format_version_0" not in latent:
            multiplier = 1.0 / 0.18215
            format_type = "old (SD1.5/SDXL/AnimateDiff)"
        else:
            format_type = "new (SD3/Flux/LTX-Video)"
        
        # ===== EXTRACT LATENT TENSOR =====
        # Priority: "latent_tensor" key first
        if "latent_tensor" in latent and isinstance(latent["latent_tensor"], torch.Tensor):
            latent_tensor = latent["latent_tensor"]
            tensor_key = "latent_tensor"
        elif "samples" in latent and isinstance(latent["samples"], torch.Tensor):
            latent_tensor = latent["samples"]
            tensor_key = "samples"
        else:
            # Search for any valid tensor
            latent_tensor = None
            tensor_key = None
            for k, v in latent.items():
                if isinstance(v, torch.Tensor) and v.dim() >= 4 and v.numel() > 1000:
                    latent_tensor = v
                    tensor_key = k
                    break
            
            if latent_tensor is None:
                raise ValueError(
                    f"No valid latent tensor found.\n"
                    f"Available keys: {list(latent.keys())}"
                )
        
        # Apply scaling and convert to float32
        samples = {"samples": latent_tensor.float() * multiplier}
        
        # Copy batch_index if present
        if "batch_index" in latent:
            samples["batch_index"] = latent["batch_index"]
        # ============================================================
        
        # ===== EXTRACT METADATA (for info output) =====
        metadata = {}
        if isinstance(latent.get("metadata"), dict):
            metadata = latent["metadata"]
        elif isinstance(latent.get("metadata"), str):
            try:
                metadata = json.loads(latent["metadata"])
            except:
                pass
        
        # ===== BUILD INFO OUTPUT =====
        scaled_min = (latent_tensor.min() * multiplier).item()
        scaled_max = (latent_tensor.max() * multiplier).item()
        
        info_lines = [
            f"File: {os.path.basename(file_path)}",
            f"Path: {file_path}",
            f"Mode: {mode}",
            f"Format: {format_used}",
            f"Type: {format_type}",
            f"Has version flag: {'latent_format_version_0' in latent}",
            f"Multiplier: {multiplier:.5f}",
            f"Tensor key: {tensor_key}",
            f"Shape: {list(latent_tensor.shape)}",
            f"Numel: {latent_tensor.numel():,}",
            f"Raw range: [{latent_tensor.min().item():.4f}, {latent_tensor.max().item():.4f}]",
            f"Scaled (to decoder): [{scaled_min:.4f}, {scaled_max:.4f}]",
        ]
        
        if metadata:
            for key in ["seed", "generation_seed", "steps", "cfg", "model_name", "sampler_name"]:
                if key in metadata:
                    val = str(metadata[key])
                    if len(val) > 80:
                        val = val[:77] + "..."
                    info_lines.append(f"{key}: {val}")
        
        info_text = "\n".join(info_lines)
        
        # Display in console if requested
        if show_info:
            print("=" * 78)
            print(f"📂 LOADED LATENT [{mode}]")
            print("=" * 78)
            print(info_text)
            print("=" * 78)
        
        return (samples, info_text)


class LatentMetadataViewer:
    """
    Latent Metadata Viewer v2.4
    Read-only diagnostic viewer for latent file metadata
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        latent_files = scan_latent_files()
        return {
            "required": {
                "latent_file": (["[Select file...]"] + latent_files, {
                    "default": "[Select file...]"
                }),
            },
            "optional": {
                "refresh_list": ("BOOLEAN", {"default": False})
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("full_report", "shape", "seed", "model", "format")
    FUNCTION = "view"
    OUTPUT_NODE = True
    CATEGORY = "latent"
    
    @classmethod
    def IS_CHANGED(cls, latent_file, refresh_list=False):
        return float(refresh_list)
    
    def view(self, latent_file, refresh_list=False):
        if latent_file == "[Select file...]":
            return ("No file selected", "N/A", "N/A", "N/A", "N/A")
        
        output_dir = folder_paths.get_output_directory()
        latents_dir = os.path.join(output_dir, "latents")
        file_path = os.path.join(latents_dir, latent_file)
        
        if not os.path.exists(file_path):
            err = f"File not found: {file_path}"
            return (err, "N/A", "N/A", "N/A", "N/A")
        
        try:
            # Load file
            latent_data = None
            metadata = {}
            format_used = "unknown"
            
            if SAFETENSORS_AVAILABLE:
                try:
                    # For viewer, use safe_open to get metadata too
                    from safetensors.torch import safe_open
                    
                    with safe_open(file_path, framework="pt", device="cpu") as f:
                        latent_data = {k: f.get_tensor(k) for k in f.keys()}
                        metadata = f.metadata() or {}
                    format_used = "SAFETENSORS"
                except Exception as e:
                    # Fallback to load_file
                    try:
                        latent_data = safetensors.torch.load_file(file_path, device="cpu")
                        format_used = "SAFETENSORS"
                    except:
                        pass
            
            if latent_data is None:
                latent_data = torch.load(file_path, map_location="cpu", weights_only=False)
                format_used = "PICKLE"
                
                # Extract metadata from pickle
                if isinstance(latent_data.get("metadata"), dict):
                    metadata = latent_data["metadata"]
                elif isinstance(latent_data.get("metadata"), str):
                    try:
                        metadata = json.loads(latent_data["metadata"])
                    except:
                        pass
            
            # Find main latent tensor
            tensor = None
            tensor_key = None
            
            for key in ["latent_tensor", "samples", "latents", "latent"]:
                if key in latent_data and isinstance(latent_data[key], torch.Tensor):
                    candidate = latent_data[key]
                    if candidate.dim() >= 4:
                        tensor = candidate
                        tensor_key = key
                        break
            
            if tensor is None:
                for k, v in latent_data.items():
                    if isinstance(v, torch.Tensor) and v.dim() >= 4:
                        tensor = v
                        tensor_key = k
                        break
            
            # Extract info
            shape = str(list(tensor.shape)) if tensor is not None else "Unknown"
            seed = str(metadata.get("seed") or metadata.get("generation_seed") or "Unknown")
            model = str(metadata.get("model_name") or metadata.get("base_model") or "Unknown")
            
            # Check format flags
            has_version_flag = "latent_format_version_0" in latent_data
            
            if has_version_flag:
                format_type = "new (SD3/Flux/LTX)"
            else:
                format_type = "old (SD1.5/SDXL/AnimateDiff)"
            
            # Build report
            report_lines = [
                "=" * 70,
                f"LATENT METADATA VIEWER: {latent_file}",
                "=" * 70,
                f"File format: {format_used}",
                f"Latent type: {format_type}",
                f"Has version flag: {has_version_flag}",
                f"Main tensor key: {tensor_key}",
                f"Shape: {shape}",
                f"Seed: {seed}",
                f"Model: {model}",
            ]
            
            if tensor is not None:
                report_lines.append(f"Numel: {tensor.numel():,}")
                report_lines.append(f"Value range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
            
            # Additional metadata
            if metadata:
                report_lines.append("")
                report_lines.append("Additional metadata:")
                for key in ["steps", "cfg", "sampler_name", "scheduler", "denoise", "positive_prompt", "negative_prompt"]:
                    if key in metadata:
                        val = str(metadata[key])
                        if len(val) > 90:
                            val = val[:87] + "..."
                        report_lines.append(f"  {key}: {val}")
            
            # All tensor keys (diagnostic)
            report_lines.append("")
            report_lines.append("All tensor keys in file:")
            for k, v in latent_data.items():
                if isinstance(v, torch.Tensor):
                    report_lines.append(f"  {k}: shape={list(v.shape)}, numel={v.numel():,}")
            
            report_lines.append("=" * 70)
            report = "\n".join(report_lines)
            
            print(report)
            return (report, shape, seed, model, format_used)
        
        except Exception as e:
            err = f"Error loading metadata: {str(e)}"
            print(err)
            import traceback
            traceback.print_exc()
            return (err, "Error", "Error", "Error", "Error")


# ────────────────────────────────────────────────
#  Node registration
# ────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "AdvancedLoadLatent": AdvancedLoadLatent,
    "LatentMetadataViewer": LatentMetadataViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedLoadLatent": "📂 Load Latent (Advanced) v2.4 STABLE",
    "LatentMetadataViewer": "📋 Latent Metadata Viewer v2.4",
}
