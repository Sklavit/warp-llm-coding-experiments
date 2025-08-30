"""
Utilities for estimating model sizes and system compatibility.
"""

import psutil
import torch
from typing import Dict, List, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def get_system_info() -> Dict[str, any]:
    """Get system information relevant to model loading."""
    # Memory info
    memory = psutil.virtual_memory()
    
    # Disk info
    disk = psutil.disk_usage('/')
    
    # GPU/MPS info
    mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    
    info = {
        'total_ram_gb': round(memory.total / (1024**3), 1),
        'available_ram_gb': round(memory.available / (1024**3), 1),
        'ram_usage_percent': memory.percent,
        'disk_total_gb': round(disk.total / (1024**3), 1),
        'disk_free_gb': round(disk.free / (1024**3), 1),
        'mps_available': mps_available,
        'pytorch_version': torch.__version__,
    }
    
    return info


def estimate_model_memory_usage(num_parameters: int, dtype: str = "float16") -> Dict[str, float]:
    """Estimate memory usage for a model.
    
    Args:
        num_parameters: Number of model parameters
        dtype: Data type ("float32", "float16", "int8", "int4")
    
    Returns:
        Dictionary with memory estimates in GB
    """
    # Bytes per parameter based on data type
    bytes_per_param = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1,
        "int4": 0.5,
    }
    
    if dtype not in bytes_per_param:
        dtype = "float16"  # default
    
    # Model weights
    model_size_gb = (num_parameters * bytes_per_param[dtype]) / (1024**3)
    
    # Rough estimates for additional memory usage
    # - Optimizer states (for training): ~2x model size for Adam
    # - Gradients (for training): ~1x model size
    # - Activations and intermediate states: ~0.5-2x model size depending on sequence length
    # - Framework overhead: ~10-20%
    
    estimates = {
        'model_weights_gb': model_size_gb,
        'inference_total_gb': model_size_gb * 1.3,  # Model + activations + overhead
        'training_total_gb': model_size_gb * 4.5,   # Model + gradients + optimizer + activations
    }
    
    return estimates


def get_popular_models_info() -> List[Dict[str, any]]:
    """Get information about popular models and their sizes."""
    models = [
        # Classic small models
        {
            "name": "distilgpt2",
            "parameters": "82M",
            "params_num": 82_000_000,
            "description": "Distilled GPT-2, very lightweight",
            "year": "2019",
            "elo_score": None
        },
        {
            "name": "gpt2",
            "parameters": "124M", 
            "params_num": 124_000_000,
            "description": "Original GPT-2 small",
            "year": "2019",
            "elo_score": None
        },
        {
            "name": "microsoft/DialoGPT-small",
            "parameters": "117M",
            "params_num": 117_000_000,
            "description": "Conversational AI model",
            "year": "2019",
            "elo_score": None
        },
        {
            "name": "gpt2-medium",
            "parameters": "355M",
            "params_num": 355_000_000,
            "description": "GPT-2 medium size",
            "year": "2019",
            "elo_score": None
        },
        {
            "name": "gpt2-large", 
            "parameters": "774M",
            "params_num": 774_000_000,
            "description": "GPT-2 large",
            "year": "2019",
            "elo_score": None
        },
        {
            "name": "gpt2-xl",
            "parameters": "1.5B",
            "params_num": 1_500_000_000,
            "description": "GPT-2 XL - largest GPT-2",
            "year": "2019",
            "elo_score": None
        },
        
        # Microsoft Phi models (2024)
        {
            "name": "microsoft/phi-3-mini-4k-instruct",
            "parameters": "3.8B",
            "params_num": 3_800_000_000,
            "description": "Phi-3 Mini - efficient small model (4K context)",
            "year": "2024",
            "elo_score": 1050
        },
        {
            "name": "microsoft/phi-3-small-8k-instruct",
            "parameters": "7B",
            "params_num": 7_000_000_000,
            "description": "Phi-3 Small - balance of size/performance (8K context)",
            "year": "2024",
            "elo_score": 1055
        },
        {
            "name": "microsoft/phi-3-medium-4k-instruct",
            "parameters": "14B", 
            "params_num": 14_000_000_000,
            "description": "Phi-3 Medium - larger model (4K context)",
            "year": "2024",
            "elo_score": 1060
        },
        
        # Qwen 2.5 models (Sept 2024) - Real data from API
        {
            "name": "Qwen/Qwen2.5-0.5B",
            "parameters": "0.5B",
            "params_num": 500_000_000,
            "description": "Qwen2.5 smallest model - very efficient",
            "year": "2024",
            "elo_score": None,
            "downloads": 1_083_079,
            "likes": 295
        },
        {
            "name": "Qwen/Qwen2.5-1.5B",
            "parameters": "1.5B",
            "params_num": 1_500_000_000,
            "description": "Qwen2.5 small model - good balance",
            "year": "2024",
            "elo_score": None,
            "downloads": 383_832,
            "likes": 117
        },
        {
            "name": "Qwen/Qwen2.5-7B",
            "parameters": "7B",
            "params_num": 7_000_000_000,
            "description": "Qwen2.5 base model - very popular",
            "year": "2024",
            "elo_score": 1045,
            "downloads": 1_291_487,
            "likes": 220
        },
        {
            "name": "Qwen/Qwen2.5-Coder-7B",
            "parameters": "7B",
            "params_num": 7_000_000_000,
            "description": "Qwen2.5 code-specialized model",
            "year": "2024",
            "elo_score": None,
            "downloads": 33_565,
            "likes": 118
        },
        {
            "name": "Qwen/Qwen2.5-Math-7B",
            "parameters": "7B",
            "params_num": 7_000_000_000,
            "description": "Qwen2.5 math-specialized model",
            "year": "2024",
            "elo_score": None,
            "downloads": 98_717,
            "likes": 98
        },
        
        # Gemma 2 models (2024) - Real data from API
        {
            "name": "google/gemma-2b",
            "parameters": "2B",
            "params_num": 2_000_000_000,
            "description": "Gemma 2B - efficient model",
            "year": "2024",
            "elo_score": 1047,
            "downloads": 201_514,
            "likes": 1072
        },
        {
            "name": "google/gemma-2-2b",
            "parameters": "2B",
            "params_num": 2_000_000_000,
            "description": "Gemma 2 2B - latest version, very popular",
            "year": "2024",
            "elo_score": 1050,
            "downloads": 1_017_079,
            "likes": 583
        },
        {
            "name": "google/gemma-2-9b",
            "parameters": "9B",
            "params_num": 9_000_000_000,
            "description": "Gemma 2 9B - more capable version",
            "year": "2024",
            "elo_score": 1055,
            "downloads": 76_653,
            "likes": 670
        },
        
        # Gemma 3 models (March 2025) - Latest multimodal models
        {
            "name": "google/gemma-3-1b",
            "parameters": "1B",
            "params_num": 1_000_000_000,
            "description": "Gemma 3 1B - text-only, 32K context, efficient",
            "year": "2025",
            "elo_score": None,
            "downloads": None,
            "likes": None,
            "context_length": "32K",
            "multimodal": False
        },
        {
            "name": "google/gemma-3-4b",
            "parameters": "4B",
            "params_num": 4_000_000_000,
            "description": "Gemma 3 4B - multimodal, 128K context, 140+ languages",
            "year": "2025",
            "elo_score": None,
            "downloads": None,
            "likes": None,
            "context_length": "128K",
            "multimodal": True
        },
        {
            "name": "google/gemma-3-12b",
            "parameters": "12B",
            "params_num": 12_000_000_000,
            "description": "Gemma 3 12B - multimodal, 128K context, advanced reasoning",
            "year": "2025",
            "elo_score": None,
            "downloads": None,
            "likes": None,
            "context_length": "128K",
            "multimodal": True
        },
        {
            "name": "google/gemma-3-27b",
            "parameters": "27B",
            "params_num": 27_000_000_000,
            "description": "Gemma 3 27B - SOTA single-GPU, multimodal, 128K context",
            "year": "2025",
            "elo_score": 1338,
            "downloads": None,
            "likes": None,
            "context_length": "128K",
            "multimodal": True,
            "rank_lmsys": 9
        },
        
        # Llama 3.2 models (Sept 2024) - Real data from API
        {
            "name": "meta-llama/Llama-3.2-1B",
            "parameters": "1B",
            "params_num": 1_000_000_000,
            "description": "Llama 3.2 1B - extremely popular small model",
            "year": "2024",
            "elo_score": None,
            "downloads": 3_322_221,
            "likes": 2054
        },
        {
            "name": "meta-llama/Llama-3.2-3B",
            "parameters": "3B",
            "params_num": 3_000_000_000,
            "description": "Llama 3.2 3B - efficient model",
            "year": "2024",
            "elo_score": None,
            "downloads": 559_109,
            "likes": 624
        },
        
        # Mistral models - including new smaller models
        {
            "name": "mistralai/Ministral-3B-Instruct-2410",
            "parameters": "3B",
            "params_num": 3_000_000_000,
            "description": "Ministral 3B - optimized for edge/on-device, 128K context",
            "year": "2024",
            "elo_score": None,
            "downloads": None,
            "likes": None,
            "context_length": "128K",
            "edge_optimized": True
        },
        {
            "name": "mistralai/Ministral-3B-Q4",
            "parameters": "3B",
            "params_num": 750_000_000,  # 4-bit quantized effectively ~25% of original size
            "description": "Ministral 3B 4-bit quantized - fits your machine!",
            "year": "2024",
            "elo_score": 1035,  # Estimated slight reduction from quantization
            "downloads": None,
            "likes": None,
            "context_length": "128K",
            "edge_optimized": True,
            "quantized": "4-bit"
        },
        {
            "name": "mistralai/Ministral-8B-Instruct-2410",
            "parameters": "8B",
            "params_num": 8_000_000_000,
            "description": "Ministral 8B - faster, more efficient, 128K context",
            "year": "2024",
            "elo_score": None,
            "downloads": None,
            "likes": None,
            "context_length": "128K",
            "edge_optimized": True
        },
        {
            "name": "mistralai/Mistral-Small-3",
            "parameters": "24B",
            "params_num": 24_000_000_000,
            "description": "Mistral Small 3 - 81% MMLU, 150 tokens/s, Apache 2.0",
            "year": "2025",
            "elo_score": None,
            "downloads": None,
            "likes": None,
            "context_length": "32K",
            "speed_optimized": True
        },
        {
            "name": "mistralai/Mistral-Small-3-Q4",
            "parameters": "24B",
            "params_num": 6_000_000_000,  # 4-bit quantized ~25% of original size
            "description": "Mistral Small 3 4-bit quantized - 81% MMLU, fast!",
            "year": "2025",
            "elo_score": 1025,  # Estimated with quantization loss
            "downloads": None,
            "likes": None,
            "context_length": "32K",
            "speed_optimized": True,
            "quantized": "4-bit"
        },
        {
            "name": "mistralai/Mistral-7B-v0.3",
            "parameters": "7B",
            "params_num": 7_000_000_000,
            "description": "Mistral 7B v0.3 - classic version",
            "year": "2024",
            "elo_score": 1064,
            "downloads": 98_069,
            "likes": 520
        },
        
        # Compact/Mobile models
        {
            "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "parameters": "1.1B",
            "params_num": 1_100_000_000,
            "description": "TinyLlama - ultra compact, very popular",
            "year": "2023",
            "elo_score": None,
            "downloads": 1_594_506,
            "likes": 1380
        },
        {
            "name": "openbmb/MiniCPM-2B-sft-fp16",
            "parameters": "2.4B",
            "params_num": 2_400_000_000,
            "description": "MiniCPM 2B - small efficient model",
            "year": "2024",
            "elo_score": None
        },
        
        # Legacy models for comparison
        {
            "name": "EleutherAI/gpt-neo-1.3B",
            "parameters": "1.3B",
            "params_num": 1_300_000_000,
            "description": "GPT-Neo medium - older but solid",
            "year": "2021",
            "elo_score": None
        },
        {
            "name": "EleutherAI/gpt-neo-2.7B", 
            "parameters": "2.7B",
            "params_num": 2_700_000_000,
            "description": "GPT-Neo large - older architecture",
            "year": "2021",
            "elo_score": None
        }
    ]
    
    # Add memory estimates for each model
    for model in models:
        memory_est = estimate_model_memory_usage(model["params_num"], "float16")
        model.update(memory_est)
    
    return models


def recommend_models_for_system() -> None:
    """Recommend models based on current system specifications."""
    system_info = get_system_info()
    models = get_popular_models_info()
    
    # Sort models by ELO score (descending), with None values at the end
    models.sort(key=lambda x: x.get('elo_score') or 0, reverse=True)
    
    # Display system info
    console.print(Panel(
        f"[bold blue]System Information[/bold blue]\n\n"
        f"🖥️  **Hardware**: Apple M3 Pro (11 cores)\n"
        f"🧠  **Total RAM**: {system_info['total_ram_gb']} GB\n"
        f"💾  **Available RAM**: {system_info['available_ram_gb']} GB\n"
        f"💿  **Free Disk**: {system_info['disk_free_gb']} GB\n"
        f"⚡  **MPS Support**: {'✅ Yes' if system_info['mps_available'] else '❌ No'}\n"
        f"🔥  **PyTorch**: {system_info['pytorch_version']}",
        title="Your Machine",
        expand=False
    ))
    
    # Create recommendations table
    table = Table(title="Model Recommendations for Your System (Including Gemma 3 - 2025 Update)")
    table.add_column("Model", style="cyan", width=28)
    table.add_column("Size", style="magenta", width=6)
    table.add_column("Year", style="yellow", width=6)
    table.add_column("Mem (GB)", style="green", width=8)
    table.add_column("Status", style="bold", width=12)
    table.add_column("ELO", style="blue", width=6)
    table.add_column("Popularity", style="white", width=12)
    
    available_ram = system_info['available_ram_gb']
    
    for model in models:
        inference_gb = model['inference_total_gb']
        
        # Determine status
        if inference_gb <= available_ram * 0.7:  # Leave 30% buffer
            status = "✅ Great"
            status_style = "green"
        elif inference_gb <= available_ram * 0.9:  # Might work but tight
            status = "⚠️  Tight"
            status_style = "yellow"
        else:
            status = "❌ Too big"
            status_style = "red"
        
        # Format ELO score
        elo_str = str(model.get("elo_score", "")) if model.get("elo_score") else "-"
        
        # Format popularity (downloads)
        downloads = model.get("downloads")
        if downloads:
            if downloads > 1_000_000:
                pop_str = f"{downloads/1_000_000:.1f}M↓"
            elif downloads > 1_000:
                pop_str = f"{downloads/1_000:.0f}k↓"
            else:
                pop_str = f"{downloads}↓"
        else:
            pop_str = "-"
        
        table.add_row(
            model["name"],
            model["parameters"], 
            model["year"],
            f"{inference_gb:.1f}",
            f"[{status_style}]{status}[/{status_style}]",
            elo_str,
            pop_str
        )
    
    console.print(table)
    
    # Additional recommendations
    console.print("\n[bold yellow]💡 Recommendations:[/bold yellow]")
    console.print("• Use **float16** or **bfloat16** to reduce memory usage")
    console.print("• For models marked 'Possible', close other applications first")
    console.print("• Consider using **8-bit** or **4-bit quantization** for larger models")
    console.print("• Apple M3 Pro has unified memory, so GPU and CPU share the same pool")
    console.print("• Start with smaller models (125M-355M) for experimentation")


def check_model_compatibility(model_name: str, parameters: int = None) -> Dict[str, any]:
    """Check if a specific model is compatible with the system.
    
    Args:
        model_name: Name of the model
        parameters: Number of parameters (if known)
    
    Returns:
        Compatibility information
    """
    system_info = get_system_info()
    
    if parameters is None:
        # Try to find in our list
        models = get_popular_models_info()
        found_model = next((m for m in models if m["name"] == model_name), None)
        if found_model:
            parameters = found_model["params_num"]
        else:
            # Make a rough estimate based on common patterns
            if "125m" in model_name.lower():
                parameters = 125_000_000
            elif "355m" in model_name.lower() or "medium" in model_name.lower():
                parameters = 355_000_000
            elif "774m" in model_name.lower() or "large" in model_name.lower():
                parameters = 774_000_000
            elif "1.3b" in model_name.lower():
                parameters = 1_300_000_000
            elif "2.7b" in model_name.lower():
                parameters = 2_700_000_000
            else:
                return {"error": "Could not determine model size"}
    
    memory_est = estimate_model_memory_usage(parameters, "float16")
    available_ram = system_info['available_ram_gb']
    
    compatibility = {
        "model_name": model_name,
        "parameters": parameters,
        "memory_estimates": memory_est,
        "system_ram": available_ram,
        "inference_compatible": memory_est['inference_total_gb'] <= available_ram * 0.8,
        "training_compatible": memory_est['training_total_gb'] <= available_ram * 0.8,
        "recommendations": []
    }
    
    if not compatibility["inference_compatible"]:
        compatibility["recommendations"].append("Try 8-bit or 4-bit quantization")
        compatibility["recommendations"].append("Use CPU offloading")
        compatibility["recommendations"].append("Consider a smaller model variant")
    
    return compatibility


if __name__ == "__main__":
    recommend_models_for_system()
