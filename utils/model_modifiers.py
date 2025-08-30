"""
Functional utilities for modifying LLM models, tokenizers, and layers.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from typing import Dict, List, Optional, Any, Tuple
import copy


# Tokenizer modification functions
def add_special_tokens(tokenizer, special_tokens: Dict[str, str]) -> int:
    """Add special tokens to the tokenizer.
    
    Args:
        tokenizer: The tokenizer to modify
        special_tokens: Dictionary of special tokens to add
        
    Returns:
        Number of tokens added
    """
    return tokenizer.add_special_tokens(special_tokens)


def add_tokens_to_vocab(tokenizer, new_tokens: List[str]) -> int:
    """Add new tokens to the vocabulary.
    
    Args:
        tokenizer: The tokenizer to modify
        new_tokens: List of new tokens to add
        
    Returns:
        Number of tokens added
    """
    return tokenizer.add_tokens(new_tokens)


def get_vocab_size(tokenizer) -> int:
    """Get the current vocabulary size."""
    return len(tokenizer)


def ensure_padding_token(tokenizer):
    """Ensure the tokenizer has a padding token."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# Layer modification functions
def freeze_layers(model, layer_names: List[str]) -> None:
    """Freeze specified layers.
    
    Args:
        model: The model to modify
        layer_names: List of layer name patterns to freeze
    """
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = False


def unfreeze_layers(model, layer_names: List[str]) -> None:
    """Unfreeze specified layers.
    
    Args:
        model: The model to modify
        layer_names: List of layer name patterns to unfreeze
    """
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = True


def replace_layer(model, layer_path: str, new_layer: nn.Module) -> None:
    """Replace a layer at the specified path.
    
    Args:
        model: The model to modify
        layer_path: Dot-separated path to the layer (e.g., 'transformer.h.0.attn')
        new_layer: New layer to replace with
    """
    def set_nested_attr(obj, path, value):
        attrs = path.split('.')
        for attr in attrs[:-1]:
            obj = getattr(obj, attr)
        setattr(obj, attrs[-1], value)
        
    set_nested_attr(model, layer_path, new_layer)


def get_layer_by_path(model, layer_path: str):
    """Get a layer by its path.
    
    Args:
        model: The model to query
        layer_path: Dot-separated path to the layer
        
    Returns:
        The layer at the specified path
    """
    attrs = layer_path.split('.')
    obj = model
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def get_layer_info(model) -> Dict[str, Any]:
    """Get information about model layers.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary with layer information
    """
    layer_info = {}
    for name, module in model.named_modules():
        layer_info[name] = {
            'type': type(module).__name__,
            'parameters': sum(p.numel() for p in module.parameters()),
            'trainable_parameters': sum(p.numel() for p in module.parameters() if p.requires_grad)
        }
    return layer_info


def count_parameters(model) -> Tuple[int, int]:
    """Count total and trainable parameters in a model.
    
    Args:
        model: The model to analyze
        
    Returns:
        Tuple of (total_parameters, trainable_parameters)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# Model loading and setup functions
def load_tokenizer(model_name_or_path: str):
    """Load and prepare a tokenizer.
    
    Args:
        model_name_or_path: Model name or path
        
    Returns:
        Prepared tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return ensure_padding_token(tokenizer)


def load_model_for_generation(model_name_or_path: str, use_gpu: bool = None):
    """Load model for text generation.
    
    Args:
        model_name_or_path: Model name or path
        use_gpu: Whether to use GPU (auto-detect if None)
        
    Returns:
        Loaded model
    """
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
        
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if use_gpu else torch.float32,
        device_map="auto" if use_gpu else None
    )


def load_model_for_encoding(model_name_or_path: str):
    """Load model for encoding/embeddings.
    
    Args:
        model_name_or_path: Model name or path
        
    Returns:
        Loaded model
    """
    return AutoModel.from_pretrained(model_name_or_path)


def resize_token_embeddings(model, tokenizer):
    """Resize token embeddings after tokenizer modifications.
    
    Args:
        model: The model to modify
        tokenizer: The tokenizer with updated vocabulary
    """
    model.resize_token_embeddings(len(tokenizer))
    return model


def get_model_info(model, tokenizer, model_name: str = "Unknown") -> Dict[str, Any]:
    """Get comprehensive model information.
    
    Args:
        model: The model to analyze
        tokenizer: The tokenizer
        model_name: Name of the model
        
    Returns:
        Dictionary with model information
    """
    total_params, trainable_params = count_parameters(model)
    
    info = {
        'model_name': model_name,
        'vocab_size': len(tokenizer),
        'special_tokens': tokenizer.special_tokens_map,
        'num_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_dtype': next(model.parameters()).dtype,
        'device': next(model.parameters()).device,
    }
    
    return info


# Utility composition functions
def prepare_model_and_tokenizer(model_name_or_path: str, for_generation: bool = True):
    """Prepare both model and tokenizer together.
    
    Args:
        model_name_or_path: Model name or path
        for_generation: Whether to load for generation (True) or encoding (False)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = load_tokenizer(model_name_or_path)
    
    if for_generation:
        model = load_model_for_generation(model_name_or_path)
    else:
        model = load_model_for_encoding(model_name_or_path)
    
    return model, tokenizer


def modify_tokenizer_and_resize_embeddings(model, tokenizer, new_tokens: List[str] = None, special_tokens: Dict[str, str] = None):
    """Modify tokenizer and resize model embeddings accordingly.
    
    Args:
        model: The model to modify
        tokenizer: The tokenizer to modify
        new_tokens: Optional list of new tokens to add
        special_tokens: Optional dict of special tokens to add
        
    Returns:
        Tuple of (modified_model, modified_tokenizer, tokens_added)
    """
    tokens_added = 0
    
    if special_tokens:
        tokens_added += add_special_tokens(tokenizer, special_tokens)
    
    if new_tokens:
        tokens_added += add_tokens_to_vocab(tokenizer, new_tokens)
    
    if tokens_added > 0:
        model = resize_token_embeddings(model, tokenizer)
    
    return model, tokenizer, tokens_added
