# 🤖 Warp LLM Coding Experiments

Testing how good is warp coding on the task of conducting several experiments with LLMs.

*Note: After 150 prompts with Warp, I switched to Claude Code to continue development.*

This project provides a framework for experimenting with Large Language Models (LLMs), including capabilities to modify tokenizers, layers, and model behavior using a functional programming approach.

## 🚀 Features

### Core Capabilities
- **Functional Architecture**: Clean, functional approach to model manipulation
- **Real-Time Streaming Generation**: Token-by-token streaming with live progress tracking
- **Model Modification Tools**: Utilities for tokenizer and layer modifications
- **Multiple Experiments**: Structured experiments for different LLM capabilities
- **Rich CLI Interface**: Beautiful terminal interface with live updating panels
- **Performance Metrics**: Real-time tokens/second and generation time tracking

### Model Support & Optimization
- **Dual Backend Support**: Works with both HuggingFace Transformers and GGUF models
- **Automatic Hardware Detection**: MPS (Apple Silicon), CUDA, and CPU support
- **Smart Quantization**: Automatic 8-bit quantization for models >1B parameters
- **GGUF Integration**: Memory-efficient 4-bit quantized models via llama-cpp-python
- **Model Caching**: Intelligent caching system in `./models_cache/`
- **8 Predefined Models**: From 82M (DistilGPT2) to 3B (Ministral) parameters

### Research & Analysis Tools
- **Live Model Research**: HuggingFace API integration with real-time model information
- **System Compatibility Analysis**: Hardware analysis and model recommendations
- **ELO Score Database**: Performance rankings from Chatbot Arena leaderboard
- **Memory Estimation**: Precise memory usage calculations for model selection

## 📁 Project Structure

```
warp-llm-coding-experiments/
├── experiments/           # Individual experiment scripts
│   ├── __init__.py
│   └── experiment_01_basic_generation.py  # Streaming text generation
├── utils/                # Utility functions and research tools
│   ├── __init__.py
│   ├── model_modifiers.py     # Functional model modification tools
│   ├── fetch_model_info.py    # HuggingFace API integration
│   └── model_size_estimator.py # System compatibility checker
├── models/               # Custom model implementations
│   └── __init__.py
├── models_cache/         # Downloaded model cache directory
├── run_experiment.py     # Main experiment runner with Rich CLI
├── pyproject.toml        # Python dependencies (uv)
├── package.json          # Node.js dependencies (tavily-mcp)
├── uv.lock              # Python dependency lock file
├── model_research_results.json # Cached model research data
├── CLAUDE.md            # Claude Code development guidance
└── README.md            # This file
```

## 🛠️ Setup

1. **Clone the repository** (already done)

2. **Install dependencies using uv**:
   ```bash
   uv sync
   ```

3. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

## 🎯 Usage

### List Available Experiments
```bash
python run_experiment.py --list
```

### Run Experiment 1 (Basic Text Generation)
```bash
# Interactive mode with model selection
python run_experiment.py 1

# With specific model
python run_experiment.py 1 --model gpt2

# Direct execution (interactive mode)
python experiments/experiment_01_basic_generation.py

# CLI mode with parameters
python experiments/experiment_01_basic_generation.py --model distilgpt2 --max-length 100
```

### Model Research & System Analysis
```bash
# Fetch live model information from HuggingFace API
python utils/fetch_model_info.py

# Analyze system compatibility and get model recommendations
python utils/model_size_estimator.py
```

## 🧪 Available Experiments

### Experiment 1: Basic Text Generation
- **Purpose**: Demonstrate real-time LLM text generation with advanced UI
- **Streaming Features**:
  - Token-by-token real-time generation display
  - Live performance metrics (tokens/second, elapsed time)
  - Progress tracking with token count
  - Visual feedback with live updating Rich panels
  - Graceful fallback to non-streaming if needed
- **Model Support**:
  - 8 predefined models from 82M to 3B parameters
  - Any custom HuggingFace text generation model
  - GGUF quantized models (4-bit) via llama-cpp-python
  - Automatic quantization for large models
- **User Experience**:
  - Interactive model selection with descriptions
  - Color-coded panels (cyan for input, green for output)
  - Configurable generation parameters
  - Detailed model information display
  - Professional terminal interface

## 🔧 Utility Modules

### Model Modification (`utils/model_modifiers.py`)
Functional utilities for model manipulation:

### Tokenizer Modifications
- `add_special_tokens()` - Add special tokens
- `add_tokens_to_vocab()` - Add new vocabulary tokens
- `ensure_padding_token()` - Ensure padding token exists

### Layer Modifications
- `freeze_layers()` / `unfreeze_layers()` - Control layer training
- `replace_layer()` - Replace specific model layers
- `get_layer_info()` - Analyze layer structure

### Model Management
- `prepare_model_and_tokenizer()` - Load model and tokenizer together
- `modify_tokenizer_and_resize_embeddings()` - Comprehensive tokenizer modification
- `get_model_info()` - Get detailed model information
- `count_parameters()` - Count total and trainable parameters

### Model Research (`utils/fetch_model_info.py`)
Live model research and analysis:
- **HuggingFace API Integration**: Fetch real-time model information
- **Model Search**: Search models by pattern with download/like metrics
- **ELO Score Database**: Chatbot Arena performance rankings
- **Research Results**: Cached JSON output for reuse
- **Batch Processing**: Research multiple models efficiently

### System Analysis (`utils/model_size_estimator.py`)
Hardware compatibility and recommendations:
- **System Information**: RAM, disk, GPU/MPS detection
- **Memory Estimation**: Precise memory usage calculations for inference/training
- **Model Recommendations**: Curated list of 25+ models with compatibility status
- **Performance Insights**: Download counts, ELO scores, creation dates
- **Compatibility Checker**: Determine if specific models will run on your system

## 📋 Dependencies

### Core ML Libraries
- `transformers` - HuggingFace Transformers library
- `torch` - PyTorch deep learning framework  
- `tokenizers` - Fast tokenizers
- `datasets` - Dataset loading utilities
- `accelerate` - Model acceleration
- `llama-cpp-python` - GGUF model support
- `huggingface-hub` - Model downloading and caching

### UI & CLI
- `rich` - Beautiful terminal output with live updates
- `click` - Command-line interface creation
- `psutil` - System information and monitoring

### Optional
- `bitsandbytes` - 8-bit quantization (Linux/Windows)
- `safetensors` - Safe tensor format
- `tavily-mcp` - Web search integration (Node.js dependency)

## 🤝 Contributing

This is an experimental project for testing LLM coding capabilities. Feel free to:
- Add new experiments
- Improve existing utilities
- Suggest model modifications
- Report issues

## 📝 Development Notes

- **Functional Programming**: Pure functions for all model modifications ensure safety and composability
- **Self-Contained Experiments**: Each experiment can run independently with its own CLI
- **Hardware Optimization**: Automatic device detection (MPS/CUDA/CPU) with appropriate data types
- **Memory Management**: Smart quantization and caching to maximize model compatibility
- **Real-Time Feedback**: Streaming generation provides immediate visual feedback
- **Research Integration**: Live model research capabilities with performance benchmarks
- **Cross-Platform**: Works on Apple Silicon (MPS), NVIDIA GPUs (CUDA), and CPU-only systems
