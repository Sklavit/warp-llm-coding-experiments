# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LLM experimentation framework for testing Large Language Models with a functional programming approach. The project supports both HuggingFace Transformers models and GGUF quantized models via llama-cpp-python.

## Development Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Running Experiments
```bash
# List available experiments
python run_experiment.py --list

# Run experiment 1 (basic text generation)
python run_experiment.py 1

# Run with specific model
python run_experiment.py 1 --model gpt2

# Direct execution of experiment
python experiments/experiment_01_basic_generation.py

# CLI mode with parameters
python experiments/experiment_01_basic_generation.py --model distilgpt2 --max-length 100
```

### Model Research & Analysis
```bash
# Fetch live model information from HuggingFace API
python utils/fetch_model_info.py

# Analyze system compatibility and get model recommendations  
python utils/model_size_estimator.py

# Check specific model compatibility
python -c "from utils.model_size_estimator import check_model_compatibility; print(check_model_compatibility('gpt2'))"
```

## Architecture Overview

### Core Components

- **`run_experiment.py`**: Main experiment runner with Rich CLI interface
- **`experiments/`**: Individual experiment scripts with streaming capabilities (currently experiment_01_basic_generation.py)
- **`utils/model_modifiers.py`**: Functional utilities for model/tokenizer modifications
- **`utils/fetch_model_info.py`**: HuggingFace API integration for model research
- **`utils/model_size_estimator.py`**: System compatibility checker and model recommendations

### Streaming Features

The framework implements advanced real-time streaming:
- **Token-by-token generation**: See text appear as the model generates it
- **Live performance metrics**: Real-time tokens/second and progress tracking
- **Visual feedback**: Live updating Rich panels with progress indicators
- **Cross-platform compatibility**: Native streaming for both Transformers and GGUF models
- **Graceful fallbacks**: Automatic retry with non-streaming if streaming fails

### Key Architecture Patterns

1. **Functional Model Modification**: All model modifications use pure functions that return modified copies
2. **Unified Model Interface**: `UnifiedGenerator` class abstracts differences between HuggingFace and GGUF models
3. **Real-Time Streaming**: Token-by-token generation with live progress tracking and performance metrics
4. **Rich Terminal UI**: All user interactions use Rich library with live updating panels and color-coded borders
5. **Automatic Hardware Detection**: Supports MPS (Apple Silicon), CUDA, and CPU with automatic device selection
6. **Robust Error Handling**: Graceful fallback from streaming to regular generation if needed

### Model Support

The framework supports:
- **HuggingFace Transformers**: Any text generation model with automatic quantization
- **GGUF Models**: Via llama-cpp-python for memory-efficient inference
- **Quantization**: 8-bit quantization for larger models, 4-bit GGUF support

### Memory Management

- Automatic 8-bit quantization for models >1B parameters
- 4-bit GGUF quantization for maximum memory efficiency
- MPS acceleration on Apple Silicon with unified memory
- Models cached in `./models_cache/` with HuggingFace Hub integration
- System compatibility checking with precise memory estimation
- Smart device selection (MPS > CUDA > CPU)

### Research & Analysis Tools

- **Live HuggingFace API Integration**: Real-time model information, downloads, likes
- **ELO Score Database**: Chatbot Arena performance rankings for model comparison
- **System Compatibility Checker**: Hardware analysis with memory recommendations
- **Model Search**: Pattern-based model discovery with popularity metrics
- **Memory Estimation**: Precise calculations for inference and training memory usage

## Development Guidelines

1. **Follow Functional Patterns**: Use the utilities in `utils/model_modifiers.py` for all model modifications
2. **Use Rich for UI**: All terminal output should use Rich panels, tables, and progress bars with live updates
3. **Implement Streaming**: New experiments should support real-time streaming with performance metrics
4. **Handle Both Model Types**: When adding new features, support both HuggingFace and GGUF models via `UnifiedGenerator`
5. **Add to Experiment Runner**: New experiments should be registered in `EXPERIMENTS` dict in `run_experiment.py`
6. **Memory Awareness**: Use system analysis tools to check compatibility before loading models
7. **Error Handling**: Implement graceful fallbacks for streaming and other operations
8. **Visual Separation**: Use color-coded panels to separate user input (cyan) from model output (green)
9. **Research Integration**: Use the model research tools to validate model choices
10. **Caching Strategy**: Leverage the models_cache directory for efficient model reuse

## Model Loading Patterns

### HuggingFace Models
```python
from utils.model_modifiers import prepare_model_and_tokenizer

model, tokenizer = prepare_model_and_tokenizer("microsoft/DialoGPT-small")
```

### GGUF Models
```python
# Format: "repo/model:quantization"
generator = create_gguf_generator("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:Q4_0")
```

### Unified Interface
```python
generator = UnifiedGenerator(pipeline_or_llm, model_type)
text = generator.generate(prompt, max_length=100)

# Real-time streaming with Rich Live display
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

response_text = Text()
with Live(Panel(response_text, title="🤖 Generating..."), refresh_per_second=10) as live:
    for token in generator.generate_streaming(prompt, max_length=100):
        response_text.append(token)
        live.update(Panel(response_text, title=f"🤖 Generating... ({len(response_text)} tokens)"))
```