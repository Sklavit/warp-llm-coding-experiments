#!/usr/bin/env python3
"""
Experiment 1: Basic LLM text generation with input processing.

A simple CLI app that:
1. Takes user input
2. Processes it with an LLM
3. Prints the results
4. Stops
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from rich.text import Text
import click
import time
from transformers import pipeline, TextIteratorStreamer
from threading import Thread
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

from utils.model_modifiers import (
    prepare_model_and_tokenizer,
    get_model_info,
    count_parameters
)

# Initialize rich console for beautiful output
console = Console()


class UnifiedGenerator:
    """Wrapper class to handle both transformers and llama-cpp models uniformly."""
    
    def __init__(self, generator, model_type="transformers"):
        self.generator = generator
        self.model_type = model_type
        
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 0.7) -> str:
        """Generate text using the appropriate method based on model type."""
        if self.model_type == "llama-cpp":
            # Use llama-cpp generation
            max_tokens = max_length - len(prompt.split())
            response = self.generator(
                prompt,
                max_tokens=max(max_tokens, 10),
                temperature=temperature,
                echo=False,  # Don't include the prompt in output
                stop=["\n\n", "<|endoftext|>", "</s>"]  # Stop tokens
            )
            generated_text = response['choices'][0]['text']
            # Clean up the response
            generated_text = generated_text.strip()
            return generated_text
        else:
            # Use transformers generation
            outputs = self.generator(
                prompt,
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id if hasattr(self.generator, 'tokenizer') else None
            )
            full_text = outputs[0]['generated_text']
            # Remove the original prompt from the response
            if full_text.startswith(prompt):
                generated_text = full_text[len(prompt):].strip()
            else:
                generated_text = full_text.strip()
            return generated_text
    
    def generate_streaming(self, prompt: str, max_length: int = 100, temperature: float = 0.7):
        """Generate text with streaming for real-time display."""
        if self.model_type == "llama-cpp":
            # Stream from llama-cpp
            max_tokens = max_length - len(prompt.split())
            for token in self.generator(
                prompt,
                max_tokens=max(max_tokens, 10),
                temperature=temperature,
                echo=False,
                stop=["\n\n", "<|endoftext|>", "</s>"],
                stream=True
            ):
                chunk = token['choices'][0]['text']
                if chunk:
                    yield chunk
        else:
            # Stream from transformers using TextIteratorStreamer
            try:
                # Create streamer
                streamer = TextIteratorStreamer(
                    self.generator.tokenizer,
                    timeout=30,
                    skip_prompt=True,
                    skip_special_tokens=True
                )
                
                # Prepare generation arguments
                generation_kwargs = {
                    "input_ids": self.generator.tokenizer.encode(prompt, return_tensors="pt"),
                    "max_length": max_length,
                    "temperature": temperature,
                    "do_sample": True,
                    "streamer": streamer,
                    "pad_token_id": self.generator.tokenizer.eos_token_id
                }
                
                # Move to correct device
                if hasattr(self.generator.model, 'device'):
                    generation_kwargs["input_ids"] = generation_kwargs["input_ids"].to(self.generator.model.device)
                
                # Start generation in a thread
                thread = Thread(target=self.generator.model.generate, kwargs=generation_kwargs)
                thread.start()
                
                # Stream tokens
                for token in streamer:
                    yield token
                    
                thread.join()
                
            except Exception as e:
                # Fallback to non-streaming
                console.print(f"[yellow]⚠️  Streaming failed, using regular generation: {e}[/yellow]")
                result = self.generate(prompt, max_length, temperature)
                yield result
    
    @property
    def model(self):
        """Get the underlying model."""
        if self.model_type == "llama-cpp":
            return self.generator
        else:
            return self.generator.model
            
    @property
    def tokenizer(self):
        """Get the tokenizer (if available)."""
        if self.model_type == "llama-cpp":
            return None  # GGUF models don't expose tokenizer
        else:
            return self.generator.tokenizer


def create_text_generator(model_name: str = "microsoft/DialoGPT-small"):
    """Create a text generation pipeline.
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        UnifiedGenerator instance
    """
    console.print(f"[blue]Loading model: {model_name}[/blue]")
    
    # Check if this is a GGUF model
    if ":" in model_name and "GGUF" in model_name.upper():
        return create_gguf_generator(model_name)
    
    try:
        # Determine if we should use quantization for larger models
        use_quantization = False
        if "ministral" in model_name.lower() or "3b" in model_name.lower():
            use_quantization = True
            console.print("[yellow]⚡ Using 8-bit quantization for memory efficiency[/yellow]")
        
        # Configure model loading parameters
        pipeline_kwargs = {
            "model": model_name,
            "tokenizer": model_name,
            "torch_dtype": torch.float16 if torch.backends.mps.is_available() else torch.float32,
        }
        
        # Add quantization for larger models
        if use_quantization:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
            )
            pipeline_kwargs["model_kwargs"] = {"quantization_config": quantization_config}
            console.print("[yellow]  → 8-bit quantization enabled[/yellow]")
        
        # Use MPS device if available (Apple Silicon)
        if torch.backends.mps.is_available() and not use_quantization:
            pipeline_kwargs["device"] = "mps"
            console.print("[yellow]  → Using MPS (Apple Silicon) acceleration[/yellow]")
        
        # Create pipeline
        generator = pipeline("text-generation", **pipeline_kwargs)
        console.print("[green]✓ Model loaded successfully![/green]")
        return UnifiedGenerator(generator, "transformers")
        
    except ImportError as e:
        if "bitsandbytes" in str(e):
            console.print("[yellow]⚠️  bitsandbytes not available, trying without quantization...[/yellow]")
            # Fallback without quantization
            try:
                generator = pipeline(
                    "text-generation",
                    model=model_name,
                    tokenizer=model_name,
                    torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
                    device="mps" if torch.backends.mps.is_available() else None
                )
                console.print("[green]✓ Model loaded successfully (without quantization)![/green]")
                return UnifiedGenerator(generator, "transformers")
            except Exception as fallback_e:
                console.print(f"[red]✗ Failed to load model even without quantization: {fallback_e}[/red]")
                return None
    except Exception as e:
        console.print(f"[red]✗ Failed to load model: {e}[/red]")
        return None


def create_gguf_generator(model_spec: str):
    """Create a GGUF model generator using llama-cpp-python.
    
    Args:
        model_spec: Model specification in format "repo/model:quantization"
        
    Returns:
        UnifiedGenerator instance for GGUF model
    """
    if not LLAMA_CPP_AVAILABLE:
        console.print("[red]✗ llama-cpp-python is required for GGUF models![/red]")
        console.print("[yellow]Install with: pip install llama-cpp-python[/yellow]")
        return None
    
    try:
        # Parse model specification
        repo_path, quantization = model_spec.split(":")
        console.print(f"[yellow]  → Loading GGUF model: {repo_path}[/yellow]")
        console.print(f"[yellow]  → Quantization: {quantization}[/yellow]")
        
        # Download and load the specific GGUF file
        from huggingface_hub import hf_hub_download
        
        # Determine filename based on model type
        if "TinyLlama" in repo_path:
            filename = f"tinyllama-1.1b-chat-v1.0.{quantization}.gguf"
        elif "Ministral" in repo_path:
            filename = f"Ministral-3b-instruct.{quantization}.gguf"
        else:
            # Generic pattern
            model_name = repo_path.split("/")[-1].lower()
            filename = f"{model_name}.{quantization}.gguf"
        
        console.print(f"[blue]  → Downloading {filename}...[/blue]")
        
        model_path = hf_hub_download(
            repo_id=repo_path,
            filename=filename,
            cache_dir="./models_cache"
        )
        
        console.print(f"[blue]  → Loading from {model_path}[/blue]")
        
        # Load with llama-cpp with optimized settings
        llm = Llama(
            model_path=model_path,
            n_ctx=1024,  # Smaller context for faster generation
            n_threads=4,  # Fewer threads for stability
            verbose=False,
            n_batch=128,  # Smaller batch size
            use_mlock=False,  # Don't lock memory for macOS
            use_mmap=True,   # Use memory mapping
        )
        
        console.print("[green]✓ GGUF Model loaded successfully![/green]")
        return UnifiedGenerator(llm, "llama-cpp")
        
    except Exception as e:
        console.print(f"[red]✗ Failed to load GGUF model: {e}[/red]")
        console.print(f"[yellow]Debug: {str(e)}[/yellow]")
        return None


def process_input(generator, user_input: str, max_length: int = 100) -> str:
    """Process user input with the LLM.
    
    Args:
        generator: The UnifiedGenerator instance
        user_input: User's input text
        max_length: Maximum length of generated text
        
    Returns:
        Generated text response
    """
    try:
        # Use the unified generate method
        return generator.generate(user_input, max_length=max_length, temperature=0.7)
    except Exception as e:
        return f"Error processing input: {e}"


def display_model_info(generator):
    """Display information about the loaded model."""
    try:
        if generator.model_type == "llama-cpp":
            # GGUF model info
            info = {
                "Model Type": "GGUF (llama-cpp-python)",
                "Backend": "llama.cpp",
                "Quantization": "4-bit (estimated)",
                "Context Length": "2048 tokens",
                "Device": "CPU"
            }
        else:
            # Transformers model info
            model = generator.model
            tokenizer = generator.tokenizer
            
            if tokenizer and hasattr(model, 'parameters'):
                try:
                    total_params, trainable_params = count_parameters(model)
                    vocab_size = len(tokenizer) if tokenizer else "Unknown"
                    model_name = model.name_or_path if hasattr(model, 'name_or_path') else "Unknown"
                    device = str(next(model.parameters()).device) if hasattr(model, 'parameters') else "Unknown"
                    dtype = str(next(model.parameters()).dtype) if hasattr(model, 'parameters') else "Unknown"
                except:
                    total_params = trainable_params = vocab_size = device = dtype = "Unknown"
                    model_name = "Unknown"
            else:
                total_params = trainable_params = vocab_size = device = dtype = model_name = "Unknown"
            
            info = {
                "Model": model_name,
                "Vocabulary Size": vocab_size,
                "Total Parameters": f"{total_params:,}" if isinstance(total_params, int) else total_params,
                "Trainable Parameters": f"{trainable_params:,}" if isinstance(trainable_params, int) else trainable_params,
                "Device": device,
                "Data Type": dtype
            }
        
        console.print("\n[yellow]Model Information:[/yellow]")
        for key, value in info.items():
            console.print(f"  {key}: {value}")
        
    except Exception as e:
        console.print(f"[red]Could not get model info: {e}[/red]")


def main():
    """Main experiment function."""
    console.print(Panel(
        "[bold blue]🤖 LLM Experiment 1: Basic Text Generation[/bold blue]\n\n"
        "This experiment demonstrates basic text generation with an LLM.\n"
        "Enter your text, and the model will generate a continuation.",
        title="Experiment 1",
        expand=False
    ))
    
    # Model selection
    model_options = [
        "microsoft/DialoGPT-small",    # 117M params (~0.5GB) - Safe choice
        "gpt2",                        # 124M params (~0.5GB) - Classic
        "distilgpt2",                  # 82M params (~0.3GB) - Lightweight
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B params (~2.8GB) - Popular small LLM
        "meta-llama/Llama-3.2-1B",     # 1B params (~2.6GB) - Latest Llama
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:Q4_0",   # 1.1B GGUF 4-bit (~608MB) - Small test
        "ministral/Ministral-3b-instruct",     # 3B params (~7.8GB) - Will use quantization
        "QuantFactory/Ministral-3b-instruct-GGUF:Q4_K_M"  # 3B params 4-bit (~1.86GB) - GGUF quantized
    ]
    
    console.print("\n[yellow]Available models:[/yellow]")
    for i, model in enumerate(model_options, 1):
        console.print(f"  {i}. {model}")
    
    model_choice = Prompt.ask(
        "Choose a model (1-8) or enter custom model name",
        default="1"
    )
    
    try:
        model_idx = int(model_choice) - 1
        if 0 <= model_idx < len(model_options):
            selected_model = model_options[model_idx]
        else:
            raise ValueError("Invalid choice")
    except ValueError:
        # Treat as custom model name
        selected_model = model_choice
    
    console.print(f"\n[blue]Selected model: {selected_model}[/blue]")
    
    # Load model
    generator = create_text_generator(selected_model)
    if generator is None:
        console.print("[red]Failed to load model. Exiting.[/red]")
        return
    
    # Display model information
    display_model_info(generator)
    
    # Get max length parameter
    max_length = Prompt.ask(
        "\nMaximum generation length",
        default="50"
    )
    try:
        max_length = int(max_length)
    except ValueError:
        max_length = 50
    
    # Main interaction loop
    console.print("\n[green]Ready for input! Type 'quit' to exit.[/green]")
    
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("\n[cyan]Your input")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input.strip():
                console.print("[yellow]Please enter some text.[/yellow]")
                continue
            
            # Display user input clearly
            console.print(Panel(
                user_input,
                title="[cyan]👤 Your Input[/cyan]",
                expand=False,
                border_style="cyan"
            ))
            
            # Stream the response with real-time display
            console.print("\n[green]🤖 Model Response:[/green]")
            
            response_text = Text()
            response_panel = Panel(
                response_text,
                title="[green]🤖 Generating...[/green]",
                border_style="green"
            )
            
            try:
                with Live(response_panel, refresh_per_second=10) as live:
                    token_count = 0
                    start_time = time.time()
                    
                    for token in generator.generate_streaming(user_input, max_length, temperature=0.7):
                        if token:
                            response_text.append(token)
                            token_count += 1
                            
                            # Update progress info
                            elapsed = time.time() - start_time
                            tokens_per_sec = token_count / elapsed if elapsed > 0 else 0
                            
                            # Update panel title with progress
                            live.update(Panel(
                                response_text,
                                title=f"[green]🤖 Generating... ({token_count} tokens, {tokens_per_sec:.1f} tok/s)[/green]",
                                border_style="green"
                            ))
                    
                    # Final update
                    elapsed = time.time() - start_time
                    tokens_per_sec = token_count / elapsed if elapsed > 0 else 0
                    live.update(Panel(
                        response_text,
                        title=f"[green]✓ Complete ({token_count} tokens, {tokens_per_sec:.1f} tok/s, {elapsed:.1f}s)[/green]",
                        border_style="green"
                    ))
                    
            except Exception as e:
                console.print(f"[red]Streaming error: {e}[/red]")
                console.print("[yellow]Falling back to regular generation...[/yellow]")
                
                with console.status("[bold green]🤖 Generating response..."):
                    result = process_input(generator, user_input, max_length)
                
                if result:
                    console.print(Panel(
                        result,
                        title="[green]🤖 Model Response[/green]",
                        expand=False,
                        border_style="green"
                    ))
                else:
                    console.print(Panel(
                        "[dim]No response generated[/dim]",
                        title="[yellow]⚠️  Empty Response[/yellow]",
                        expand=False,
                        border_style="yellow"
                    ))
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    console.print("\n[blue]Experiment complete! 👋[/blue]")


@click.command()
@click.option('--model', '-m', default='microsoft/DialoGPT-small', 
              help='Model name to use for generation')
@click.option('--max-length', '-l', default=50, 
              help='Maximum length of generated text')
def cli_main(model: str, max_length: int):
    """CLI interface for the experiment."""
    console.print(Panel(
        f"[bold blue]🤖 LLM Experiment 1: Basic Text Generation[/bold blue]\n\n"
        f"Model: {model}\n"
        f"Max Length: {max_length}",
        title="Experiment 1 - CLI Mode",
        expand=False
    ))
    
    generator = create_text_generator(model)
    if generator is None:
        return
    
    display_model_info(generator)
    
    # Single input mode for CLI
    user_input = Prompt.ask("\n[cyan]Enter your text")
    
    if user_input.strip():
        console.print(f"\n[blue]Processing: '{user_input}'[/blue]")
        
        with console.status("[bold green]Generating response..."):
            result = process_input(generator, user_input, max_length)
        
        console.print(Panel(
            result,
            title="[green]Generated Text[/green]",
            expand=False
        ))


if __name__ == "__main__":
    # Check if running with CLI arguments
    if len(sys.argv) > 1:
        cli_main()
    else:
        main()
