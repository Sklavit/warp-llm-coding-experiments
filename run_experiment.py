#!/usr/bin/env python3
"""
Main runner script for LLM experiments.

Usage:
    python run_experiment.py 1           # Run experiment 1
    python run_experiment.py --help      # Show help
"""

import sys
import subprocess
from pathlib import Path
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Available experiments
EXPERIMENTS = {
    "1": {
        "name": "Basic Text Generation",
        "script": "experiments/experiment_01_basic_generation.py",
        "description": "Simple text generation with input processing using various LLM models."
    }
}


def list_experiments():
    """Display available experiments."""
    table = Table(title="Available LLM Experiments")
    table.add_column("ID", style="cyan", width=4)
    table.add_column("Name", style="magenta")
    table.add_column("Description", style="green")
    
    for exp_id, exp_info in EXPERIMENTS.items():
        table.add_row(exp_id, exp_info["name"], exp_info["description"])
    
    console.print(table)


def run_experiment(experiment_id: str, args: list = None):
    """Run a specific experiment.
    
    Args:
        experiment_id: ID of the experiment to run
        args: Additional arguments to pass to the experiment
    """
    if experiment_id not in EXPERIMENTS:
        console.print(f"[red]Experiment {experiment_id} not found![/red]")
        list_experiments()
        return False
    
    exp_info = EXPERIMENTS[experiment_id]
    script_path = Path(exp_info["script"])
    
    if not script_path.exists():
        console.print(f"[red]Script {script_path} not found![/red]")
        return False
    
    console.print(Panel(
        f"[bold blue]Starting: {exp_info['name']}[/bold blue]\n\n"
        f"{exp_info['description']}",
        title=f"Experiment {experiment_id}",
        expand=False
    ))
    
    # Prepare command
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    try:
        # Run the experiment
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        console.print(f"[red]Error running experiment: {e}[/red]")
        return False


@click.command()
@click.argument('experiment_id', required=False)
@click.option('--list', '-l', 'list_exp', is_flag=True, help='List available experiments')
@click.option('--model', '-m', help='Model to use (for applicable experiments)')
@click.option('--max-length', help='Maximum generation length (for applicable experiments)')
def main(experiment_id, list_exp, model, max_length):
    """Run LLM experiments."""
    
    if list_exp or experiment_id is None:
        list_experiments()
        if experiment_id is None:
            return
    
    # Prepare additional arguments
    args = []
    if model:
        args.extend(['--model', model])
    if max_length:
        args.extend(['--max-length', max_length])
    
    success = run_experiment(experiment_id, args)
    if success:
        console.print("\n[green]✓ Experiment completed successfully![/green]")
    else:
        console.print("\n[red]✗ Experiment failed or was interrupted.[/red]")


if __name__ == "__main__":
    main()
