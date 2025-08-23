#!/usr/bin/env python3
"""
Standalone script to generate comparison plots from existing LLC analysis results.

This script can:
1. Generate clean vs adversarial comparison plots from a single experiment directory
2. Generate defense method comparison plots from multiple experiment directories
3. Generate combined training metrics + LLC trajectory plots

Usage:
    # Generate clean vs adversarial plot from a single experiment
    python generate_comparison_plots.py --mode clean_vs_adv --experiment_dir ./llc_analysis/llc_analysis_20250819_094945/ResNet18_Standard_clean_vs_adv

    # Generate defense comparison plot from multiple experiments
    python generate_comparison_plots.py --mode defense_comparison --experiment_dirs ./llc_analysis/llc_analysis_20250819_094945/ResNet18_Standard_clean_vs_adv ./llc_analysis/llc_analysis_20250819_095046/ResNet18_AT_clean_vs_adv --dataset CIFAR10

    # Auto-discover all clean_vs_adv experiments in a results directory
    python generate_comparison_plots.py --mode defense_comparison --results_dir ./llc_analysis/llc_analysis_20250819_094945 --dataset CIFAR10
    
    # Generate combined training metrics + LLC plots
    python generate_comparison_plots.py --mode training_llc --model_dir ./models/LeNet_AT --experiment_dir ./llc_analysis/llc_analysis_20250819_094945/LeNet_AT_clean_vs_adv
    
    # Generate combined training + LLC + effective dimensionality plots
    python generate_comparison_plots.py --mode training_llc --model_dir ./models/LeNet_AT --experiment_dir ./llc_analysis/llc_analysis_20250819_094945/LeNet_AT_clean_vs_adv --eff_dim_path ./eff_dim_analysis/LeNet_AT_effective_dim.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class LLCPlotGenerator:
    """Generate comparison plots from existing LLC analysis results."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the plot generator.
        
        Args:
            output_dir: Directory to save plots. If None, saves to current directory.
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_experiment_data(self, experiment_dir: str) -> Dict:
        """
        Load experiment data from a directory.
        
        Args:
            experiment_dir: Path to experiment directory
            
        Returns:
            Dictionary containing experiment data
        """
        experiment_path = Path(experiment_dir)
        
        # First try to construct from individual trajectory files (more reliable)
        try:
            return self._construct_experiment_data(experiment_path)
        except Exception as e:
            print(f"Warning: Could not load from individual trajectory files: {e}")
        
        # Fall back to clean_vs_adv_comparison.json
        comparison_file = experiment_path / "clean_vs_adv_comparison.json"
        if comparison_file.exists():
            print("Loading from clean_vs_adv_comparison.json (may contain null values)")
            with open(comparison_file, 'r') as f:
                return json.load(f)
        
        raise ValueError(f"Could not find any valid data in {experiment_path}")
    
    def _construct_experiment_data(self, experiment_path: Path) -> Dict:
        """
        Construct experiment data from individual trajectory files.
        
        Args:
            experiment_path: Path to experiment directory
            
        Returns:
            Dictionary containing experiment data
        """
        clean_trajectory = None
        adversarial_trajectory = None
        
        # Load clean trajectory
        clean_dir = experiment_path / "clean_llc_results"
        if clean_dir.exists():
            clean_trajectory_file = clean_dir / "llc_trajectory.json"
            if clean_trajectory_file.exists():
                with open(clean_trajectory_file, 'r') as f:
                    clean_trajectory = json.load(f)
        
        # Load adversarial trajectory
        adv_dir = experiment_path / "adversarial_llc_results"
        if adv_dir.exists():
            adv_trajectory_file = adv_dir / "llc_trajectory.json"
            if adv_trajectory_file.exists():
                with open(adv_trajectory_file, 'r') as f:
                    adversarial_trajectory = json.load(f)
        
        if not clean_trajectory or not adversarial_trajectory:
            raise ValueError(f"Could not find trajectory data in {experiment_path}")
        
        # Extract metadata from directory name or trajectory data
        dir_name = experiment_path.name
        parts = dir_name.split('_')
        
        model_name = parts[0] if parts else "Unknown"
        defense_method = parts[1] if len(parts) > 1 else "Unknown"
        
        # Try to get more accurate info from trajectory data
        if 'model_name' in clean_trajectory:
            model_name = clean_trajectory['model_name']
        if 'defense_method' in clean_trajectory:
            defense_method = clean_trajectory['defense_method']
        
        return {
            'model_name': model_name,
            'dataset_name': clean_trajectory.get('dataset_name', 'Unknown'),
            'defense_method': defense_method,
            'clean_trajectory': clean_trajectory,
            'adversarial_trajectory': adversarial_trajectory,
            'checkpoint_names': clean_trajectory.get('checkpoint_names', [])
        }
    
    def generate_clean_vs_adv_plot(self, experiment_dir: str, output_name: Optional[str] = None) -> str:
        """
        Generate clean vs adversarial comparison plot.
        
        Args:
            experiment_dir: Path to experiment directory
            output_name: Custom output filename (without extension)
            
        Returns:
            Path to generated plot
        """
        print(f"Loading experiment data from: {experiment_dir}")
        results = self.load_experiment_data(experiment_dir)
        
        if output_name is None:
            output_name = f"{results['model_name']}_{results['defense_method']}_clean_vs_adversarial"
        
        # Save plot inside the experiment directory if no custom output_dir specified
        if self.output_dir == Path.cwd():
            output_path = Path(experiment_dir) / f"{output_name}.png"
        else:
            output_path = self.output_dir / f"{output_name}.png"
        
        print(f"Generating clean vs adversarial plot...")
        self._create_clean_vs_adv_visualization(results, output_path)
        
        print(f"✅ Plot saved to: {output_path}")
        return str(output_path)
    
    def generate_defense_comparison_plot(self, experiment_dirs: List[str], dataset_name: str, 
                                       output_name: Optional[str] = None) -> str:
        """
        Generate defense method comparison plot.
        
        Args:
            experiment_dirs: List of experiment directories
            dataset_name: Dataset name for the plot title
            output_name: Custom output filename (without extension)
            
        Returns:
            Path to generated plot
        """
        print(f"Loading data from {len(experiment_dirs)} experiments...")
        
        results = {}
        for exp_dir in experiment_dirs:
            try:
                exp_data = self.load_experiment_data(exp_dir)
                defense_method = exp_data['defense_method']
                results[defense_method] = {
                    'llc_results': exp_data,
                    'model_name': exp_data['model_name']
                }
                print(f"  ✅ Loaded {defense_method}")
            except Exception as e:
                print(f"  ⚠️ Failed to load {exp_dir}: {e}")
        
        if not results:
            raise ValueError("No valid experiment data found")
        
        if output_name is None:
            model_name = list(results.values())[0]['model_name']
            output_name = f"defense_methods_comparison_{model_name}_{dataset_name}"
        
        # Save plot in the first experiment directory unless custom output_dir is explicitly specified
        if self.output_dir == Path.cwd() and experiment_dirs:
            output_path = Path(experiment_dirs[0]).parent / f"{output_name}.png"
        else:
            output_path = self.output_dir / f"{output_name}.png"
        
        print(f"Generating defense comparison plot...")
        self._create_defense_comparison_plots(results, dataset_name, output_path)
        
        print(f"✅ Plot saved to: {output_path}")
        return str(output_path)
    
    def auto_discover_experiments(self, results_dir: str) -> List[str]:
        """
        Auto-discover clean_vs_adv experiments in a results directory.
        
        Args:
            results_dir: Path to results directory
            
        Returns:
            List of experiment directories
        """
        results_path = Path(results_dir)
        experiment_dirs = []
        
        for item in results_path.iterdir():
            if item.is_dir() and "clean_vs_adv" in item.name:
                experiment_dirs.append(str(item))
        
        return sorted(experiment_dirs)
    
    def _create_clean_vs_adv_visualization(self, results: Dict, output_path: Path):
        """Create visualization comparing clean vs adversarial LLC trajectories"""
        clean_means = results['clean_trajectory']['llc_means']
        clean_stds = results['clean_trajectory']['llc_stds']
        adv_means = results['adversarial_trajectory']['llc_means']
        adv_stds = results['adversarial_trajectory']['llc_stds']
        checkpoint_names = results['checkpoint_names']
        
        # Count valid data points
        clean_valid = sum(1 for x in clean_means if x is not None)
        adv_valid = sum(1 for x in adv_means if x is not None)
        
        # Filter out None values
        valid_indices = [i for i, (clean, adv) in enumerate(zip(clean_means, adv_means)) 
                        if clean is not None and adv is not None]
        
        if not valid_indices:
            raise ValueError(f"No valid LLC measurements to plot. Clean data points: {clean_valid}/{len(clean_means)}, Adversarial data points: {adv_valid}/{len(adv_means)}. Check if the adversarial LLC measurement failed during the original pipeline run.")
        
        valid_clean_means = [clean_means[i] for i in valid_indices]
        valid_clean_stds = [clean_stds[i] for i in valid_indices]
        valid_adv_means = [adv_means[i] for i in valid_indices]
        valid_adv_stds = [adv_stds[i] for i in valid_indices]
        valid_names = [checkpoint_names[i] for i in valid_indices]
        
        # Create single comparison plot (removed difference plot as requested)
        plt.figure(figsize=(12, 8))
        
        x_positions = range(len(valid_indices))
        
        # Main comparison plot
        plt.errorbar(x_positions, valid_clean_means, yerr=valid_clean_stds, 
                    marker='o', capsize=5, capthick=2, linewidth=2, markersize=8,
                    label='Clean Data', color='blue')
        plt.errorbar(x_positions, valid_adv_means, yerr=valid_adv_stds, 
                    marker='s', capsize=5, capthick=2, linewidth=2, markersize=8,
                    label='Adversarial Data', color='red')
        
        plt.xlabel("Checkpoint")
        plt.ylabel("Local Learning Coefficient (LLC)")
        plt.title(f"Clean vs Adversarial LLC - {results['model_name']} ({results['defense_method']})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_defense_comparison_plots(self, results: Dict, dataset_name: str, output_path: Path):
        """Create comparison plots for different defense methods"""
        # Filter out failed experiments
        valid_results = {k: v for k, v in results.items() 
                        if 'llc_results' in v and 'error' not in v}
        
        if not valid_results:
            raise ValueError("No valid results to plot")
        
        # Extract data for plotting
        defense_methods = list(valid_results.keys())
        
        # For clean vs adv results, we need to extract the right trajectory data
        clean_data = {}
        adv_data = {}
        
        for method, data in valid_results.items():
            llc_results = data['llc_results']
            
            if 'clean_trajectory' in llc_results and 'adversarial_trajectory' in llc_results:
                # This is clean_vs_adv data
                clean_means = llc_results['clean_trajectory']['llc_means']
                adv_means = llc_results['adversarial_trajectory']['llc_means']
                clean_stds = llc_results['clean_trajectory']['llc_stds']
                adv_stds = llc_results['adversarial_trajectory']['llc_stds']
            else:
                # This might be single trajectory data
                clean_means = llc_results.get('llc_means', [])
                adv_means = []
                clean_stds = llc_results.get('llc_stds', [])
                adv_stds = []
            
            # Calculate mean LLC across checkpoints (filtering None values)
            valid_clean = [x for x in clean_means if x is not None]
            valid_adv = [x for x in adv_means if x is not None]
            
            clean_data[method] = {
                'mean': np.mean(valid_clean) if valid_clean else 0,
                'std': np.std(valid_clean) if valid_clean else 0,
                'trajectory': valid_clean
            }
            
            adv_data[method] = {
                'mean': np.mean(valid_adv) if valid_adv else 0,
                'std': np.std(valid_adv) if valid_adv else 0,
                'trajectory': valid_adv
            }
        
        # Create comparison plots
        fig, (ax1, ax2, ax4) = plt.subplots(1, 3, figsize=(20, 6))
        
        # 1. Mean LLC comparison (Clean)
        methods = list(clean_data.keys())
        clean_means = [clean_data[m]['mean'] for m in methods]
        clean_errs = [clean_data[m]['std'] for m in methods]
        
        bars1 = ax1.bar(methods, clean_means, yerr=clean_errs, capsize=5, alpha=0.7, color='blue')
        ax1.set_title(f'Mean LLC - Clean Data ({dataset_name})')
        ax1.set_ylabel('LLC')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, mean in zip(bars1, clean_means):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.2f}', ha='center', va='bottom')
        
        # 2. Mean LLC comparison (Adversarial)
        if any(adv_data[m]['trajectory'] for m in methods):
            adv_means = [adv_data[m]['mean'] for m in methods]
            adv_errs = [adv_data[m]['std'] for m in methods]
            
            bars2 = ax2.bar(methods, adv_means, yerr=adv_errs, capsize=5, alpha=0.7, color='red')
            ax2.set_title(f'Mean LLC - Adversarial Data ({dataset_name})')
            ax2.set_ylabel('LLC')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, mean in zip(bars2, adv_means):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mean:.2f}', ha='center', va='bottom')
        else:
            ax2.text(0.5, 0.5, 'No Adversarial Data Available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title(f'Mean LLC - Adversarial Data ({dataset_name})')
        
        # 3. Trajectory comparison (if we have trajectory data)
        ax4.set_title(f'LLC Trajectories ({dataset_name})')
        ax4.set_xlabel('Checkpoint')
        ax4.set_ylabel('LLC')
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
        for i, method in enumerate(methods):
            if clean_data[method]['trajectory']:
                x_pos = range(len(clean_data[method]['trajectory']))
                ax4.plot(x_pos, clean_data[method]['trajectory'], 
                        marker='o', label=f'{method} (Clean)', color=colors[i], linestyle='-')
            
            if adv_data[method]['trajectory']:
                x_pos = range(len(adv_data[method]['trajectory']))
                ax4.plot(x_pos, adv_data[method]['trajectory'], 
                        marker='s', label=f'{method} (Adv)', color=colors[i], linestyle='--')
        
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def parse_training_log(self, log_path: str) -> pd.DataFrame:
        """
        Parse training log file to extract training metrics.
        
        Args:
            log_path: Path to log.txt file
            
        Returns:
            DataFrame with training metrics
        """
        log_file = Path(log_path)
        if not log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")
        
        data = []
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Find the header line
        header_found = False
        for line in lines:
            line = line.strip()
            if line.startswith("Epoch") and ("CALoss" in line or "Loss" in line):
                # This is the header line - handle both CALoss and Loss formats
                header = line.split()
                header_found = True
                continue
            
            if header_found and line and not line.startswith("Epoch"):
                # This is a data line
                try:
                    values = line.split()
                    if len(values) >= len(header):
                        row_data = {}
                        for i, col in enumerate(header):
                            if i < len(values):
                                # Try to convert to float, keep as string if it fails
                                try:
                                    row_data[col] = float(values[i])
                                except ValueError:
                                    row_data[col] = values[i]
                        data.append(row_data)
                except Exception as e:
                    # Skip malformed lines
                    continue
        
        if not data:
            raise ValueError(f"No valid training data found in {log_path}")
        
        return pd.DataFrame(data)

    def generate_training_llc_plot(self, model_dir: str, experiment_dir: str, 
                                  eff_dim_path: Optional[str] = None, output_name: Optional[str] = None) -> str:
        """
        Generate combined training metrics and LLC trajectory plots.
        
        Args:
            model_dir: Path to model directory containing log.txt
            experiment_dir: Path to LLC experiment directory
            eff_dim_path: Optional path to effective dimensionality JSON file
            output_name: Custom output filename (without extension)
            
        Returns:
            Path to generated plot
        """
        # Load training log
        log_path = Path(model_dir) / "log.txt"
        print(f"Loading training log from: {log_path}")
        training_data = self.parse_training_log(str(log_path))
        
        # Load LLC data
        print(f"Loading LLC data from: {experiment_dir}")
        llc_data = self.load_experiment_data(experiment_dir)
        
        # Load effective dimensionality data if provided
        eff_dim_data = None
        if eff_dim_path:
            try:
                print(f"Loading effective dimensionality data from: {eff_dim_path}")
                with open(eff_dim_path, 'r') as f:
                    eff_dim_data = json.load(f)
                print("✅ Effective dimensionality data loaded successfully")
            except Exception as e:
                print(f"⚠️ Could not load effective dimensionality data: {e}")
                eff_dim_data = None
        
        if output_name is None:
            model_name = llc_data.get('model_name', 'Unknown')
            defense_method = llc_data.get('defense_method', 'Unknown')
            output_name = f"{model_name}_{defense_method}_training_llc_combined"
        
        # Always save plot inside the experiment directory unless custom output_dir is explicitly specified
        if self.output_dir == Path.cwd():
            output_path = Path(experiment_dir) / f"{output_name}.png"
        else:
            output_path = self.output_dir / f"{output_name}.png"
        
        print(f"Generating combined training + LLC plot...")
        self._create_training_llc_visualization(training_data, llc_data, output_path, eff_dim_data)
        
        print(f"✅ Plot saved to: {output_path}")
        return str(output_path)

    def _create_training_llc_visualization(self, training_data: pd.DataFrame, 
                                         llc_data: Dict, output_path: Path, eff_dim_data: Optional[Dict] = None):
        """
        Create visualization combining training metrics, LLC trajectories, and effective dimensionality.
        
        Args:
            training_data: DataFrame with training metrics
            llc_data: Dictionary with LLC experiment data
            output_path: Path to save the plot
            eff_dim_data: Optional dictionary with effective dimensionality data
        """
        # Extract LLC data
        clean_llc_means = llc_data['clean_trajectory']['llc_means']
        clean_llc_stds = llc_data['clean_trajectory']['llc_stds']
        adv_llc_means = llc_data['adversarial_trajectory']['llc_means']
        adv_llc_stds = llc_data['adversarial_trajectory']['llc_stds']
        
        # Filter out None values from LLC data
        clean_valid_indices = [i for i, x in enumerate(clean_llc_means) if x is not None]
        adv_valid_indices = [i for i, x in enumerate(adv_llc_means) if x is not None]
        
        valid_clean_llc = [clean_llc_means[i] for i in clean_valid_indices]
        valid_clean_llc_stds = [clean_llc_stds[i] for i in clean_valid_indices]
        valid_adv_llc = [adv_llc_means[i] for i in adv_valid_indices]
        valid_adv_llc_stds = [adv_llc_stds[i] for i in adv_valid_indices]
        
        # Extract effective dimensionality data if available
        eff_dim_checkpoints = []
        eff_dim_values = []
        if eff_dim_data:
            for checkpoint_idx, checkpoint_data in eff_dim_data.items():
                if isinstance(checkpoint_idx, str) and checkpoint_idx.isdigit():
                    checkpoint_idx = int(checkpoint_idx)
                eff_dim_checkpoints.append(checkpoint_idx * 10)  # Convert to epoch numbers
                eff_dim_values.append(checkpoint_data['effective_dimensionality'])
        
        # Create figure with 2x1 subplots with better spacing
        fig, axes = plt.subplots(2, 1, figsize=(18, 14))
        plt.subplots_adjust(hspace=0.5, top=0.90)  # Increase spacing and adjust top margin
        
        model_name = llc_data.get('model_name', 'Unknown')
        defense_method = llc_data.get('defense_method', 'Unknown')
        fig.suptitle(f'Training Metrics + LLC Analysis: {model_name} ({defense_method})', 
                    fontsize=16, y=0.95)
        
        epochs = training_data['Epoch'].values
        
        # Top plot: Clean data metrics combined
        ax1 = axes[0]
        
        # Create secondary y-axes for different metrics
        ax1_loss = ax1
        ax1_acc = ax1.twinx()
        ax1_llc = ax1.twinx()
        ax1_eff = ax1.twinx() if eff_dim_data else None
        
        # Offset the axes
        ax1_llc.spines['right'].set_position(('outward', 60))
        if ax1_eff:
            ax1_eff.spines['right'].set_position(('outward', 120))
        
        # Plot training loss (left y-axis) - handle both CALoss and Loss column names
        loss_col = 'CALoss' if 'CALoss' in training_data.columns else 'Loss'
        line1 = ax1_loss.plot(epochs, training_data[loss_col], 'b-', linewidth=2, label='Training Loss')
        ax1_loss.set_xlabel('Epoch')
        ax1_loss.set_ylabel('Cross-Entropy Loss', color='blue')
        ax1_loss.tick_params(axis='y', labelcolor='blue')
        
        # Plot clean accuracies (middle y-axis)
        lines_acc = []
        if 'Clean(Tr)' in training_data.columns:
            line2 = ax1_acc.plot(epochs, training_data['Clean(Tr)'], 'g-', linewidth=2, label='Clean Train Acc')
            lines_acc.extend(line2)
        if 'Clean(Val)' in training_data.columns:
            line3 = ax1_acc.plot(epochs, training_data['Clean(Val)'], 'orange', linewidth=2, label='Clean Test Acc')
            lines_acc.extend(line3)
        
        ax1_acc.set_ylabel('Accuracy (%)', color='green')
        ax1_acc.tick_params(axis='y', labelcolor='green')
        
        # Set dynamic y-limits for clean accuracies
        clean_acc_values = []
        if 'Clean(Tr)' in training_data.columns:
            clean_acc_values.extend(training_data['Clean(Tr)'].values)
        if 'Clean(Val)' in training_data.columns:
            clean_acc_values.extend(training_data['Clean(Val)'].values)
        
        if clean_acc_values:
            min_acc = min(clean_acc_values) - 2  # Add some padding
            max_acc = max(clean_acc_values) + 2
            ax1_acc.set_ylim(max(0, min_acc), min(100, max_acc))
        
        # Plot clean LLC trajectory (right y-axis)
        lines_llc = []
        if valid_clean_llc:
            # Convert checkpoint indices to epoch numbers (assuming checkpoints every 10 epochs)
            llc_epochs = [idx * 10 for idx in clean_valid_indices]
            line4 = ax1_llc.errorbar(llc_epochs, valid_clean_llc, yerr=valid_clean_llc_stds,
                                   marker='o', capsize=5, capthick=2, linewidth=2, markersize=6,
                                   color='navy', label='Clean LLC')
            lines_llc.append(line4)
        
        ax1_llc.set_ylabel('LLC', color='navy')
        ax1_llc.tick_params(axis='y', labelcolor='navy')
        
        # Plot effective dimensionality (far right y-axis)
        lines_eff = []
        if ax1_eff and eff_dim_checkpoints:
            line_eff = ax1_eff.plot(eff_dim_checkpoints, eff_dim_values, 
                                  'D-', linewidth=2, markersize=5, color='purple', 
                                  label='Effective Dimensionality')
            lines_eff.extend(line_eff)
            ax1_eff.set_ylabel('Effective Dimensionality', color='purple')
            ax1_eff.tick_params(axis='y', labelcolor='purple')
        
        # Add legend
        all_lines = line1 + lines_acc + lines_llc + lines_eff
        labels = [l.get_label() for l in all_lines]
        ax1.legend(all_lines, labels, loc='upper right', fontsize=10)
        
        title_suffix = ' + Effective Dimensionality' if eff_dim_data else ''
        ax1.set_title(f'Clean Data: Training Loss + Accuracies + LLC{title_suffix}', 
                     fontsize=12, pad=25)
        ax1.set_xlim(0, 100)  # Set x-axis bounds from 0 to 200 epochs
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Adversarial data metrics combined
        ax2 = axes[1]
        
        # Create secondary y-axes for different metrics
        ax2_loss = ax2
        ax2_acc = ax2.twinx()
        ax2_llc = ax2.twinx()
        ax2_eff = ax2.twinx() if eff_dim_data else None
        
        # Offset the axes
        ax2_llc.spines['right'].set_position(('outward', 60))
        if ax2_eff:
            ax2_eff.spines['right'].set_position(('outward', 120))
        
        # Plot training loss (left y-axis) - handle both CALoss and Loss column names
        line5 = ax2_loss.plot(epochs, training_data[loss_col], 'b-', linewidth=2, label='Training Loss')
        ax2_loss.set_xlabel('Epoch')
        ax2_loss.set_ylabel('Cross-Entropy Loss', color='blue')
        ax2_loss.tick_params(axis='y', labelcolor='blue')
        
        # Plot PGD accuracies (middle y-axis)
        lines_acc_adv = []
        if 'PGD(Tr)' in training_data.columns:
            line6 = ax2_acc.plot(epochs, training_data['PGD(Tr)'], 'r-', linewidth=2, label='PGD Train Acc')
            lines_acc_adv.extend(line6)
        if 'PGD(Val)' in training_data.columns:
            line7 = ax2_acc.plot(epochs, training_data['PGD(Val)'], 'darkred', linewidth=2, label='PGD Test Acc')
            lines_acc_adv.extend(line7)
        
        ax2_acc.set_ylabel('Accuracy (%)', color='red')
        ax2_acc.tick_params(axis='y', labelcolor='red')
        
        # Set dynamic y-limits for adversarial accuracies
        adv_acc_values = []
        if 'PGD(Tr)' in training_data.columns:
            adv_acc_values.extend(training_data['PGD(Tr)'].values)
        if 'PGD(Val)' in training_data.columns:
            adv_acc_values.extend(training_data['PGD(Val)'].values)
        
        if adv_acc_values:
            min_acc = min(adv_acc_values) - 5  # Add some padding
            max_acc = max(adv_acc_values) + 5
            ax2_acc.set_ylim(max(0, min_acc), min(100, max_acc))
        
        # Plot adversarial LLC trajectory (right y-axis)
        lines_llc_adv = []
        if valid_adv_llc:
            # Convert checkpoint indices to epoch numbers (assuming checkpoints every 10 epochs)
            llc_epochs = [idx * 10 for idx in adv_valid_indices]
            line8 = ax2_llc.errorbar(llc_epochs, valid_adv_llc, yerr=valid_adv_llc_stds,
                                   marker='s', capsize=5, capthick=2, linewidth=2, markersize=6,
                                   color='maroon', label='Adversarial LLC')
            lines_llc_adv.append(line8)
        
        ax2_llc.set_ylabel('LLC', color='maroon')
        ax2_llc.tick_params(axis='y', labelcolor='maroon')
        
        # Plot effective dimensionality (far right y-axis) - same data as top plot
        lines_eff_adv = []
        if ax2_eff and eff_dim_checkpoints:
            line_eff_adv = ax2_eff.plot(eff_dim_checkpoints, eff_dim_values, 
                                      'D-', linewidth=2, markersize=5, color='purple', 
                                      label='Effective Dimensionality')
            lines_eff_adv.extend(line_eff_adv)
            ax2_eff.set_ylabel('Effective Dimensionality', color='purple')
            ax2_eff.tick_params(axis='y', labelcolor='purple')
        
        # Add legend
        all_lines_adv = line5 + lines_acc_adv + lines_llc_adv + lines_eff_adv
        labels_adv = [l.get_label() for l in all_lines_adv]
        ax2.legend(all_lines_adv, labels_adv, loc='upper right', fontsize=10)
        
        title_suffix_adv = ' + Effective Dimensionality' if eff_dim_data else ''
        ax2.set_title(f'Adversarial Data: Training Loss + PGD Accuracies + LLC{title_suffix_adv}', 
                     fontsize=12, pad=25)
        ax2.set_xlim(0, 100)  # Set x-axis bounds from 0 to 200 epochs
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate LLC comparison plots from existing results')
    parser.add_argument('--mode', choices=['clean_vs_adv', 'defense_comparison', 'training_llc'], required=True,
                       help='Type of plot to generate')
    parser.add_argument('--experiment_dir', type=str,
                       help='Single experiment directory (for clean_vs_adv mode)')
    parser.add_argument('--experiment_dirs', nargs='+', type=str,
                       help='Multiple experiment directories (for defense_comparison mode)')
    parser.add_argument('--results_dir', type=str,
                       help='Auto-discover experiments in this directory (for defense_comparison mode)')
    parser.add_argument('--model_dir', type=str,
                       help='Model directory containing log.txt (for training_llc mode)')
    parser.add_argument('--eff_dim_path', type=str,
                       help='Path to effective dimensionality JSON file (for training_llc mode)')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                       help='Dataset name for plot titles')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for plots')
    parser.add_argument('--output_name', type=str,
                       help='Custom output filename (without extension)')
    
    args = parser.parse_args()
    
    generator = LLCPlotGenerator(output_dir=args.output_dir)
    
    try:
        if args.mode == 'clean_vs_adv':
            if not args.experiment_dir:
                raise ValueError("--experiment_dir is required for clean_vs_adv mode")
            
            generator.generate_clean_vs_adv_plot(args.experiment_dir, args.output_name)
            
        elif args.mode == 'defense_comparison':
            experiment_dirs = args.experiment_dirs
            
            if not experiment_dirs and args.results_dir:
                print(f"Auto-discovering experiments in: {args.results_dir}")
                experiment_dirs = generator.auto_discover_experiments(args.results_dir)
                print(f"Found {len(experiment_dirs)} experiments")
            
            if not experiment_dirs:
                raise ValueError("No experiment directories specified or found")
            
            generator.generate_defense_comparison_plot(experiment_dirs, args.dataset, args.output_name)
            
        elif args.mode == 'training_llc':
            if not args.model_dir or not args.experiment_dir:
                raise ValueError("Both --model_dir and --experiment_dir are required for training_llc mode")
            
            generator.generate_training_llc_plot(args.model_dir, args.experiment_dir, args.eff_dim_path, args.output_name)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
