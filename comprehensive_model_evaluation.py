#!/usr/bin/env python3
"""
Simplified Model Evaluation Script

This script provides 3 clean modes:
1. "clean-test-llc": Calculate LLC trajectory on clean test data across training checkpoints
2. "adv-data-llc": Calculate LLC trajectory on adversarially perturbed test data across training checkpoints  
3. "plot-compare-trajectories": Plot clean and adversarial trajectories together with training metrics

Usage:
    # Generate clean test LLC trajectory
    python comprehensive_model_evaluation.py --mode clean-test-llc --checkpoint_dir ./models/ResNet18_AT/epoch_iter/ --model_name ResNet18 --output_dir ./results/
    
    # Generate adversarial test LLC trajectory (single epsilon)
    python comprehensive_model_evaluation.py --mode adv-data-llc --checkpoint_dir ./models/ResNet18_AT/epoch_iter/ --model_name ResNet18 --output_dir ./results/ --adversarial_eps 0.1
    
    # Generate multi-epsilon adversarial LLC analysis and comparison plot
    python comprehensive_model_evaluation.py --mode adv-data-llc --checkpoint_dir ./models/ResNet18_AT/epoch_iter/ --model_name ResNet18 --output_dir ./results/ --test_multiple_epsilons
    
    # Plot comparison of trajectories with training metrics
    python comprehensive_model_evaluation.py --mode plot-compare-trajectories --model_path ./models/ResNet18_AT/best.pth --model_name ResNet18 --clean_trajectory_path ./results/clean_trajectory.json --adv_trajectory_path ./results/adv_trajectory.json
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

# Import existing modules
from AT_replication_complete import create_model_and_config, setup_cifar10_data, setup_mnist_data
from llc_measurement import LLCMeasurer, LLCConfig
from llc_analysis_pipeline import LLCAnalysisPipeline


class SimplifiedModelEvaluator:
    """
    Simplified evaluation suite with 3 clean modes:
    1. clean-test-llc: Generate LLC trajectory on clean test data
    2. adv-data-llc: Generate LLC trajectory on adversarial test data
    3. plot-compare-trajectories: Plot trajectories with training metrics
    """
    
    def __init__(self, output_dir: str = "./evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {self.output_dir}")
    
    def mode_clean_test_llc(self, 
                           checkpoint_dir: str,
                                  model_name: str,
                                  dataset_name: str = "CIFAR10",
                           defense_method: str = "Unknown",
                           max_checkpoints: int = None,
                           calibration_path: str = None,
                           skip_calibration: bool = False) -> str:
        """
        Mode 1: Calculate LLC trajectory on clean test data across training checkpoints
        
        Returns:
            Path to the generated trajectory JSON file
        """
        print(f"\n{'='*60}")
        print(f"MODE: CLEAN TEST LLC TRAJECTORY")
        print(f"{'='*60}")
        
        # Setup LLC config for clean data (using fast/practical settings)
        llc_config = LLCConfig(
            model_name=model_name,
            data_type="clean",
            epsilon=1e-4,
            gamma=1.0,
            num_chains=2,  # Fast setting (was 8 in theory, 2 in practice)
            num_steps=500,  # Fast setting (was 2000 in theory, 500 in practice)
            batch_size=512,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize pipeline
        pipeline = LLCAnalysisPipeline(llc_config, str(self.output_dir))
        
        # Handle calibration if provided
        if calibration_path and skip_calibration:
            print(f"Loading calibration from: {calibration_path}")
            optimal_params = self._load_calibration_from_json(calibration_path)
        if optimal_params:
                pipeline.optimal_hyperparams = optimal_params
                print(f"Using calibration parameters: {optimal_params}")
        
        # Run trajectory analysis on TEST data
        trajectory_results = pipeline.analyze_checkpoint_trajectory(
            checkpoint_dir=checkpoint_dir,
            model_name=model_name,
            dataset_name=dataset_name,
            defense_method=defense_method,
            max_checkpoints=max_checkpoints,
            skip_calibration=skip_calibration,
            calibration_path=calibration_path,
            data_split="test"  # Use TEST data
        )
        
        # Save trajectory results with a clean name (convert numpy types to JSON-serializable)
        trajectory_path = self.output_dir / "clean_test_llc_trajectory.json"
        with open(trajectory_path, 'w') as f:
            json.dump(self._convert_to_serializable(trajectory_results), f, indent=2)
        
        print(f"✅ Clean test LLC trajectory saved to: {trajectory_path}")
        return str(trajectory_path)
    
    def mode_adv_data_llc(self,
                                   checkpoint_dir: str, 
                                   model_name: str,
                                   dataset_name: str = "CIFAR10",
                                   defense_method: str = "Unknown",
                         max_checkpoints: int = None,
                         calibration_path: str = None,
                         skip_calibration: bool = False,
                         adversarial_eps: float = 8/255,
                         adversarial_steps: int = 10,
                         test_multiple_epsilons: bool = False) -> str:
        """
        Mode 2: Calculate LLC trajectory on adversarially perturbed test data across training checkpoints
        
        If test_multiple_epsilons is True, runs analysis for multiple epsilon values and creates comparison plot.
        
        Returns:
            Path to the generated trajectory JSON file (or comparison plot if multiple epsilons)
        """
        if test_multiple_epsilons:
            return self._run_multiple_epsilon_analysis(
                checkpoint_dir=checkpoint_dir,
                model_name=model_name,
                dataset_name=dataset_name,
                defense_method=defense_method,
                max_checkpoints=max_checkpoints,
                calibration_path=calibration_path,
                skip_calibration=skip_calibration,
                adversarial_steps=adversarial_steps
            )
        
        print(f"\n{'='*60}")
        print(f"MODE: ADVERSARIAL DATA LLC TRAJECTORY (ε={adversarial_eps:.3f})")
        print(f"{'='*60}")
        
        # Setup LLC config for adversarial data (using fast/practical settings)
        llc_config = LLCConfig(
            model_name=model_name,
            data_type="adversarial",
            epsilon=1e-4,
            gamma=1.0,
            num_chains=2,  # Fast setting (was 8 in theory, 2 in practice)
            num_steps=500,  # Fast setting (was 2000 in theory, 500 in practice)
            batch_size=512,
            device="cuda" if torch.cuda.is_available() else "cpu",
            adversarial_attack="pgd",
            adversarial_eps=adversarial_eps,
            adversarial_steps=adversarial_steps
        )
        
        # Initialize pipeline
        pipeline = LLCAnalysisPipeline(llc_config, str(self.output_dir))
        
        # Handle calibration if provided
        if calibration_path and skip_calibration:
            print(f"Loading calibration from: {calibration_path}")
            optimal_params = self._load_calibration_from_json(calibration_path)
            if optimal_params:
                pipeline.optimal_hyperparams = optimal_params
                print(f"Using calibration parameters: {optimal_params}")
        
        # Run trajectory analysis on TEST data with adversarial perturbations
        trajectory_results = pipeline.analyze_checkpoint_trajectory(
            checkpoint_dir=checkpoint_dir,
            model_name=model_name,
            dataset_name=dataset_name,
            defense_method=defense_method,
            max_checkpoints=max_checkpoints,
            skip_calibration=skip_calibration,
            calibration_path=calibration_path,
            data_split="test"  # Use TEST data
        )
        
        # Save trajectory results with epsilon-specific name
        eps_str = f"{adversarial_eps:.3f}".replace(".", "_")
        trajectory_path = self.output_dir / f"adversarial_test_llc_trajectory_eps_{eps_str}.json"
        with open(trajectory_path, 'w') as f:
            json.dump(self._convert_to_serializable(trajectory_results), f, indent=2)
        
        print(f"✅ Adversarial test LLC trajectory (ε={adversarial_eps:.3f}) saved to: {trajectory_path}")
        return str(trajectory_path)
    
    def mode_plot_compare_trajectories(self,
                                     model_path: str,
                                     model_name: str,
                                     clean_trajectory_path: str = None,
                                     adv_trajectory_path: str = None,
                                      save_name: str = None) -> None:
        """
        Mode 3: Plot clean and adversarial trajectories together with training metrics
        
        Args:
            model_path: Path to model (used to find log.txt)
            model_name: Model architecture name
            clean_trajectory_path: Path to clean LLC trajectory JSON
            adv_trajectory_path: Path to adversarial LLC trajectory JSON
            save_name: Custom name for saved plot
        """
        print(f"\n{'='*60}")
        print(f"MODE: PLOT COMPARE TRAJECTORIES")
        print(f"{'='*60}")
        
        if not clean_trajectory_path and not adv_trajectory_path:
            raise ValueError("At least one trajectory path must be provided")
        
        # Find training log file
        model_path = Path(model_path)
        model_dir = model_path.parent
        log_file = model_dir / "log.txt"
        
        if not log_file.exists():
            print(f"Warning: No training log found at: {log_file}")
            print("Plot will only show LLC trajectories without training metrics")
            training_metrics = {}
        else:
            training_metrics = self._parse_training_log(str(log_file))
            print(f"Loaded training metrics from: {log_file}")
        
        # Load trajectory data
        clean_data = None
        if clean_trajectory_path:
            clean_data = self._load_trajectory_json(clean_trajectory_path)
            print(f"Loaded clean trajectory from: {clean_trajectory_path}")
        
        adv_data = None
        if adv_trajectory_path:
            adv_data = self._load_trajectory_json(adv_trajectory_path)
            print(f"Loaded adversarial trajectory from: {adv_trajectory_path}")
        
        # Create the comparison plot
        if save_name is None:
            save_name = f"{model_name}_trajectory_comparison"
        
        self._create_trajectory_comparison_plot(
            training_metrics=training_metrics,
            clean_trajectory_data=clean_data,
            adv_trajectory_data=adv_data,
            model_name=model_name,
            save_name=save_name
        )
    
    def _run_multiple_epsilon_analysis(self,
                                     checkpoint_dir: str,
                                     model_name: str,
                                                      dataset_name: str = "CIFAR10",
                                     defense_method: str = "Unknown",
                                                      max_checkpoints: int = None,
                                                      calibration_path: str = None,
                                                      skip_calibration: bool = False,
                                     adversarial_steps: int = 10) -> str:
        """
        Run LLC trajectory analysis across multiple epsilon values and create comparison plot
        
        Returns:
            Path to the generated comparison plot
        """
        print(f"\n{'='*70}")
        print(f"MODE: MULTI-EPSILON ADVERSARIAL LLC TRAJECTORY ANALYSIS")
        print(f"{'='*70}")
        
        # Define epsilon values to test in range [2/255, 16/255] plus clean baseline
        epsilon_values = [0.0, 2/255, 4/255, 8/255, 12/255, 16/255]  # [0.0, 0.008, 0.016, 0.031, 0.047, 0.063]
        print(f"Testing epsilon values: {[f'{eps:.3f}' for eps in epsilon_values]}")
        
        # Store trajectory results for each epsilon
        epsilon_trajectories = {}
        
        for eps in epsilon_values:
            print(f"\n🔥 Analyzing ε = {eps:.2f}")
            
            # Special case for epsilon = 0.0 (clean data)
            if eps == 0.0:
                data_type = "clean"
                adversarial_eps = None
                adversarial_attack = None
            else:
                data_type = "adversarial"
                adversarial_eps = eps
                adversarial_attack = "pgd"
            
            # Setup LLC config for this epsilon
            llc_config = LLCConfig(
                model_name=model_name,
                data_type=data_type,
                epsilon=1e-4,
                gamma=1.0,
                num_chains=2,  # Fast setting
                num_steps=500,  # Fast setting
                batch_size=512,
                device="cuda" if torch.cuda.is_available() else "cpu",
                adversarial_attack=adversarial_attack,
                adversarial_eps=adversarial_eps,
                adversarial_steps=adversarial_steps if eps > 0 else None
            )
            
            # Initialize pipeline
            pipeline = LLCAnalysisPipeline(llc_config, str(self.output_dir))
            
            # Handle calibration if provided
            if calibration_path and skip_calibration:
                optimal_params = self._load_calibration_from_json(calibration_path)
                if optimal_params:
                    pipeline.optimal_hyperparams = optimal_params
            
            # Run trajectory analysis
            trajectory_results = pipeline.analyze_checkpoint_trajectory(
                checkpoint_dir=checkpoint_dir,
                model_name=model_name,
                dataset_name=dataset_name,
                defense_method=defense_method,
                max_checkpoints=max_checkpoints,
                skip_calibration=skip_calibration,
                calibration_path=calibration_path,
                data_split="test"
            )
            
            # Save individual trajectory file
            eps_str = f"{eps:.2f}".replace(".", "_")
            trajectory_path = self.output_dir / f"llc_trajectory_eps_{eps_str}.json"
            with open(trajectory_path, 'w') as f:
                json.dump(self._convert_to_serializable(trajectory_results), f, indent=2)
            
            # Store for plotting
            epsilon_trajectories[eps] = trajectory_results
            print(f"✅ Trajectory for ε={eps:.2f} saved to: {trajectory_path}")
        
        # Create multi-epsilon comparison plot
        plot_path = self._create_multi_epsilon_plot(epsilon_trajectories, model_name)
        
        print(f"\n✅ Multi-epsilon analysis completed!")
        print(f"📊 Comparison plot saved to: {plot_path}")
        return plot_path
    
    def _create_trajectory_comparison_plot(self,
                                         training_metrics: Dict[int, Dict[str, float]],
                                         clean_trajectory_data: Dict[str, Any] = None,
                                         adv_trajectory_data: Dict[str, Any] = None,
                                         model_name: str = "Unknown",
                                         save_name: str = "trajectory_comparison") -> None:
        """Create 3-panel comparison plot: clean (top) vs adversarial (middle) vs direct LLC comparison (bottom)"""
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 18))
        plt.subplots_adjust(hspace=0.4, top=0.94)
        
        fig.suptitle(f'{model_name}: Training Metrics + LLC Trajectory Comparison', 
                    fontsize=16, fontweight='bold')
        
        # Extract training data
        epochs = list(training_metrics.keys()) if training_metrics else []
        
        # ============= TOP PANEL: Clean Data =============
        ax1.set_title(f'Clean Data: Training Metrics + LLC Trajectory ({model_name})', fontsize=14)
        
        if training_metrics:
            # Create multiple y-axes for top panel
            ax1_loss = ax1
            ax1_acc = ax1.twinx()
            ax1_llc = ax1.twinx()
            
            # Offset the LLC axis
            ax1_llc.spines['right'].set_position(('outward', 60))
            
            # Plot validation loss
            val_losses = [training_metrics[e].get('Loss', None) for e in epochs]
            val_losses = [l for l in val_losses if l is not None]
            if val_losses:
                ax1_loss.plot(epochs[:len(val_losses)], val_losses, 'b-', linewidth=2, label='Val Loss')
                ax1_loss.set_ylabel('Cross-Entropy Loss', color='blue')
                ax1_loss.tick_params(axis='y', labelcolor='blue')
            
                # Plot clean test accuracy
                clean_test = [training_metrics[e].get('Clean(Val)', None) for e in epochs]
                clean_test = [c for c in clean_test if c is not None]
                if clean_test:
                    ax1_acc.plot(epochs[:len(clean_test)], clean_test, 'orange', linewidth=2, label='Clean Test Acc')
                ax1_acc.set_ylabel('Accuracy (%)', color='orange')
                ax1_acc.tick_params(axis='y', labelcolor='orange')
                
                # Set dynamic y-limits
                min_acc = min(clean_test) - 2
                max_acc = max(clean_test) + 2
                ax1_acc.set_ylim(max(0, min_acc), min(100, max_acc))
        else:
            ax1_llc = ax1
        
        # Plot Clean LLC trajectory
        if clean_trajectory_data:
            clean_epochs, clean_llc_means, clean_llc_stds = self._extract_trajectory_data(clean_trajectory_data)
            if clean_epochs and clean_llc_means:
                ax1_llc.errorbar(clean_epochs, clean_llc_means, yerr=clean_llc_stds, 
                               marker='o', capsize=3, capthick=1, markersize=6,
                           color='navy', label='Clean Test LLC', linewidth=2)
            ax1_llc.set_ylabel('LLC', color='navy')
            ax1_llc.tick_params(axis='y', labelcolor='navy')
        
        ax1.set_xlabel('Epoch')
        ax1.grid(True, alpha=0.3)
        
        # Add legend for top panel
        lines1 = []
        labels1 = []
        if training_metrics and val_losses:
            lines1.append(ax1_loss.lines[0])
            labels1.append('Val Loss')
        if training_metrics and clean_test:
            lines1.append(ax1_acc.lines[0])
            labels1.append('Clean Test Acc')
        if clean_trajectory_data and len(ax1_llc.lines) > 0:
            lines1.append(ax1_llc.lines[0])
            labels1.append('Clean Test LLC')
        
        if lines1:
            ax1.legend(lines1, labels1, loc='upper right')
        
        # ============= BOTTOM PANEL: Adversarial Data =============
        ax2.set_title(f'Adversarial Data: Training Metrics + LLC Trajectory ({model_name})', fontsize=14)
        
        if training_metrics:
            # Create multiple y-axes for bottom panel
            ax2_loss = ax2
            ax2_acc = ax2.twinx()
            ax2_llc = ax2.twinx()
            
            # Offset the LLC axis
            ax2_llc.spines['right'].set_position(('outward', 60))
        
            # Plot validation loss (same as top)
        if val_losses:
            ax2_loss.plot(epochs[:len(val_losses)], val_losses, 'b-', linewidth=2, label='Val Loss')
            ax2_loss.set_ylabel('Cross-Entropy Loss', color='blue')
            ax2_loss.tick_params(axis='y', labelcolor='blue')
        
            # Plot adversarial test accuracies
            pgd_test = [training_metrics[e].get('PGD(Val)', None) for e in epochs]
            fgsm_test = [training_metrics[e].get('FGSM(Val)', None) for e in epochs]
        
        lines_adv = []
        if pgd_test:
            pgd_test = [p for p in pgd_test if p is not None]
            if pgd_test:
                line_pgd = ax2_acc.plot(epochs[:len(pgd_test)], pgd_test, 'red', linewidth=2, label='PGD Test Acc')
                lines_adv.extend(line_pgd)
        
        if fgsm_test:
            fgsm_test = [f for f in fgsm_test if f is not None]
            if fgsm_test:
                line_fgsm = ax2_acc.plot(epochs[:len(fgsm_test)], fgsm_test, 'purple', linewidth=2, label='FGSM Test Acc')
                lines_adv.extend(line_fgsm)
        
        if lines_adv:
            ax2_acc.set_ylabel('Adversarial Accuracy (%)', color='red')
            ax2_acc.tick_params(axis='y', labelcolor='red')
            
                # Set dynamic y-limits
            all_adv_vals = []
            if pgd_test:
                all_adv_vals.extend(pgd_test)
            if fgsm_test:
                all_adv_vals.extend(fgsm_test)
            
            if all_adv_vals:
                min_adv = min(all_adv_vals) - 2
                max_adv = max(all_adv_vals) + 2
                ax2_acc.set_ylim(max(0, min_adv), min(100, max_adv))
        else:
            ax2_llc = ax2
        
        # Plot Adversarial LLC trajectory
        if adv_trajectory_data:
            adv_epochs, adv_llc_means, adv_llc_stds = self._extract_trajectory_data(adv_trajectory_data)
            if adv_epochs and adv_llc_means:
                ax2_llc.errorbar(adv_epochs, adv_llc_means, yerr=adv_llc_stds, 
                               marker='o', capsize=3, capthick=1, markersize=6,
                           color='darkred', label='Adversarial Test LLC', linewidth=2)
            ax2_llc.set_ylabel('LLC', color='darkred')
            ax2_llc.tick_params(axis='y', labelcolor='darkred')
        
        ax2.set_xlabel('Epoch')
        ax2.grid(True, alpha=0.3)
        
        # Add legend for bottom panel
        lines2 = []
        labels2 = []
        
        # Initialize lines_adv in case training_metrics is None
        if 'lines_adv' not in locals():
            lines_adv = []
        if training_metrics and val_losses:
            lines2.append(ax2_loss.lines[0])
            labels2.append('Val Loss')
        if training_metrics and pgd_test:
            labels2.append('PGD Test Acc')
        if training_metrics and fgsm_test:
            labels2.append('FGSM Test Acc')
        
        # Add adversarial accuracy line objects if they exist
        if lines_adv:
            lines2.extend(lines_adv)
        if adv_trajectory_data and len(ax2_llc.lines) > 0:
            lines2.append(ax2_llc.lines[0])
            labels2.append('Adversarial Test LLC')
        
        if lines2:
            ax2.legend(lines2, labels2, loc='upper right')
        
        # ============= BOTTOM PANEL: Direct LLC Comparison =============
        ax3.set_title(f'Direct LLC Comparison: Clean vs Adversarial Test Data ({model_name})', fontsize=14)
        
        # Plot both LLC trajectories on the same axes for direct comparison
        lines3 = []
        labels3 = []
        
        # Plot Clean LLC trajectory
        if clean_trajectory_data:
            clean_epochs, clean_llc_means, clean_llc_stds = self._extract_trajectory_data(clean_trajectory_data)
            if clean_epochs and clean_llc_means:
                line_clean = ax3.errorbar(clean_epochs, clean_llc_means, yerr=clean_llc_stds, 
                                        marker='o', capsize=3, capthick=1, markersize=6,
                                        color='navy', label='Clean Test LLC', linewidth=2, alpha=0.8)
                lines3.append(line_clean)
                labels3.append('Clean Test LLC')
        
        # Plot Adversarial LLC trajectory
        if adv_trajectory_data:
            adv_epochs, adv_llc_means, adv_llc_stds = self._extract_trajectory_data(adv_trajectory_data)
            if adv_epochs and adv_llc_means:
                line_adv = ax3.errorbar(adv_epochs, adv_llc_means, yerr=adv_llc_stds, 
                                      marker='s', capsize=3, capthick=1, markersize=6,
                                      color='darkred', label='Adversarial Test LLC', linewidth=2, alpha=0.8)
                lines3.append(line_adv)
                labels3.append('Adversarial Test LLC')
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('LLC', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Add legend for bottom panel
        if lines3:
            ax3.legend(fontsize=12, loc='best')
        
        # Set tight y-axis limits for better visibility of differences
        all_llc_values = []
        
        # Collect all LLC values (means and error bounds)
        if clean_trajectory_data and clean_epochs and clean_llc_means:
            all_llc_values.extend(clean_llc_means)
            if clean_llc_stds:
                # Include error bounds in range calculation
                for mean, std in zip(clean_llc_means, clean_llc_stds):
                    all_llc_values.extend([mean - std, mean + std])
        
        if adv_trajectory_data and adv_epochs and adv_llc_means:
            all_llc_values.extend(adv_llc_means)
            if adv_llc_stds:
                # Include error bounds in range calculation
                for mean, std in zip(adv_llc_means, adv_llc_stds):
                    all_llc_values.extend([mean - std, mean + std])
        
        # Set tight y-axis limits with small padding
        if all_llc_values:
            y_min = min(all_llc_values)
            y_max = max(all_llc_values)
            y_range = y_max - y_min
            
            # Use 10% padding if there's a reasonable range, otherwise use fixed padding
            if y_range > 1e-6:  # Avoid division by zero for very small ranges
                padding = y_range * 0.1
            else:
                padding = 0.01  # Small fixed padding for very flat trajectories
            
            ax3.set_ylim(y_min - padding, y_max + padding)
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        print(f"✅ Trajectory comparison plot saved to: {save_path}")
        plt.show()
    
    def _create_multi_epsilon_plot(self, epsilon_trajectories: Dict[float, Dict[str, Any]], model_name: str) -> str:
        """
        Create a plot showing LLC evolution across different epsilon values (similar to attached image)
        
        Args:
            epsilon_trajectories: Dict mapping epsilon values to trajectory results
            model_name: Model architecture name
            
        Returns:
            Path to the saved plot
        """
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Color scheme for different epsilons in range [2/255, 16/255]
        colors = {
            0.0: '#8B4513',        # Brown for ε=0.0 (clean)
            2/255: '#FF6347',      # Tomato for ε=2/255
            4/255: '#FFA500',      # Orange for ε=4/255
            8/255: '#32CD32',      # Green for ε=8/255
            12/255: '#4169E1',     # Blue for ε=12/255
            16/255: '#9370DB'      # Purple for ε=16/255
        }
        
        # Plot each epsilon trajectory
        for eps in sorted(epsilon_trajectories.keys()):
            trajectory_data = epsilon_trajectories[eps]
            
            # Extract trajectory data
            epochs, llc_means, llc_stds = self._extract_trajectory_data(trajectory_data)
            
            if epochs and llc_means:
                color = colors.get(eps, 'gray')
                
                # Plot with error bars and shaded region (like your image)
                if eps == 0.0:
                    label = 'ε=0.0'
                else:
                    label = f'ε={eps:.3f} ({int(eps*255)}/255)'
                
                ax.plot(epochs, llc_means, color=color, linewidth=2, 
                       marker='o', markersize=4, label=label)
                
                # Add shaded error region
                if llc_stds:
                    llc_means_np = np.array(llc_means)
                    llc_stds_np = np.array(llc_stds)
                    ax.fill_between(epochs, 
                                   llc_means_np - llc_stds_np, 
                                   llc_means_np + llc_stds_np,
                                   color=color, alpha=0.2)
        
        # Formatting (similar to your image)
        ax.set_xlabel('Training Checkpoint', fontsize=14)
        ax.set_ylabel('Learning Coefficient (LLC)', fontsize=14)
        ax.set_title(f'LLC Evolution During Adversarial Training\n{model_name}', fontsize=16, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(fontsize=12, loc='best')
        
        # Set reasonable y-axis limits
        all_means = []
        for trajectory_data in epsilon_trajectories.values():
            _, means, _ = self._extract_trajectory_data(trajectory_data)
            all_means.extend(means)
        
        if all_means:
            y_min = min(all_means) - 0.5
            y_max = max(all_means) + 0.5
            ax.set_ylim(y_min, y_max)
        
        # Tight layout
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / f"{model_name}_multi_epsilon_llc_evolution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return str(save_path)
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if hasattr(obj, 'dtype'):  # numpy arrays and scalars
            if obj.dtype.kind in 'fc':  # float or complex
                return float(obj)
            elif obj.dtype.kind in 'iu':  # integer or unsigned
                return int(obj)
            elif obj.dtype.kind == 'b':  # boolean
                return bool(obj)
            else:
                return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def _load_calibration_from_json(self, calibration_path: str) -> Optional[Dict[str, float]]:
        """Load calibrated hyperparameters from JSON file"""
        try:
            calibration_path = Path(calibration_path)
            
            if calibration_path.is_dir():
                calibration_file = calibration_path / "calibration_results.json"
                if not calibration_file.exists():
                    json_files = list(calibration_path.glob("*.json"))
                    if not json_files:
                        print(f"No JSON files found in directory: {calibration_path}")
                        return None
                    calibration_file = json_files[0]
                calibration_path = calibration_file
            
            if not calibration_path.exists():
                print(f"Calibration file not found: {calibration_path}")
                return None
            
            with open(calibration_path, 'r') as f:
                data = json.load(f)
            
            params_source = data.get('optimal_params', data)
            
            if not isinstance(params_source, dict) or 'epsilon' not in params_source:
                print(f"Invalid calibration format. Available keys: {list(data.keys())}")
                return None
            
            optimal_params = {
                'epsilon': float(params_source['epsilon']),
                'gamma': float(params_source.get('gamma', 1.0)),
            }
            
            if 'nbeta' in params_source:
                optimal_params['nbeta'] = float(params_source['nbeta'])
            
            print(f"Loaded calibration parameters: {optimal_params}")
            return optimal_params
                
        except Exception as e:
            print(f"Error loading calibration from {calibration_path}: {e}")
            return None
    
    def _load_trajectory_json(self, trajectory_path: str) -> Optional[Dict[str, Any]]:
        """Load trajectory data from JSON file"""
        try:
            trajectory_path = Path(trajectory_path)
            
            if not trajectory_path.exists():
                print(f"Trajectory file not found: {trajectory_path}")
                return None
            
            with open(trajectory_path, 'r') as f:
                trajectory_data = json.load(f)
            
            return trajectory_data
            
        except Exception as e:
            print(f"Error loading trajectory from {trajectory_path}: {e}")
            return None
    
    def _extract_trajectory_data(self, trajectory_data: Dict[str, Any]) -> Tuple[List[int], List[float], List[float]]:
        """Extract epochs, means, and stds from trajectory data"""
        try:
            if 'results' in trajectory_data:
                # LLCAnalysisPipeline format
                results = trajectory_data['results']
                epochs = []
                llc_means = []
                llc_stds = []
                
                for result in results:
                    if 'epoch' in result and 'llc_results' in result:
                        epochs.append(result['epoch'])
                        llc_means.append(result['llc_results']['llc/mean'])
                        llc_stds.append(result['llc_results']['llc/std'])
                
                return epochs, llc_means, llc_stds
            
            elif 'llc_means' in trajectory_data and 'llc_stds' in trajectory_data:
                # Direct LLCMeasurer format (your actual JSON structure!)
                llc_means = trajectory_data['llc_means']
                llc_stds = trajectory_data['llc_stds']
                checkpoint_names = trajectory_data.get('checkpoint_names', [])
                
                # Convert checkpoint indices to epochs (assuming 30 checkpoints -> 100 epochs)
                num_checkpoints = len(llc_means)
                if num_checkpoints > 0:
                    # Map checkpoints evenly across 100 epochs
                    epochs = [int(1 + (i * 99) / max(1, num_checkpoints - 1)) for i in range(num_checkpoints)]
                else:
                    epochs = []
                
                print(f"✅ Extracted {len(llc_means)} LLC trajectory points (direct format)")
                return epochs, llc_means, llc_stds
            
            else:
                print(f"❌ Unrecognized trajectory format. Available keys: {list(trajectory_data.keys())}")
                return [], [], []
                
        except Exception as e:
            print(f"❌ Error extracting trajectory data: {e}")
            return [], [], []
    
    def _parse_training_log(self, log_file: str) -> Dict[int, Dict[str, float]]:
        """Parse training log file to extract metrics"""
        metrics = {}
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 3 and parts[0].isdigit():
                    epoch = int(parts[0])
                    
                    metric_dict = {}
                    if len(parts) >= 2:
                        try:
                            metric_dict['Loss'] = float(parts[1])
                        except:
                            pass
                    
                    if len(parts) >= 7:
                        try:
                            metric_dict['Clean(Val)'] = float(parts[6])
                        except:
                            pass
                    
                    if len(parts) >= 8:
                        try:
                            metric_dict['PGD(Val)'] = float(parts[7])
                        except:
                            pass
                    
                    if len(parts) >= 9:
                        try:
                            metric_dict['FGSM(Val)'] = float(parts[8])
                        except:
                            pass
                    
                    metrics[epoch] = metric_dict
        
        except Exception as e:
            print(f"Error parsing training log: {e}")
            return {}
        
        return metrics


def main():
    """Main function with clean command-line interface"""
    parser = argparse.ArgumentParser(
        description="Simplified Model Evaluation with 3 Clean Modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate clean test LLC trajectory
  python comprehensive_model_evaluation.py --mode clean-test-llc --checkpoint_dir ./models/ResNet18_AT/epoch_iter/ --model_name ResNet18
  
  # Generate adversarial test LLC trajectory (single epsilon)
  python comprehensive_model_evaluation.py --mode adv-data-llc --checkpoint_dir ./models/ResNet18_AT/epoch_iter/ --model_name ResNet18 --adversarial_eps 0.1
  
  # Generate adversarial test LLC trajectories for multiple epsilons (0.0, 2/255, 4/255, 8/255, 12/255, 16/255) and create comparison plot
  python comprehensive_model_evaluation.py --mode adv-data-llc --checkpoint_dir ./models/ResNet18_AT/epoch_iter/ --model_name ResNet18 --test_multiple_epsilons
  
  # Plot comparison of trajectories
  python comprehensive_model_evaluation.py --mode plot-compare-trajectories --model_path ./models/ResNet18_AT/best.pth --model_name ResNet18 --clean_trajectory_path ./results/clean_test_llc_trajectory.json --adv_trajectory_path ./results/adversarial_test_llc_trajectory.json
        """
    )

    parser.add_argument('--mode', type=str, required=True,
                       choices=['clean-test-llc', 'adv-data-llc', 'plot-compare-trajectories'],
                       help='Evaluation mode') 
    parser.add_argument('--model_name', type=str, required=True, 
                       choices=['LeNet', 'VGG11', 'ResNet18'],
                       help='Model architecture name')  
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                       choices=['CIFAR10', 'MNIST'], help='Dataset name')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    
    # Arguments for trajectory generation modes
    parser.add_argument('--checkpoint_dir', type=str,
                       help='Directory containing training checkpoints (required for trajectory modes)')
    parser.add_argument('--defense_method', type=str,
                       help='Defense method name')
    parser.add_argument('--max_checkpoints', type=int,
                       help='Maximum number of checkpoints to analyze')
    parser.add_argument('--calibration_path', type=str, 
                       help='Path to calibration JSON file')
    parser.add_argument('--skip_calibration', action='store_true',
                       help='Skip calibration and use provided calibration_path')
   
   # Arguments for adversarial mode
    parser.add_argument('--adversarial_eps', type=float, default=8/255,
                       help='Adversarial epsilon value (default: 8/255)')
    parser.add_argument('--adversarial_steps', type=int, default=10,
                       help='Number of adversarial attack steps (default: 10)')
    parser.add_argument('--test_multiple_epsilons', action='store_true',
                       help='Test multiple epsilon values (0.0, 2/255, 4/255, 8/255, 12/255, 16/255) and create comparison plot')
    
    # Arguments for plotting mode
    parser.add_argument('--model_path', type=str,
                       help='Path to model checkpoint (required for plotting mode)')  
    parser.add_argument('--clean_trajectory_path', type=str,
                       help='Path to clean LLC trajectory JSON file')   
    parser.add_argument('--adv_trajectory_path', type=str,
                       help='Path to adversarial LLC trajectory JSON file')  
    parser.add_argument('--save_name', type=str,
                       help='Custom name for saved plot')   
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = SimplifiedModelEvaluator(output_dir=args.output_dir)
    
    if args.mode == 'clean-test-llc':
        if not args.checkpoint_dir:
            print("Error: --checkpoint_dir is required for clean-test-llc mode")
            return
        
        evaluator.mode_clean_test_llc(
            checkpoint_dir=args.checkpoint_dir,
                model_name=args.model_name,
                dataset_name=args.dataset,
            defense_method=args.defense_method,
            max_checkpoints=args.max_checkpoints,
                calibration_path=args.calibration_path,
                skip_calibration=args.skip_calibration
            )
            
    elif args.mode == 'adv-data-llc':
        if not args.checkpoint_dir:
            print("Error: --checkpoint_dir is required for adv-data-llc mode")
            return
        
        evaluator.mode_adv_data_llc(
            checkpoint_dir=args.checkpoint_dir,
            model_name=args.model_name,
            dataset_name=args.dataset,
            defense_method=args.defense_method,
            max_checkpoints=args.max_checkpoints,
            calibration_path=args.calibration_path,
            skip_calibration=args.skip_calibration,
            adversarial_eps=args.adversarial_eps,
            adversarial_steps=args.adversarial_steps,
            test_multiple_epsilons=args.test_multiple_epsilons
        )
        
    elif args.mode == 'plot-compare-trajectories':
        if not args.model_path:
            print("Error: --model_path is required for plot-compare-trajectories mode")
            return
        
        if not args.clean_trajectory_path and not args.adv_trajectory_path:
            print("Error: At least one of --clean_trajectory_path or --adv_trajectory_path is required")
        return
    
        evaluator.mode_plot_compare_trajectories(
            model_path=args.model_path,
            model_name=args.model_name,
            clean_trajectory_path=args.clean_trajectory_path,
            adv_trajectory_path=args.adv_trajectory_path,
            save_name=args.save_name
        )
    
    print(f"\n✅ Evaluation completed! Results saved in: {evaluator.output_dir}")


if __name__ == "__main__":
    main()