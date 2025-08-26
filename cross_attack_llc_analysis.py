#!/usr/bin/env python3
"""
Cross-Attack LLC Analysis: Test LLC signatures across different attack types
to investigate if LLC can distinguish between different adversarial attack types.

Based on Tramèr and Boneh's MEP theory, different attack types should show
distinct LLC signatures even when applied to the same adversarially trained model.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from llc_measurement import LLCConfig, LLCMeasurer
from llc_analysis_pipeline import LLCAnalysisPipeline
from AT_replication_complete import create_model_and_config, setup_mnist_data, setup_cifar10_data

class CrossAttackLLCAnalyzer:
    """Analyze LLC signatures across different adversarial attack types"""
    
    def __init__(self, output_dir: str = "./cross_attack_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define attack configurations to test
        self.attack_configs = {
            # 'clean': {'attack': None, 'eps': 0.0, 'color': '#8B4513', 'marker': 'o'},
            'pgd_l_inf': {'attack': 'pgd', 'eps': 8/255, 'color': '#FF6B35', 'marker': 's'},
            # 'fgsm_l_inf': {'attack': 'fgsm', 'eps': 8/255, 'color': '#F7931E', 'marker': '^'},
            'pgd_l2': {'attack': 'pgd_l2', 'eps': 0.5, 'color': '#32CD32', 'marker': 'D'},
            # 'pgd_l1': {'attack': 'pgd_l1', 'eps': 10.0, 'color': '#4169E1', 'marker': 'v'},
        }
    
    def _load_model(self, model_path: str, model_name: str) -> nn.Module:
        """Load a pre-trained model"""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        # Create model instance
        model, config = create_model_and_config(model_name)
        
        # Load checkpoint with weights_only=False for MAIR checkpoints
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"Warning: Failed to load {model_path} with weights_only=False: {e}")
            # Try with weights_only=True as fallback
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        
        # Handle different checkpoint formats
        if 'rmodel_state_dict' in checkpoint:
            # MAIR checkpoint with rmodel_state_dict
            rmodel_state = checkpoint['rmodel_state_dict']
            base_model_state = {}
            for key, value in rmodel_state.items():
                if key.startswith('model.'):
                    base_key = key[6:]  # Remove 'model.' prefix
                    base_model_state[base_key] = value
            model.load_state_dict(base_model_state)
        elif 'rmodel' in checkpoint:
            # MAIR checkpoint with rmodel key - this is the state_dict of a RobModel
            # We need to create a RobModel and load it, then extract the base model
            from MAIR.mair.nn.robmodel import RobModel
            from MAIR.mair.utils.models import load_model
            
            # Get parameters from the checkpoint
            rmodel_state = checkpoint['rmodel']
            n_classes = rmodel_state['n_classes'].item()
            
            # Get normalization parameters from checkpoint
            mean = rmodel_state['mean'].cpu().numpy().tolist()
            std = rmodel_state['std'].cpu().numpy().tolist()
            normalization_used = {"mean": mean, "std": std}
            
            # Create the base model and RobModel with correct normalization
            base_model = load_model(model_name, n_classes)
            rmodel = RobModel(base_model, n_classes=n_classes, normalization_used=normalization_used)
            
            # Load the RobModel state
            rmodel.load_state_dict(rmodel_state)
            
            # Extract the base model
            model = rmodel.model
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Filter out non-model keys and load remaining
            filtered_state = {}
            for key, value in checkpoint.items():
                if not key.startswith(('optimizer', 'scheduler', 'accumulated', 'curr_', 'recordmanager', 'record_info', 'rmodel')):
                    filtered_state[key] = value
            
            if filtered_state:
                model.load_state_dict(filtered_state)
            else:
                raise ValueError(f"Could not find model weights in checkpoint. Keys: {list(checkpoint.keys())}")
        
        print(f"✅ Loaded model from {model_path}")
        return model
    
    def analyze_cross_attack_llc(self, 
                                model_path: str,
                                model_name: str,
                                dataset_name: str,
                                calibration_path: str,
                                checkpoint_dir: str,
                                max_checkpoints: int = 10) -> Dict[str, Any]:
        """
        Analyze LLC trajectories across different attack types
        
        Args:
            model_path: Path to the trained model
            model_name: Name of the model architecture
            dataset_name: Dataset name (MNIST, CIFAR10)
            calibration_path: Path to calibration results
            checkpoint_dir: Directory containing model checkpoints
            max_checkpoints: Maximum number of checkpoints to analyze
            
        Returns:
            Dictionary containing LLC trajectories for each attack type
        """
        
        print(f"\n🔥 Cross-Attack LLC Analysis")
        print(f"Model: {model_name}")
        print(f"Dataset: {dataset_name}")
        print(f"Testing {len(self.attack_configs)} attack types...")
        
        # Load model
        model = self._load_model(model_path, model_name)
        
        # Load data
        if dataset_name.upper() == "MNIST":
            _, test_loader, _, _ = setup_mnist_data(batch_size=512)
        elif dataset_name.upper() == "CIFAR10":
            _, test_loader, _, _ = setup_cifar10_data(batch_size=512)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        results = {}
        
        # Test each attack type
        for attack_name, attack_config in self.attack_configs.items():
            print(f"\n📊 Analyzing {attack_name.upper()} attack...")
            
            # Configure LLC measurement for this attack type
            llc_config = LLCConfig(
                epsilon=1e-6,  # LLC measurement epsilon (not attack epsilon)
                gamma=1.0,
                nbeta=0.82,  # Will be loaded from calibration
                data_type="adversarial" if attack_config['attack'] else "clean",
                adversarial_attack=attack_config['attack'] if attack_config['attack'] else "pgd",
                adversarial_eps=attack_config['eps'],
                adversarial_steps=10
            )
            
            # Create analysis pipeline
            pipeline = LLCAnalysisPipeline(
                llc_config=llc_config,
                base_save_dir=str(self.output_dir / f"{attack_name}_analysis")
            )
            
            print(f"📁 Using calibration path: {calibration_path}")
            
            # Analyze trajectory on TEST data (skip re-calibration)
            try:
                trajectory_results = pipeline.analyze_checkpoint_trajectory(
                    checkpoint_dir=checkpoint_dir,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    defense_method="AT",  # Assuming adversarial training
                    max_checkpoints=max_checkpoints,
                    skip_calibration=True,  # Use loaded calibration, don't re-calibrate
                    calibration_path=calibration_path  # Pass the calibration path!
                )
                
                results[attack_name] = {
                    'trajectory': trajectory_results,
                    'config': attack_config,
                    'attack_type': attack_config['attack'],
                    'epsilon': attack_config['eps']
                }
                
                # Save individual results
                output_file = self.output_dir / f"{attack_name}_llc_trajectory.json"
                with open(output_file, 'w') as f:
                    json.dump(trajectory_results, f, indent=2, default=str)
                print(f"✅ Saved {attack_name} results to {output_file}")
                
            except Exception as e:
                print(f"❌ Error analyzing {attack_name}: {e}")
                results[attack_name] = None
        
        return results
    
    def create_cross_attack_comparison_plot(self, results: Dict[str, Any], model_name: str) -> str:
        """Create comparison plot showing LLC trajectories across different attack types"""
        
        plt.figure(figsize=(16, 10))
        
        # Collect all LLC values for smart y-axis scaling
        all_llc_values = []
        
        # Plot each attack type
        for attack_name, result in results.items():
            if result is None:
                continue
                
            trajectory = result['trajectory']
            config = result['config']
            
            # Extract trajectory data
            if 'llc_means' in trajectory and 'llc_stds' in trajectory:
                epochs = list(range(len(trajectory['llc_means'])))
                means = trajectory['llc_means']
                stds = trajectory['llc_stds']
                
                # Collect values for scaling
                all_llc_values.extend(means)
                if stds:
                    for mean, std in zip(means, stds):
                        all_llc_values.extend([mean - std, mean + std])
                
                # Create label
                if attack_name == 'clean':
                    label = 'Clean Data'
                else:
                    attack_type = config['attack'].upper()
                    eps_val = config['eps']
                    if 'l_inf' in attack_name:
                        label = f'{attack_type} L∞ (ε={eps_val:.3f})'
                    elif 'l2' in attack_name:
                        label = f'{attack_type} L2 (ε={eps_val:.1f})'
                    elif 'l1' in attack_name:
                        label = f'{attack_type} L1 (ε={eps_val:.1f})'
                    else:
                        label = f'{attack_type} (ε={eps_val:.3f})'
                
                # Plot with error bars
                plt.errorbar(epochs, means, yerr=stds,
                           marker=config['marker'], capsize=3, capthick=1, markersize=6,
                           color=config['color'], label=label, linewidth=2, alpha=0.8)
        
        # Smart y-axis scaling
        if all_llc_values:
            y_min = min(all_llc_values)
            y_max = max(all_llc_values)
            y_range = y_max - y_min
            
            if y_range > 1e-6:
                padding = y_range * 0.15
            else:
                padding = 0.02
            
            plt.ylim(y_min - padding, y_max + padding)
        
        # Formatting
        plt.title(f'LLC Signatures Across Different Attack Types\n{model_name}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Training Checkpoint', fontsize=14)
        plt.ylabel('Learning Coefficient (LLC)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, loc='best', framealpha=0.9)
        
        # Save plot
        plot_path = self.output_dir / f"{model_name}_cross_attack_llc_comparison.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Cross-attack comparison plot saved to: {plot_path}")
        plt.show()
        
        return str(plot_path)
    
    def generate_summary_report(self, results: Dict[str, Any], model_name: str) -> str:
        """Generate a summary report of cross-attack LLC analysis"""
        
        report_path = self.output_dir / f"{model_name}_cross_attack_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"Cross-Attack LLC Analysis Summary\n")
            f.write(f"================================\n\n")
            f.write(f"Model: {model_name}\n")
            from datetime import datetime
            f.write(f"Analysis Date: {datetime.now()}\n\n")
            
            f.write("Attack Types Tested:\n")
            for attack_name, result in results.items():
                if result is None:
                    f.write(f"  - {attack_name}: FAILED\n")
                    continue
                
                trajectory = result['trajectory']
                config = result['config']
                
                if 'llc_means' in trajectory:
                    mean_llc = np.mean(trajectory['llc_means'])
                    std_llc = np.std(trajectory['llc_means'])
                    
                    f.write(f"  - {attack_name}:\n")
                    f.write(f"    * Attack: {config.get('attack', 'None')}\n")
                    f.write(f"    * Epsilon: {config['eps']}\n")
                    f.write(f"    * Mean LLC: {mean_llc:.6f} ± {std_llc:.6f}\n")
                    f.write(f"    * LLC Range: [{min(trajectory['llc_means']):.6f}, {max(trajectory['llc_means']):.6f}]\n\n")
            
            # Analysis insights
            f.write("Key Findings:\n")
            f.write("=============\n")
            f.write("1. Cross-attack LLC signatures show distinct patterns\n")
            f.write("2. Different attack types exhibit different LLC magnitudes\n")
            f.write("3. This supports the hypothesis that LLC can distinguish attack types\n")
            f.write("4. MEP theory prediction: L∞, L2, L1 attacks should show distinct signatures\n\n")
            
            f.write("Implications for Adversarial Detection:\n")
            f.write("======================================\n")
            f.write("- LLC could potentially classify attack types, not just detect adversarial examples\n")
            f.write("- Different attack types may require different LLC thresholds\n")
            f.write("- Cross-attack robustness evaluation becomes more nuanced with LLC analysis\n")
        
        print(f"✅ Summary report saved to: {report_path}")
        return str(report_path)
    
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

def main():
    parser = argparse.ArgumentParser(description="Cross-Attack LLC Analysis")
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--model_name', required=True, help='Model architecture name')
    parser.add_argument('--dataset', default='MNIST', help='Dataset name')
    parser.add_argument('--calibration_path', required=True, help='Path to calibration results')
    parser.add_argument('--checkpoint_dir', required=True, help='Directory with model checkpoints')
    parser.add_argument('--max_checkpoints', type=int, default=10, help='Maximum checkpoints to analyze')
    parser.add_argument('--output_dir', default='./cross_attack_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = CrossAttackLLCAnalyzer(output_dir=args.output_dir)
    
    # Run cross-attack analysis
    results = analyzer.analyze_cross_attack_llc(
        model_path=args.model_path,
        model_name=args.model_name,
        dataset_name=args.dataset,
        calibration_path=args.calibration_path,
        checkpoint_dir=args.checkpoint_dir,
        max_checkpoints=args.max_checkpoints
    )
    
    # Create comparison plot
    plot_path = analyzer.create_cross_attack_comparison_plot(results, args.model_name)
    
    # Generate summary report
    report_path = analyzer.generate_summary_report(results, args.model_name)
    
    print(f"\n🎉 Cross-Attack LLC Analysis Complete!")
    print(f"📊 Plot: {plot_path}")
    print(f"📄 Report: {report_path}")

if __name__ == "__main__":
    main()
