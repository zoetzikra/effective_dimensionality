#!/usr/bin/env python3
"""
Lightweight script to create cross-attack comparison plot from existing trajectory files.
This avoids memory issues by only loading the trajectory data and creating the plot.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Any
import glob

class CrossAttackPlotter:
    """Create comparison plots from existing cross-attack LLC trajectory files"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        
        # Attack configurations for plotting
        self.attack_configs = {
            'clean': {'color': '#8B4513', 'marker': 'o', 'label': 'Clean'},
            'pgd_l_inf': {'color': '#FF6B35', 'marker': 's', 'label': 'PGD L∞ (ε=8/255)'},
            'fgsm_l_inf': {'color': '#F7931E', 'marker': '^', 'label': 'FGSM L∞ (ε=8/255)'},
            'pgd_l2': {'color': '#32CD32', 'marker': 'D', 'label': 'PGD L2 (ε=0.5)'},
            'pgd_l1': {'color': '#4169E1', 'marker': 'v', 'label': 'PGD L1 (ε=10.0)'},
        }
    
    def find_trajectory_files(self) -> Dict[str, str]:
        """Find all trajectory JSON files in the results directory"""
        trajectory_files = {}
        
        # Look for files matching pattern: {attack_name}_llc_trajectory.json
        for attack_name in self.attack_configs.keys():
            pattern = f"{attack_name}_llc_trajectory.json"
            file_path = self.results_dir / pattern
            if file_path.exists():
                trajectory_files[attack_name] = str(file_path)
                print(f"✅ Found {attack_name} trajectory: {file_path}")
            else:
                print(f"⚠️  Missing {attack_name} trajectory: {file_path}")
        
        return trajectory_files
    
    def load_trajectory_data(self, file_path: str) -> Tuple[List[int], List[float], List[float]]:
        """Load trajectory data from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if 'llc_means' in data and 'llc_stds' in data:
                llc_means = data['llc_means']
                llc_stds = data['llc_stds']
                epochs = list(range(len(llc_means)))
                return epochs, llc_means, llc_stds
            else:
                print(f"❌ Unexpected data format in {file_path}")
                return [], [], []
                
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return [], [], []
    
    def create_comparison_plot(self, model_name: str = "Model") -> str:
        """Create the cross-attack comparison plot"""
        trajectory_files = self.find_trajectory_files()
        
        if not trajectory_files:
            raise ValueError("No trajectory files found!")
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 9))
        
        all_llc_values = []
        
        for attack_name, file_path in trajectory_files.items():
            if attack_name not in self.attack_configs:
                continue
                
            epochs, llc_means, llc_stds = self.load_trajectory_data(file_path)
            
            if epochs and llc_means:
                config = self.attack_configs[attack_name]
                
                # Plot the main line
                ax.plot(epochs, llc_means, 
                       color=config['color'], 
                       linewidth=2.5, 
                       marker=config['marker'], 
                       markersize=7, 
                       label=config['label'],
                       alpha=0.9)
                
                # Add error bars if available
                if llc_stds and len(llc_stds) == len(llc_means):
                    llc_means_np = np.array(llc_means)
                    llc_stds_np = np.array(llc_stds)
                    ax.fill_between(epochs, 
                                   llc_means_np - llc_stds_np, 
                                   llc_means_np + llc_stds_np,
                                   color=config['color'], 
                                   alpha=0.15)
                    
                    # Collect all values for y-axis scaling
                    all_llc_values.extend(llc_means)
                    all_llc_values.extend((llc_means_np - llc_stds_np).tolist())
                    all_llc_values.extend((llc_means_np + llc_stds_np).tolist())
                else:
                    all_llc_values.extend(llc_means)
        
        # Styling
        ax.set_xlabel('Training Checkpoint', fontsize=14, fontweight='bold')
        ax.set_ylabel('Learning Coefficient (LLC)', fontsize=14, fontweight='bold')
        ax.set_title(f'LLC Evolution Across Different Attack Types\n{model_name}', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Grid and legend
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.legend(fontsize=12, loc='best', framealpha=0.9, fancybox=True, shadow=True)
        
        # Auto-scale y-axis with padding
        if all_llc_values:
            y_min = min(all_llc_values)
            y_max = max(all_llc_values)
            y_range = y_max - y_min
            padding = y_range * 0.1 if y_range > 1e-6 else 0.01
            ax.set_ylim(y_min - padding, y_max + padding)
        
        # Improve appearance
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()
        
        # Save plot
        save_path = self.results_dir / f"{model_name}_cross_attack_llc_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"✅ Cross-attack comparison plot saved to: {save_path}")
        return str(save_path)
    
    def generate_summary_report(self, model_name: str = "Model") -> str:
        """Generate a summary report of final LLC values"""
        trajectory_files = self.find_trajectory_files()
        
        report_path = self.results_dir / f"{model_name}_cross_attack_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"Cross-Attack LLC Analysis Summary\n")
            f.write(f"================================\n\n")
            f.write(f"Model: {model_name}\n")
            from datetime import datetime
            f.write(f"Analysis Date: {datetime.now()}\n\n")
            
            f.write("Final LLC Values (Last Checkpoint):\n")
            f.write("-" * 50 + "\n")
            
            for attack_name, file_path in trajectory_files.items():
                if attack_name not in self.attack_configs:
                    continue
                    
                epochs, llc_means, llc_stds = self.load_trajectory_data(file_path)
                
                if llc_means:
                    final_mean = llc_means[-1]
                    final_std = llc_stds[-1] if llc_stds else 0.0
                    label = self.attack_configs[attack_name]['label']
                    f.write(f"{label:25}: {final_mean:.4f} ± {final_std:.4f}\n")
                else:
                    f.write(f"{attack_name:25}: FAILED\n")
            
            f.write("\nObservations:\n")
            f.write("-" * 20 + "\n")
            f.write("• Compare the final LLC values to see how different attack types\n")
            f.write("  affect the model's learning complexity.\n")
            f.write("• Lower LLC typically indicates simpler/more generalizable representations.\n")
            f.write("• Higher LLC may indicate more complex/overfitted representations.\n")
        
        print(f"✅ Summary report saved to: {report_path}")
        return str(report_path)


def main():
    parser = argparse.ArgumentParser(description="Create cross-attack LLC comparison plot from existing trajectory files")
    parser.add_argument('--results_dir', type=str, required=True, 
                       help='Directory containing the LLC trajectory JSON files')
    parser.add_argument('--model_name', type=str, default='LeNet', 
                       help='Model name for plot title')
    
    args = parser.parse_args()
    
    try:
        plotter = CrossAttackPlotter(args.results_dir)
        
        # Create comparison plot
        plot_path = plotter.create_comparison_plot(args.model_name)
        
        # Generate summary report
        report_path = plotter.generate_summary_report(args.model_name)
        
        print(f"\n🎉 Cross-attack analysis visualization complete!")
        print(f"📊 Plot: {plot_path}")
        print(f"📝 Report: {report_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
