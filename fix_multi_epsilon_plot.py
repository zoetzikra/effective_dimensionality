#!/usr/bin/env python3
"""
Standalone script to regenerate the multi-epsilon LLC evolution plot with better scaling
Uses existing trajectory data files to avoid recomputation.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any

def load_trajectory_data(file_path: Path) -> Dict[str, Any]:
    """Load trajectory data from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def extract_trajectory_data(trajectory_data: Dict[str, Any]) -> Tuple[List[int], List[float], List[float]]:
    """Extract epochs, means, and stds from trajectory data"""
    if not trajectory_data:
        return [], [], []
    
    try:
        # Handle the specific format from your multi-epsilon analysis
        if 'llc_means' in trajectory_data and 'llc_stds' in trajectory_data:
            means = trajectory_data['llc_means']
            stds = trajectory_data['llc_stds']
            # Create checkpoint indices (0, 1, 2, ...)
            epochs = list(range(len(means)))
            return epochs, means, stds
        
        # Handle other possible JSON structures
        elif 'trajectory_data' in trajectory_data:
            data = trajectory_data['trajectory_data']
        elif 'results' in trajectory_data:
            data = trajectory_data['results']
        else:
            data = trajectory_data
        
        epochs = []
        means = []
        stds = []
        
        # Extract data points
        for checkpoint_idx, result in enumerate(data):
            if isinstance(result, dict):
                if 'epoch' in result:
                    epochs.append(result['epoch'])
                else:
                    epochs.append(checkpoint_idx)
                
                if 'llc_mean' in result:
                    means.append(result['llc_mean'])
                elif 'mean' in result:
                    means.append(result['mean'])
                else:
                    continue
                    
                if 'llc_std' in result:
                    stds.append(result['llc_std'])
                elif 'std' in result:
                    stds.append(result['std'])
                else:
                    stds.append(0.0)
        
        return epochs, means, stds
    
    except Exception as e:
        print(f"Error extracting trajectory data: {e}")
        return [], [], []

def create_improved_multi_epsilon_plot(data_dir: Path, output_path: Path, model_name: str = "LeNet") -> None:
    """Create improved multi-epsilon plot with better scaling"""
    
    # Define epsilon values and their corresponding file patterns
    epsilon_values = [0.0, 2/255, 4/255, 8/255, 12/255, 16/255]
    epsilon_files = [
        "llc_trajectory_eps_0_00.json",
        "llc_trajectory_eps_0_01.json", 
        "llc_trajectory_eps_0_02.json",
        "llc_trajectory_eps_0_03.json",
        "llc_trajectory_eps_0_05.json",
        "llc_trajectory_eps_0_06.json"
    ]
    
    # Color scheme for different epsilon values
    colors = ['#8B4513', '#FF6B35', '#F7931E', '#32CD32', '#4169E1', '#9932CC']
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Collect all LLC values for smart y-axis scaling
    all_llc_values = []
    
    # Load and plot each epsilon trajectory
    trajectories = {}
    for eps, filename, color in zip(epsilon_values, epsilon_files, colors):
        file_path = data_dir / filename
        
        if file_path.exists():
            trajectory_data = load_trajectory_data(file_path)
            epochs, means, stds = extract_trajectory_data(trajectory_data)
            
            if epochs and means:
                trajectories[eps] = (epochs, means, stds, color)
                
                # Collect values for scaling
                all_llc_values.extend(means)
                if stds:
                    for mean, std in zip(means, stds):
                        all_llc_values.extend([mean - std, mean + std])
                
                # Plot with error bars
                eps_label = f"ε={eps:.3f}" if eps > 0 else "ε=0.0"
                if eps > 0:
                    eps_frac = f"({int(eps*255)}/255)"
                    eps_label += f" {eps_frac}"
                
                plt.errorbar(epochs, means, yerr=stds, 
                           marker='o', capsize=3, capthick=1, markersize=4,
                           color=color, label=eps_label, linewidth=2, alpha=0.8)
        else:
            print(f"Warning: File not found: {file_path}")
    
    # Improved y-axis scaling
    if all_llc_values:
        y_min = min(all_llc_values)
        y_max = max(all_llc_values)
        y_range = y_max - y_min
        
        # Use smart padding
        if y_range > 1e-6:
            padding = y_range * 0.15  # 15% padding for better visibility
        else:
            padding = 0.02  # Fixed padding for very flat trajectories
        
        plt.ylim(y_min - padding, y_max + padding)
        
        print(f"Y-axis range: {y_min - padding:.4f} to {y_max + padding:.4f}")
        print(f"Data range: {y_min:.4f} to {y_max:.4f} (span: {y_range:.4f})")
    
    # Formatting
    plt.title(f'LLC Evolution During Adversarial Training\n{model_name}', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Training Checkpoint', fontsize=14)
    plt.ylabel('Learning Coefficient (LLC)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='upper right', framealpha=0.9)
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Improved multi-epsilon plot saved to: {output_path}")
    plt.show()

def main():
    # Set paths
    data_dir = Path("/home/ztzifa/effective_dimensionality/epsilon_analysis")
    output_path = data_dir / "LeNet_multi_epsilon_llc_evolution_improved.png"
    
    print(f"Looking for data files in: {data_dir}")
    print(f"Output will be saved to: {output_path}")
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    # List available files
    json_files = list(data_dir.glob("llc_trajectory_eps_*.json"))
    print(f"Found {len(json_files)} trajectory files:")
    for f in sorted(json_files):
        print(f"  - {f.name}")
    
    # Create improved plot
    create_improved_multi_epsilon_plot(data_dir, output_path, "LeNet")

if __name__ == "__main__":
    main()
