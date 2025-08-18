#!/usr/bin/env python3
"""
Effective Dimensionality Analysis Script

Converted from eigs.ipynb - runs the effective dimensionality analysis 
and generates plots for model complexity comparison.

Usage:
    python run_effective_dimensionality_analysis.py --model LeNet --defense AT
"""

import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import glob
import json
# Add project paths
mair_path = '/home/ztzifa/effective_dimensionality/MAIR'
project_path = '/home/ztzifa/effective_dimensionality'

if mair_path not in sys.path:
    sys.path.insert(0, mair_path)
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# Import required modules
import mair
from hess_vec_prod import min_max_hessian_eigs
from AT_replication_complete import create_model_and_config, setup_mnist_data, setup_cifar10_data

def eff_dim(x, s=1.):
    """Effective dimensionality calculation (from authors)"""
    x = x[x != 1.]  # Remove unconverged eigenvalues
    return np.sum(x / (x + s))

def analyze_checkpoints(model_name="LeNet", defense_method="AT", max_checkpoints=10, nsteps=50):
    """
    Analyze effective dimensionality across training checkpoints
    
    Args:
        model_name: Model architecture (LeNet, ResNet18, VGG11)
        defense_method: Defense method (AT, TRADES, Standard, etc.)
        max_checkpoints: Maximum number of checkpoints to analyze
        nsteps: Number of Lanczos steps for eigenvalue computation
    
    Returns:
        Dictionary with results
    """
    print(f"Analyzing {model_name} {defense_method}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model and get data loaders
    model, config = create_model_and_config(model_name)
    
    if config["dataset"] == "MNIST":
        train_loader, test_loader, mean, std = setup_mnist_data(batch_size=128)
    else:
        train_loader, test_loader, mean, std = setup_cifar10_data(batch_size=128)
    
    # Find checkpoint directory
    models_base_path = '/home/ztzifa/effective_dimensionality/models'
    checkpoint_dir = f"{models_base_path}/{model_name}_{defense_method}/epoch_iter"
    
    if not os.path.exists(checkpoint_dir):
        checkpoint_dir = f"{models_base_path}/{model_name}_{defense_method}"
    
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return {}
    
    # Get checkpoint files
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pth")))
    checkpoint_files = [f for f in checkpoint_files if not any(x in os.path.basename(f) 
                       for x in ['best', 'last', 'init'])]
    
    if len(checkpoint_files) > max_checkpoints:
        step = len(checkpoint_files) // max_checkpoints
        checkpoint_files = checkpoint_files[::step]
    
    print(f"Processing {len(checkpoint_files)} checkpoints")
    
    if len(checkpoint_files) == 0:
        print("No checkpoints found!")
        return {}
    
    # Analyze each checkpoint
    results = {}
    criterion = nn.CrossEntropyLoss()
    
    for i, checkpoint_path in enumerate(checkpoint_files):
        print(f"Processing checkpoint {i+1}/{len(checkpoint_files)}: {os.path.basename(checkpoint_path)}")
        
        try:
            # Create fresh model
            model, config = create_model_and_config(model_name)
            model = model.to(device)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Handle different checkpoint formats
            if 'rmodel_state_dict' in checkpoint:
                rmodel_state = checkpoint['rmodel_state_dict']
                base_model_state = {}
                for key, value in rmodel_state.items():
                    if key.startswith('model.'):
                        base_key = key[6:]
                        base_model_state[base_key] = value
                model.load_state_dict(base_model_state)
            elif 'rmodel' in checkpoint:
                rmodel_state = checkpoint['rmodel']
                base_model_state = {}
                for key, value in rmodel_state.items():
                    if key.startswith('model.'):
                        base_key = key[6:]
                        base_model_state[base_key] = value
                model.load_state_dict(base_model_state)
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            
            # Compute effective dimensionality using authors' function
            max_eval, min_eval, hvps, pos_evals, neg_evals, pos_bases = min_max_hessian_eigs(
                model, train_loader, criterion, use_cuda=(device.type == 'cuda'), 
                verbose=False, nsteps=nsteps
            )
            
            eigs = pos_evals.cpu().numpy()
            effective_dimension = eff_dim(eigs)
            
            results[i] = {
                'checkpoint': os.path.basename(checkpoint_path),
                'effective_dimensionality': float(effective_dimension),
                'max_eigenvalue': float(max_eval),
                'num_eigenvalues': int(len(eigs))
            }
            
            print(f"  Effective Dimensionality: {effective_dimension:.4f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    return results

def plot_results(results, model_name, defense_method, save_path=None):
    """
    Create plots similar to the original eigs.ipynb notebook
    
    Args:
        results: Results dictionary from analyze_checkpoints
        model_name: Model architecture name
        defense_method: Defense method name
        save_path: Optional path to save the plot
    """
    if not results:
        print("No results to plot!")
        return
    
    # Extract data
    checkpoints = list(results.keys())
    eff_dims = [results[i]['effective_dimensionality'] for i in checkpoints]
    max_eigs = [results[i]['max_eigenvalue'] for i in checkpoints]
    
    # Set up plotting style (similar to original notebook)
    plt.rcParams.update({'font.size': 12})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Effective Dimensionality Trajectory
    ax1.plot(checkpoints, eff_dims, 'o-', linewidth=2, markersize=6, color='blue', alpha=0.7)
    ax1.set_xlabel('Checkpoint Index')
    ax1.set_ylabel('Effective Dimensionality')
    ax1.set_title(f'{model_name} {defense_method} - Effective Dimensionality Evolution')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Max Eigenvalue Trajectory
    ax2.plot(checkpoints, max_eigs, 's-', linewidth=2, markersize=6, color='red', alpha=0.7)
    ax2.set_xlabel('Checkpoint Index')
    ax2.set_ylabel('Max Eigenvalue')
    ax2.set_title(f'{model_name} {defense_method} - Max Eigenvalue Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale for eigenvalues
    
    plt.tight_layout()
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Effective Dimensionality Range: {min(eff_dims):.4f} - {max(eff_dims):.4f}")
    print(f"  Mean Effective Dimensionality: {np.mean(eff_dims):.4f}")
    print(f"  Final Effective Dimensionality: {eff_dims[-1]:.4f}")
    print(f"  Max Eigenvalue Range: {min(max_eigs):.6f} - {max(max_eigs):.6f}")

def compare_multiple_methods(model_name="LeNet", defense_methods=["Standard", "AT", "TRADES"], 
                           max_checkpoints=10, nsteps=50):
    """
    Compare effective dimensionality across multiple defense methods
    
    Args:
        model_name: Model architecture
        defense_methods: List of defense methods to compare
        max_checkpoints: Maximum checkpoints per method
        nsteps: Lanczos steps
    """
    all_results = {}
    
    for defense_method in defense_methods:
        print(f"\n{'='*60}")
        results = analyze_checkpoints(model_name, defense_method, max_checkpoints, nsteps)
        if results:
            all_results[defense_method] = results
    
    if not all_results:
        print("No results to compare!")
        return
    
    # Create comparison plot
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (method, results) in enumerate(all_results.items()):
        checkpoints = list(results.keys())
        eff_dims = [results[j]['effective_dimensionality'] for j in checkpoints]
        
        ax.plot(checkpoints, eff_dims, 'o-', linewidth=2, markersize=4, 
                color=colors[i % len(colors)], alpha=0.7, label=method)
    
    ax.set_xlabel('Checkpoint Index')
    ax.set_ylabel('Effective Dimensionality')
    ax.set_title(f'{model_name} - Effective Dimensionality Comparison Across Defense Methods')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    # Save comparison plot
    save_path = f'/home/ztzifa/effective_dimensionality/{model_name}_defense_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_path}")
    plt.show()
    
    return all_results

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Effective Dimensionality Analysis")
    parser.add_argument("--model", type=str, default="LeNet", 
                       choices=["LeNet", "ResNet18", "VGG11"],
                       help="Model architecture")
    parser.add_argument("--defense", type=str, default="AT",
                       help="Defense method (AT, TRADES, Standard, etc.)")
    parser.add_argument("--max_checkpoints", type=int, default=10,
                       help="Maximum number of checkpoints to analyze")
    parser.add_argument("--nsteps", type=int, default=50,
                       help="Number of Lanczos steps")
    parser.add_argument("--compare", action="store_true",
                       help="Compare multiple defense methods")
    parser.add_argument("--save_plot", type=str,
                       help="Path to save the plot")
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple methods
        defense_methods = ["Standard", "AT", "TRADES"]
        if args.model == "LeNet":
            defense_methods = ["Standard", "AT", "TRADES"]  # Add more if available
        
        compare_multiple_methods(args.model, defense_methods, 
                               args.max_checkpoints, args.nsteps)
    else:
        # Analyze single method
        results = analyze_checkpoints(args.model, args.defense, 
                                    args.max_checkpoints, args.nsteps)
        
        if results:
            # Save results
            results_path = f'/home/ztzifa/effective_dimensionality/{args.model}_{args.defense}_effective_dim_trajectory.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {results_path}") 
            
            # Create plot
            plot_save_path = args.save_plot or f'/home/ztzifa/effective_dimensionality/{args.model}_{args.defense}_effective_dim_plot.png'
            plot_results(results, args.model, args.defense, plot_save_path)

if __name__ == "__main__":
    main()
