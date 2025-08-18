#!/usr/bin/env python3
"""
Debug script for LLC estimation issues
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys

# Add current directory to path
sys.path.append('.')

from llc_measurement import LLCConfig, LLCMeasurer
from AT_replication_complete import create_model_and_config

def test_single_checkpoint(checkpoint_path: str, model_name: str = "LeNet"):
    """Test LLC estimation on a single checkpoint"""
    
    print(f"Testing LLC estimation on {checkpoint_path}")
    print(f"Model: {model_name}")
    
    # Create model
    model, config = create_model_and_config(model_name)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        if 'rmodel_state_dict' in checkpoint:
            rmodel_state = checkpoint['rmodel_state_dict']
            base_model_state = {}
            for key, value in rmodel_state.items():
                if key.startswith('model.'):
                    base_key = key[6:]
                    base_model_state[base_key] = value
            model.load_state_dict(base_model_state)
            print("Loaded MAIR checkpoint")
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded standard checkpoint")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded state dict directly")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Set model to evaluation mode
    model.eval()
    print(f"Model device: {next(model.parameters()).device}")
    
    # Create data loader
    MEAN = [0.1307]
    STD = [0.3081]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    
    train_dataset = datasets.MNIST("./data", train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    
    print(f"Dataset size: {len(train_dataset)}")
    
    # Create LLC config with minimal settings
    llc_config = LLCConfig(
        model_name=model_name,
        num_chains=2,  # Very small for testing
        num_steps=500,  # Very small for testing
        batch_size=128,  # Smaller batch size
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"LLC config: {llc_config}")
    
    # Create measurer
    measurer = LLCMeasurer(llc_config)
    
    # Test basic model evaluation first
    print("\nTesting basic model evaluation...")
    try:
        batch = next(iter(train_loader))
        inputs, targets = batch
        print(f"Input shape: {inputs.shape}, target shape: {targets.shape}")
        
        # Test evaluation function
        loss, info = measurer.evaluate_model(model, (inputs, targets))
        print(f"Evaluation successful: loss = {loss.item():.4f}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test LLC estimation
    print("\nTesting LLC estimation...")
    try:
        results = measurer.estimate_llc(model, train_loader, run_diagnostics=False)
        print(f"LLC estimation successful!")
        print(f"Results keys: {list(results.keys())}")
        if 'llc/mean' in results:
            print(f"LLC mean: {results['llc/mean']:.4f}")
            print(f"LLC std: {results['llc/std']:.4f}")
        else:
            print("No llc/mean in results!")
            print(f"Full results: {results}")
            
    except Exception as e:
        print(f"LLC estimation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_llc.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    test_single_checkpoint(checkpoint_path) 