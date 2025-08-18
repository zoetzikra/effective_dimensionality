#!/usr/bin/env python3
"""
Quick script to inspect checkpoint structure
"""
import torch
import sys

if len(sys.argv) != 2:
    print("Usage: python inspect_checkpoint.py <checkpoint_path>")
    sys.exit(1)

checkpoint_path = sys.argv[1]

try:
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"Checkpoint file: {checkpoint_path}")
    print(f"Top-level keys: {list(checkpoint.keys())}")
    print()
    
    # Inspect each top-level key
    for key, value in checkpoint.items():
        print(f"Key: '{key}'")
        print(f"  Type: {type(value)}")
        
        if isinstance(value, dict):
            print(f"  Dict keys ({len(value)}): {list(value.keys())[:10]}{'...' if len(value) > 10 else ''}")
        elif hasattr(value, 'state_dict'):
            print(f"  Has state_dict method")
            try:
                state_dict = value.state_dict()
                print(f"  State dict keys ({len(state_dict)}): {list(state_dict.keys())[:5]}{'...' if len(state_dict) > 5 else ''}")
            except Exception as e:
                print(f"  Error getting state_dict: {e}")
        elif hasattr(value, '__dict__'):
            print(f"  Object attributes: {list(value.__dict__.keys())[:10]}{'...' if len(value.__dict__) > 10 else ''}")
        else:
            print(f"  Value: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
        print()
        
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    import traceback
    traceback.print_exc()
