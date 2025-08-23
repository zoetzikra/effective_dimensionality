#!/usr/bin/env python3
"""
Resume clean vs adversarial LLC analysis from a specific checkpoint.
This script helps you complete interrupted clean_vs_adv analysis.
"""

import subprocess
import sys
import os

def main():
    # Configuration for your specific case
    checkpoint_dir = "./models/ResNet18_AT/epoch_iter"
    model_name = "ResNet18"
    dataset = "CIFAR10"
    defense_method = "AT"
    max_checkpoints = 20
    resume_from_checkpoint = 10  # Resume from checkpoint 10 (you have 0-9 completed)
    output_dir = "./llc_analysis/llc_analysis_20250819_162146/ResNet18_AT_clean_vs_adv"  # Existing directory
    
    print("🔄 Resuming clean vs adversarial LLC analysis...")
    print(f"📂 Checkpoint directory: {checkpoint_dir}")
    print(f"📂 Output directory: {output_dir}")
    print(f"📊 Resume from checkpoint: {resume_from_checkpoint}")
    print(f"🎯 Max checkpoints: {max_checkpoints}")
    
    # Verify checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Error: Checkpoint directory not found: {checkpoint_dir}")
        return 1
    
    # Verify output directory exists
    if not os.path.exists(output_dir):
        print(f"❌ Error: Output directory not found: {output_dir}")
        return 1
    
    # Build command
    cmd = [
        "python", "llc_analysis_pipeline.py",
        "--mode", "clean_vs_adv",
        "--checkpoint_dir", checkpoint_dir,
        "--model_name", model_name,
        "--dataset", dataset,
        "--defense_method", defense_method,
        "--max_checkpoints", str(max_checkpoints),
        "--resume_from_checkpoint", str(resume_from_checkpoint),
        "--output_dir", output_dir
    ]
    
    print("🚀 Running command:")
    print(" ".join(cmd))
    print()
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Resume completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: Command failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        return 1

if __name__ == "__main__":
    sys.exit(main())
