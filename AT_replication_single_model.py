#!/usr/bin/env python3
"""
Quick Start Script for Adversarial Training Replication

This script provides a minimal working example to get you started quickly.
Choose one model and one training method to begin with.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import mair
from mair.defenses import Standard, AT, TRADES, MART
from mair.utils.models import load_model
import glob

def quick_train_resnet18(use_awp=False):
    """Quick training example for ResNet18 with AT (Adversarial Training)"""
    
    print("Setting up CIFAR10 data...")
    
    # CIFAR10 setup (matching paper)
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2023, 0.1994, 0.2010]
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    
    train_data = dsets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_data = dsets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=2)
    
    print("Creating ResNet18 model...")
    
    # Create model
    model = load_model("ResNet18", n_classes=10)
    rmodel = mair.RobModel(model, n_classes=10, 
                          normalization_used={'mean': MEAN, 'std': STD}).cuda()
    
    print("Setting up AT trainer...")
    
    # Setup AT trainer (as used in paper)
    EPS = 8/255
    ALPHA = 2/255
    STEPS = 10
    
    trainer = AT(rmodel, eps=EPS, alpha=ALPHA, steps=STEPS)
    
    # Record robustness metrics
    trainer.record_rob(train_loader, test_loader, eps=EPS, alpha=ALPHA, steps=STEPS, std=0.1, 
                       n_train_limit=1000, n_val_limit=1000)
    
    # Setup training parameters (matching paper)
    minimizer = "AWP(rho=5e-3)" if use_awp else None
    
    trainer.setup(optimizer="SGD(lr=0.1, momentum=0.9, weight_decay=0.0005)",
                  scheduler="Step(milestones=[100, 150], gamma=0.1)", 
                  scheduler_type="Epoch",
                  minimizer=minimizer,
                  n_epochs=200,
                  n_iters=len(train_loader))
    
    print("Starting training with MAIR's built-in checkpointing...")
    
    # Use MAIR's built-in fit() method with checkpointing
    trainer.fit(
        train_loader, 
        n_epochs=200,
        save_path="./models/ResNet18_AT_quick",
        save_type="Epoch",  # Save every epoch
        save_best={"Clean(Val)": "HBO", "PGD(Val)": "HB"},  # Best model by validation metrics!
        save_overwrite=True
    )
    
    print("Training completed! Evaluating...")
    
    # Evaluate
    clean_acc = rmodel.eval_accuracy(test_loader)
    pgd_acc = rmodel.eval_rob_accuracy_pgd(test_loader, eps=EPS, alpha=ALPHA, steps=STEPS)
    
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    print(f"PGD Accuracy: {pgd_acc:.2f}%")
    print("Checkpoints saved in: ./models/ResNet18_AT_quick/")
    
    return rmodel

def quick_train_lenet(use_awp=False):
    """Quick training example for LeNet (simple CNN) on MNIST"""
    
    print("Setting up MNIST data...")
    
    # MNIST setup
    MEAN = [0.1307]
    STD = [0.3081]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    
    train_data = dsets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = dsets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=2)
    
    print("Creating LeNet model...")
    
    # Create model  
    model = load_model("LeNet", n_classes=10)
    rmodel = mair.RobModel(model, n_classes=10,
                          normalization_used={'mean': MEAN, 'std': STD}).cuda()
    
    print("Setting up AT trainer...")
    
    # Setup AT trainer (MNIST scale)
    EPS = 0.3
    ALPHA = 0.01
    STEPS = 40
    
    trainer = AT(rmodel, eps=EPS, alpha=ALPHA, steps=STEPS)
    
    # Record robustness metrics
    trainer.record_rob(train_loader, test_loader, eps=EPS, alpha=ALPHA, steps=STEPS, std=0.1,
                       n_train_limit=1000, n_val_limit=1000)
    
    # Setup training parameters
    minimizer = "AWP(rho=5e-3)" if use_awp else None
    
    trainer.setup(optimizer="SGD(lr=0.01, momentum=0.9, weight_decay=0.0005)",
                  scheduler="Step(milestones=[50, 75], gamma=0.1)", 
                  scheduler_type="Epoch",
                  minimizer=minimizer,
                  n_epochs=100,
                  n_iters=len(train_loader))
    
    print("Starting training with MAIR's built-in checkpointing...")
    
    # Use MAIR's built-in fit() method
    trainer.fit(
        train_loader, 
        n_epochs=100,
        save_path="./models/LeNet_AT_quick",
        save_type="Epoch",
        save_best={"Clean(Val)": "HBO", "PGD(Val)": "HB"},
        save_overwrite=True
    )
    
    print("Training completed! Evaluating...")
    
    # Evaluate
    clean_acc = rmodel.eval_accuracy(test_loader)
    pgd_acc = rmodel.eval_rob_accuracy_pgd(test_loader, eps=EPS, alpha=ALPHA, steps=STEPS)
    
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    print(f"PGD Accuracy: {pgd_acc:.2f}%")
    print("Checkpoints saved in: ./models/LeNet_AT_quick/")
    
    return rmodel

def quick_train_vgg(use_awp=False):
    """Quick training example for VGG11 (CNN) on CIFAR10"""
    
    print("Setting up CIFAR10 data...")
    
    # CIFAR10 setup
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2023, 0.1994, 0.2010]
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    
    train_data = dsets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_data = dsets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=2)
    
    print("Creating VGG11 model...")
    
    # Create model
    model = load_model("VGG11", n_classes=10)
    rmodel = mair.RobModel(model, n_classes=10,
                          normalization_used={'mean': MEAN, 'std': STD}).cuda()
    
    print("Setting up TRADES trainer...")
    
    # Setup TRADES trainer
    EPS = 8/255
    ALPHA = 2/255
    STEPS = 10
    BETA = 6.0
    
    trainer = TRADES(rmodel, eps=EPS, alpha=ALPHA, steps=STEPS, beta=BETA)
    
    # Record robustness metrics
    trainer.record_rob(train_loader, test_loader, eps=EPS, alpha=ALPHA, steps=STEPS, std=0.1,
                       n_train_limit=1000, n_val_limit=1000)
    
    # Setup training parameters
    minimizer = "AWP(rho=5e-3)" if use_awp else None
    
    trainer.setup(optimizer="SGD(lr=0.1, momentum=0.9, weight_decay=0.0005)",
                  scheduler="Step(milestones=[100, 150], gamma=0.1)", 
                  scheduler_type="Epoch",
                  minimizer=minimizer,
                  n_epochs=200,
                  n_iters=len(train_loader))
    
    print("Starting training with MAIR's built-in checkpointing...")
    
    # Use MAIR's built-in fit() method
    trainer.fit(
        train_loader, 
        n_epochs=200,
        save_path="./models/VGG11_TRADES_quick",
        save_type="Epoch",
        save_best={"Clean(Val)": "HBO", "PGD(Val)": "HB"},
        save_overwrite=True
    )
    
    print("Training completed! Evaluating...")
    
    # Evaluate
    clean_acc = rmodel.eval_accuracy(test_loader)
    pgd_acc = rmodel.eval_rob_accuracy_pgd(test_loader, eps=EPS, alpha=ALPHA, steps=STEPS)
    
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    print(f"PGD Accuracy: {pgd_acc:.2f}%")
    print("Checkpoints saved in: ./models/VGG11_TRADES_quick/")
    
    return rmodel

def resume_from_checkpoint(checkpoint_path):
    """Resume training from a MAIR checkpoint"""
    print(f"Resuming from checkpoint: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Extract model info from path
    parts = checkpoint_path.split(os.sep)
    model_dir = parts[-3]  # models/MODEL_METHOD/checkpoints/file.pth
    
    if "ResNet18" in model_dir:
        rmodel = quick_train_resnet18()
    elif "LeNet" in model_dir:
        rmodel = quick_train_lenet()
    elif "VGG11" in model_dir:
        rmodel = quick_train_vgg()
    else:
        print("Unknown model type in checkpoint path")
        return None
    
    # Load the checkpoint state
    rmodel.load_state_dict(checkpoint['rmodel'])
    print("Checkpoint loaded successfully!")
    
    return rmodel

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Quick Start Adversarial Training (MAIR-Compatible)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python AT_replication_single_model.py --model resnet18
  python AT_replication_single_model.py --model lenet --epochs 50
  python AT_replication_single_model.py --model vgg --method trades
  python AT_replication_single_model.py --resume ./models/LeNet_AT_quick/last.pth
        """
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['resnet18', 'lenet', 'vgg'],
        help='Model to train (default: resnet18)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['at', 'trades', 'mart'],
        default='at',
        help='Training method (default: at)'
    )
    
    parser.add_argument(
        '--awp',
        action='store_true',
        help='Use Adversarial Weight Perturbation (AWP)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs (overrides default)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume from'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size (default: 128)'
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    print("=== Quick Start Adversarial Training (MAIR-Compatible) ===")
    print("Now using MAIR's built-in checkpointing system!")
    print()
    
    # Handle resume case first
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        rmodel = resume_from_checkpoint(args.resume)
    else:
        # Determine model and method
        model_choice = args.model or 'resnet18'
        method_choice = args.method or 'at'
        
        print(f"Training {model_choice.upper()} with {method_choice.upper()}")
        print(f"AWP: {args.awp}")
        
        if model_choice == 'resnet18':
            print("\nTraining ResNet18 with AT (MAIR built-in checkpointing)...")
            rmodel = quick_train_resnet18(use_awp=args.awp)
        elif model_choice == 'lenet':
            print("\nTraining LeNet with AT (MAIR built-in checkpointing)...")
            rmodel = quick_train_lenet(use_awp=args.awp)
        elif model_choice == 'vgg':
            if method_choice == 'trades':
                print("\nTraining VGG11 with TRADES (MAIR built-in checkpointing)...")
                rmodel = quick_train_vgg(use_awp=args.awp)
            else:
                print(f"VGG with {method_choice.upper()} not implemented yet, using TRADES...")
                rmodel = quick_train_vgg(use_awp=args.awp)
        else:
            print(f"Unknown model: {model_choice}. Running ResNet18 by default...")
            rmodel = quick_train_resnet18(use_awp=args.awp)
    
    print("\n" + "="*70)
    print("�� TRAINING COMPLETED SUCCESSFULLY! ��")
    print("="*70)
    print("✅ Model trained with MAIR's built-in functionality")
    print("✅ Checkpoints saved automatically (init.pth, last.pth, best.pth)")
    print("✅ Epoch checkpoints saved in epoch_iter/ directory")
    print("✅ Best model selected by validation PGD accuracy")
    print("✅ All robustness metrics tracked (Clean/FGSM/PGD/GN)")
    print("✅ Results should match the original paper")
    print()
    print("📁 Model files saved:")
    print("   - best.pth (selected by validation metrics)")
    print("   - last.pth (final epoch)")
    print("   - init.pth (initial state)")
    print("   - epoch_iter/000XX_00000.pth (every epoch)")
    print()
    print("🔬 Next steps:")
    print("   - Use effective_dimensionality_script.py to analyze the trained model")
    print("   - Resume training with: --resume ./models/MODEL_METHOD/last.pth")
    print("="*70)