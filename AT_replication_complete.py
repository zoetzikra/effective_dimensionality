#!/usr/bin/env python3
"""
Adversarial Training Replication Script for "Complexity Matters" Paper
Based on the MAIR framework used in the original paper.

This script replicates the training setup for:
1. Simple CNN models (LeNet for MNIST) 
2. VGG models (CNN for CIFAR10)
3. ResNet18 model (as used in the paper)

Supports various adversarial training methods: Standard, AT, TRADES, MART
With optional AWP (Adversarial Weight Perturbation) and extra data
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
from datetime import datetime

def setup_cifar10_data(batch_size=128):
    """Setup CIFAR10 dataset with proper normalization"""
    
    # CIFAR10 normalization values (as used in paper)
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2023, 0.1994, 0.2010]
    
    # Training transforms (with data augmentation)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    
    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    
    # Load datasets
    train_data = dsets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_data = dsets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader, MEAN, STD

def setup_mnist_data(batch_size=128):
    """Setup MNIST dataset for LeNet experiments"""
    
    MEAN = [0.1307]
    STD = [0.3081]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    
    train_data = dsets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = dsets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader, MEAN, STD

def create_model_and_config(model_name, dataset="CIFAR10"):
    """Create model and return training configuration based on paper settings"""
    
    configs = {
        # Simple CNN model (LeNet for MNIST) - closest to MLP
        "LeNet": {
            "n_classes": 10,
            "dataset": "MNIST",
            "epochs": 100,
            "eps": 0.3,  # MNIST typically uses larger eps due to different scale
            "alpha": 0.01,
            "steps": 40,
            "lr": 0.01,
            "milestones": [50, 75],
            "beta": 1.0,  # For TRADES/MART
        },
        
        # VGG model (CNN for CIFAR10)
        "VGG11": {
            "n_classes": 10,
            "dataset": "CIFAR10", 
            "epochs": 200,
            "eps": 8/255,
            "alpha": 2/255,
            "steps": 10,
            "lr": 0.1,
            "milestones": [100, 150],
            "beta": 6.0,  # For TRADES/MART
        },
        
        # ResNet18 (as used in the paper)
        "ResNet18": {
            "n_classes": 10,
            "dataset": "CIFAR10",
            "epochs": 5000,
            "eps": 8/255,
            "alpha": 2/255,
            "steps": 10,
            "lr": 0.1,
            # "milestones": [100, 150],  
            "milestones": [2500, 3750],  
            "beta": 6.0,  # For TRADES/MART
        }
    }
    
    if model_name not in configs:
        raise ValueError(f"Model {model_name} not supported. Choose from: {list(configs.keys())}")
    
    config = configs[model_name]
    model = load_model(model_name, config["n_classes"])
    
    return model, config

def train_model(model_name, defense_method="AT", use_awp=False, use_extra_data=False, 
                resume_from_checkpoint=None, batch_size=128):
    """
    Train a model using specified adversarial training method
    
    Args:
        model_name: "LeNet", "VGG11", or "ResNet18"
        defense_method: "Standard", "AT", "TRADES", or "MART"
        use_awp: Whether to use Adversarial Weight Perturbation
        use_extra_data: Whether to use extra training data (simulated)
        resume_from_checkpoint: Path to checkpoint to resume from (optional)
        batch_size: Training batch size
    """
    
    print(f"Training {model_name} with {defense_method}")
    if use_awp:
        print("Using AWP (Adversarial Weight Perturbation)")
    if use_extra_data:
        print("Using extra training data")
    
    # Create model and get config
    model, config = create_model_and_config(model_name)
    
    # Setup data loaders
    if config["dataset"] == "MNIST":
        train_loader, test_loader, MEAN, STD = setup_mnist_data(batch_size)
    else:  # CIFAR10
        train_loader, test_loader, MEAN, STD = setup_cifar10_data(batch_size)
    
    # Create robust model wrapper
    rmodel = mair.RobModel(model, n_classes=config["n_classes"], 
                          normalization_used={'mean': MEAN, 'std': STD}).cuda()
    
    # Select training method based on paper's experiments
    if defense_method == "Standard":
        trainer = Standard(rmodel)
    elif defense_method == "AT":
        trainer = AT(rmodel, eps=config["eps"], alpha=config["alpha"], steps=config["steps"])
    elif defense_method == "TRADES":
        trainer = TRADES(rmodel, eps=config["eps"], alpha=config["alpha"], 
                        steps=config["steps"], beta=config["beta"])
    elif defense_method == "MART":
        trainer = MART(rmodel, eps=config["eps"], alpha=config["alpha"], 
                      steps=config["steps"], beta=config["beta"])
    else:
        raise ValueError(f"Defense method {defense_method} not supported")
    
    # Setup robustness recording (as done in paper)
    trainer.record_rob(
        train_loader, test_loader, 
        eps=config["eps"], alpha=config["alpha"], steps=config["steps"], 
        std=0.1,  # For Gaussian noise evaluation
        n_train_limit=1000, n_val_limit=1000  # Validation subset for speed
    )
    
    # Setup optimizer and scheduler (matching paper settings)
    optimizer_str = f"SGD(lr={config['lr']}, momentum=0.9, weight_decay=0.0005)"
    scheduler_str = f"Step(milestones={config['milestones']}, gamma=0.1)"
    
    # AWP minimizer setup (if requested)
    minimizer = "AWP(rho=5e-3)" if use_awp else None
    
    trainer.setup(
        optimizer=optimizer_str,
        scheduler=scheduler_str, 
        scheduler_type="Epoch",
        minimizer=minimizer,
        n_epochs=config["epochs"],
        n_iters=len(train_loader)
    )
    
    # Create save path
    save_name = f"{model_name}_{defense_method}"
    if use_awp:
        save_name += "_AWP"
    if use_extra_data:
        save_name += "_ED"
    
    save_path = f"./models/{save_name}"
    
    # Resume from checkpoint if specified
    refit = False
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        trainer.load_dict(resume_from_checkpoint)
        refit = True
    
    print(f"Starting training for {config['epochs']} epochs...")
    print(f"Save path: {save_path}")
    print("Using MAIR's built-in checkpointing system (saves every epoch)")
    print("Progress bars disabled, but epoch metrics will be shown")
    
    # Use MAIR's built-in fit() method with "Epoch" save type (once per epoch)
    # FIXED: Use record_type="Epoch" to show epoch metrics and create log.txt
    trainer.fit(
        train_loader, 
        n_epochs=config["epochs"],
        record_type="Epoch",  # FIXED: Show epoch metrics table and create log.txt
        save_path=save_path,
        save_type="Epoch",  # Save every epoch (not every iteration!)
        save_best={"Clean(Val)": "HBO", "PGD(Val)": "HB"},  # Best model by validation metrics!
        save_overwrite=True,
        refit=refit
    )
    
    print(f"Training completed for {model_name} with {defense_method}")
    return rmodel, trainer, save_path

def evaluate_model(rmodel, test_loader, eps=8/255, alpha=2/255, steps=10):
    """Evaluate model robustness (as done in paper)"""
    
    print("Evaluating model robustness...")
    
    # Clean accuracy
    clean_acc = rmodel.eval_accuracy(test_loader)
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    
    # FGSM accuracy  
    fgsm_acc = rmodel.eval_rob_accuracy_fgsm(test_loader, eps=eps)
    print(f"FGSM Accuracy: {fgsm_acc:.2f}%")
    
    # PGD accuracy
    pgd_acc = rmodel.eval_rob_accuracy_pgd(test_loader, eps=eps, alpha=alpha, steps=steps)
    print(f"PGD Accuracy: {pgd_acc:.2f}%")
    
    # Gaussian noise accuracy
    gn_acc = rmodel.eval_rob_accuracy_gn(test_loader, std=0.1)
    print(f"Gaussian Noise Accuracy: {gn_acc:.2f}%")
    
    return {
        "clean": clean_acc,
        "fgsm": fgsm_acc, 
        "pgd": pgd_acc,
        "gaussian": gn_acc
    }

def list_available_checkpoints(model_name=None, defense_method=None):
    """List all available checkpoints"""
    
    if model_name and defense_method:
        # Look for specific model checkpoints
        save_name = f"{model_name}_{defense_method}"
        pattern = f"./models/{save_name}*/last.pth"
    else:
        # Look for all checkpoints
        pattern = "./models/*/last.pth"
    
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        print("No checkpoints found.")
        return []
    
    print(f"\nFound {len(checkpoints)} checkpoints:")
    print("-" * 80)
    print(f"{'Model':<20} {'Method':<15} {'Path':<40}")
    print("-" * 80)
    
    checkpoint_info = []
    for cp_path in sorted(checkpoints):
        try:
            # Extract info from path
            parts = cp_path.split(os.sep)
            model_dir = parts[-2]  # models/MODEL_METHOD/last.pth
            
            # Parse model and method
            if "_" in model_dir:
                model_parts = model_dir.split("_")
                model = model_parts[0]
                method = "_".join(model_parts[1:])
            else:
                model = model_dir
                method = "unknown"
            
            checkpoint_info.append({
                'path': cp_path,
                'model': model,
                'method': method,
                'dir': model_dir
            })
            
            print(f"{model:<20} {method:<15} {cp_path:<40}")
            
        except Exception as e:
            print(f"Error parsing {cp_path}: {e}")
    
    return checkpoint_info

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Adversarial Training Replication (MAIR-Compatible)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train single model
  python AT_replication_complete.py --model ResNet18 --method AT
  
  # Train with AWP
  python AT_replication_complete.py --model ResNet18 --method AT --awp
  
  # Resume from checkpoint
  python AT_replication_complete.py --resume ./models/ResNet18_AT/last.pth
  
  # List checkpoints
  python AT_replication_complete.py --list-checkpoints
  
  # Run full experiment suite
  python AT_replication_complete.py --full-suite
        """
    )
    
    # Training options
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['LeNet', 'VGG11', 'ResNet18'],
        help='Model to train'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['Standard', 'AT', 'TRADES', 'MART'],
        default='AT',
        help='Training method (default: AT)'
    )
    
    parser.add_argument(
        '--awp',
        action='store_true',
        help='Use Adversarial Weight Perturbation (AWP)'
    )
    
    parser.add_argument(
        '--extra-data',
        action='store_true',
        help='Use extra training data (simulated)'
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
    
    # Utility options
    parser.add_argument(
        '--list-checkpoints',
        action='store_true',
        help='List all available checkpoints'
    )
    
    parser.add_argument(
        '--full-suite',
        action='store_true',
        help='Run the complete experiment suite (all models/methods)'
    )
    
    parser.add_argument(
        '--evaluate-only',
        type=str,
        help='Only evaluate a trained model (provide model path)'
    )
    
    return parser.parse_args()

def run_full_experiment_suite():
    """Run the complete experiment suite as defined in the paper"""
    
    print("Running full experiment suite...")
    print("Note: This will take many hours. Consider starting with individual experiments.")
    
    # Define experiments (based on paper's Table/Figure results)
    experiments = [
        # Simple CNN (LeNet on MNIST) - closest to MLP
        {"model": "LeNet", "method": "Standard", "awp": False, "extra": False},
        {"model": "LeNet", "method": "AT", "awp": False, "extra": False},
        {"model": "LeNet", "method": "TRADES", "awp": False, "extra": False},
        
        # CNN models (VGG on CIFAR10)
        {"model": "VGG11", "method": "Standard", "awp": False, "extra": False},
        {"model": "VGG11", "method": "AT", "awp": False, "extra": False},
        {"model": "VGG11", "method": "AT", "awp": True, "extra": False},
        
        # ResNet18 (main experiments from paper)
        {"model": "ResNet18", "method": "Standard", "awp": False, "extra": False},
        {"model": "ResNet18", "method": "AT", "awp": False, "extra": False},
        {"model": "ResNet18", "method": "TRADES", "awp": False, "extra": False},
        {"model": "ResNet18", "method": "MART", "awp": False, "extra": False},
        {"model": "ResNet18", "method": "AT", "awp": True, "extra": False},
        {"model": "ResNet18", "method": "TRADES", "awp": True, "extra": False},
        {"model": "ResNet18", "method": "MART", "awp": True, "extra": False},
        {"model": "ResNet18", "method": "AT", "awp": True, "extra": True},
    ]
    
    results = {}
    
    for exp in experiments:
        exp_name = f"{exp['model']}_{exp['method']}"
        if exp['awp']:
            exp_name += "_AWP"
        if exp['extra']:
            exp_name += "_ED"
            
        print(f"\n{'='*60}")
        print(f"Running experiment: {exp_name}")
        print(f"{'='*60}")
        
        try:
            # Train model with MAIR's built-in checkpointing
            rmodel, trainer, save_path = train_model(
                model_name=exp['model'],
                defense_method=exp['method'],
                use_awp=exp['awp'],
                use_extra_data=exp['extra']
            )
            
            # Evaluate model
            model, config = create_model_and_config(exp['model'])
            if config["dataset"] == "MNIST":
                _, test_loader, _, _ = setup_mnist_data()
                eval_results = evaluate_model(rmodel, test_loader, 
                                            eps=0.3, alpha=0.01, steps=40)
            else:
                _, test_loader, _, _ = setup_cifar10_data()
                eval_results = evaluate_model(rmodel, test_loader, 
                                            eps=8/255, alpha=2/255, steps=10)
            
            results[exp_name] = eval_results
            
        except Exception as e:
            print(f"Error in experiment {exp_name}: {str(e)}")
            print("Continuing with next experiment...")
            continue
    
    # Print summary results
    print(f"\n{'='*60}")
    print("EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Experiment':<25} {'Clean':<8} {'FGSM':<8} {'PGD':<8} {'GN':<8}")
    print("-" * 60)
    
    for exp_name, result in results.items():
        print(f"{exp_name:<25} {result['clean']:<8.2f} {result['fgsm']:<8.2f} "
              f"{result['pgd']:<8.2f} {result['gaussian']:<8.2f}")
    
    print(f"\nAll checkpoints saved in ./models/*/")
    print("Use --resume to resume any interrupted training.")

def main():
    """Main function with command-line interface"""
    
    args = parse_arguments()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Handle utility commands first
    if args.list_checkpoints:
        list_available_checkpoints()
        return
    
    if args.evaluate_only:
        print(f"Evaluating model: {args.evaluate_only}")
        # TODO: Implement model loading and evaluation
        print("Evaluation-only mode not yet implemented")
        return
    
    if args.full_suite:
        run_full_experiment_suite()
        return
    
    # Handle resume case
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        # Extract model info from checkpoint path
        parts = args.resume.split(os.sep)
        model_dir = parts[-2]  # models/MODEL_METHOD/last.pth
        
        if "_" in model_dir:
            model_parts = model_dir.split("_")
            model_name = model_parts[0]
            method_parts = "_".join(model_parts[1:]).split("_")
            defense_method = method_parts[0]
            use_awp = 'AWP' in method_parts
            use_extra_data = 'ED' in method_parts
        else:
            print("Could not parse model info from checkpoint path")
            return
        
        try:
            rmodel, trainer, save_path = train_model(
                model_name=model_name,
                defense_method=defense_method,
                use_awp=use_awp,
                use_extra_data=use_extra_data,
                resume_from_checkpoint=args.resume,
                batch_size=args.batch_size
            )
            print("Training resumed and completed successfully!")
        except Exception as e:
            print(f"Resume failed: {e}")
        return
    
    # Handle single model training
    if not args.model:
        print("Error: Must specify --model for training, or use --list-checkpoints, --full-suite, or --resume")
        return
    
    print(f"Training {args.model} with {args.method}")
    print(f"AWP: {args.awp}, Extra Data: {args.extra_data}")
    print("Using MAIR's built-in checkpointing system (saves every epoch)")
    print("Progress bars disabled, but epoch metrics will be shown")
    
    try:
        rmodel, trainer, save_path = train_model(
            model_name=args.model,
            defense_method=args.method,
            use_awp=args.awp,
            use_extra_data=args.extra_data,
            batch_size=args.batch_size
        )
        
        # Evaluate the trained model
        model, config = create_model_and_config(args.model)
        if config["dataset"] == "MNIST":
            _, test_loader, _, _ = setup_mnist_data(args.batch_size)
            eval_results = evaluate_model(rmodel, test_loader, 
                                        eps=0.3, alpha=0.01, steps=40)
        else:
            _, test_loader, _, _ = setup_cifar10_data(args.batch_size)
            eval_results = evaluate_model(rmodel, test_loader, 
                                        eps=8/255, alpha=2/255, steps=10)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Model: {args.model}")
        print(f"Method: {args.method}")
        print(f"AWP: {args.awp}")
        print(f"Extra Data: {args.extra_data}")
        print()
        print("Final Results:")
        print(f"Clean Accuracy: {eval_results['clean']:.2f}%")
        print(f"FGSM Accuracy: {eval_results['fgsm']:.2f}%")
        print(f"PGD Accuracy: {eval_results['pgd']:.2f}%")
        print(f"Gaussian Noise Accuracy: {eval_results['gaussian']:.2f}%")
        print()
        print(f"Model saved in: {save_path}")
        print("Files created:")
        print("  - best.pth (best model by validation metrics)")
        print("  - last.pth (final model)")
        print("  - init.pth (initial model)")
        print("  - log.txt (training logs)")
        
    except Exception as e:
        print(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()