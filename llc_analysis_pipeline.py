# -*- coding: utf-8 -*-
"""
LLC Analysis Pipeline for Pre-trained Adversarial Training Models

This script provides a comprehensive pipeline for measuring Local Learning Coefficients
on pre-trained models and checkpoints from adversarial training experiments.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

# Import your existing modules
from AT_replication_complete import create_model_and_config
from llc_measurement import LLCMeasurer, LLCConfig

# Import torchvision for datasets (same as existing code)
from torchvision import datasets, transforms


class LLCAnalysisPipeline:
    """
    Comprehensive LLC analysis pipeline for pre-trained models.
    
    This class provides methods to:
    1. Load pre-trained models and checkpoints
    2. Measure LLC for individual models
    3. Analyze LLC trajectories across training checkpoints
    4. Compare LLC across different defense methods
    5. Generate comprehensive reports and visualizations
    """
    
    def __init__(self, llc_config: LLCConfig, base_save_dir: str = "./llc_analysis"):
        self.llc_config = llc_config
        self.measurer = LLCMeasurer(llc_config)
        self.base_save_dir = Path(base_save_dir)
        self.base_save_dir.mkdir(exist_ok=True)
        
        # Create timestamped results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = self.base_save_dir / f"llc_analysis_{timestamp}"
        self.results_dir.mkdir(exist_ok=True)
        
        print(f"LLC Analysis Pipeline initialized")
        print(f"Results will be saved to: {self.results_dir}")
    
    def analyze_single_model(self, 
                           model_path: str,
                           model_name: str,
                           dataset_name: str = "CIFAR10",
                           defense_method: str = "Unknown",
                           run_calibration: bool = True,
                           skip_calibration: bool = False,
                           calibration_path: Optional[str] = None) -> Dict:
        """
        Analyze LLC for a single pre-trained model.
        
        Args:
            model_path: Path to the model checkpoint
            model_name: Name of the model architecture
            dataset_name: Name of the dataset
            defense_method: Defense method used during training
            run_calibration: Whether to run hyperparameter calibration (deprecated, use skip_calibration)
            skip_calibration: Whether to skip hyperparameter calibration
            calibration_path: Path to pre-calibrated hyperparameters JSON file
            
        Returns:
            Dictionary with analysis results
        """
        print(f"\nAnalyzing single model: {model_path}")
        print(f"Model: {model_name}, Dataset: {dataset_name}, Defense: {defense_method}")
        
        # Create experiment directory
        experiment_name = f"{model_name}_{defense_method}_single"
        experiment_dir = self.results_dir / experiment_name
        experiment_dir.mkdir(exist_ok=True)
        
        # Load model
        model = self._load_model(model_path, model_name)
        
        # Get data
        train_loader, test_loader = self._get_data_loaders(dataset_name)
        
        # Handle hyperparameter calibration
        optimal_params = None
        if skip_calibration and calibration_path:
            print("Loading pre-calibrated hyperparameters...")
            optimal_params = self.measurer.load_calibrated_hyperparameters(calibration_path)
            if optimal_params is None:
                print("Failed to load pre-calibrated parameters, falling back to calibration...")
                optimal_params = self.measurer.calibrate_hyperparameters(
                    model, train_loader, save_path=str(experiment_dir / "calibration")
                )
        elif run_calibration and not skip_calibration:
            # Run calibration if requested (legacy behavior)
            print("Running hyperparameter calibration...")
            optimal_params = self.measurer.calibrate_hyperparameters(
                model, train_loader, save_path=str(experiment_dir / "calibration")
            )
        
        # Run diagnostics
        print("Running diagnostics...")
        diagnostic_results = self.measurer.run_diagnostics(
            model, train_loader, hyperparams=optimal_params,
            save_path=str(experiment_dir / "diagnostics")
        )
        
        # Estimate LLC
        print("Estimating LLC...")
        llc_results = self.measurer.estimate_llc(
            model, train_loader, hyperparams=optimal_params, run_diagnostics=True
        )
        
        # Save results
        results = {
            'model_path': model_path,
            'model_name': model_name,
            'dataset_name': dataset_name,
            'defense_method': defense_method,
            'llc_results': llc_results,
            'diagnostic_results': diagnostic_results,
            'optimal_params': optimal_params,
            'experiment_dir': str(experiment_dir)
        }
        
        self._save_analysis_results(results, experiment_dir)
        
        print(f"LLC Estimate: {llc_results['llc/mean']:.4f} ± {llc_results['llc/std']:.4f}")
        return results
    
    def analyze_checkpoint_trajectory(self,
                                    checkpoint_dir: str,
                                    model_name: str,
                                    dataset_name: str = "CIFAR10",
                                    defense_method: str = "Unknown",
                                    checkpoint_pattern: str = "*.pth",
                                    max_checkpoints: Optional[int] = None,
                                    skip_calibration: bool = False,
                                    calibration_path: Optional[str] = None,
                                    data_split: str = "train") -> Dict:
        """
        Analyze LLC trajectory across model checkpoints.
        
        Args:
            checkpoint_dir: Directory containing model checkpoints
            model_name: Name of the model architecture
            dataset_name: Name of the dataset
            defense_method: Defense method used
            checkpoint_pattern: Pattern to match checkpoint files
            max_checkpoints: Maximum number of checkpoints to analyze
            skip_calibration: Whether to skip hyperparameter calibration
            calibration_path: Path to pre-calibrated hyperparameters JSON file
            data_split: Which data split to use for LLC evaluation ("train" or "test", default: "train")
            
        Returns:
            Dictionary with trajectory analysis results
        """
        print(f"\nAnalyzing LLC trajectory for {model_name} on {dataset_name}")
        print(f"Checkpoint directory: {checkpoint_dir}")
        print(f"Defense method: {defense_method}")
        print(f"Data split for LLC evaluation: {data_split}")
        
        # Validate data_split parameter
        if data_split not in ["train", "test"]:
            raise ValueError(f"data_split must be 'train' or 'test', got: {data_split}")
        
        # Create experiment directory
        experiment_name = f"{model_name}_{dataset_name}_{defense_method}_trajectory"
        experiment_dir = self.results_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Load checkpoints
        checkpoints = self._load_checkpoints(checkpoint_dir, model_name, checkpoint_pattern)
        
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {checkpoint_dir}")
        
        # Limit number of checkpoints if specified
        if max_checkpoints and len(checkpoints) > max_checkpoints:
            # Sample evenly across the trajectory
            indices = np.linspace(0, len(checkpoints)-1, max_checkpoints, dtype=int)
            checkpoints = [checkpoints[i] for i in indices]
            print(f"Sampled {max_checkpoints} checkpoints from trajectory")
        
        print(f"Loaded {len(checkpoints)} checkpoints")
        
        # Get data loaders
        train_loader, test_loader = self._get_data_loaders(dataset_name)
        
        # Select appropriate data loader based on data_split parameter
        if data_split == "train":
            llc_data_loader = train_loader
            print(f"Using TRAINING data for LLC evaluation (dataset size: {len(train_loader.dataset)})")
        else:  # data_split == "test"
            llc_data_loader = test_loader
            print(f"Using TEST data for LLC evaluation (dataset size: {len(test_loader.dataset)})")
        
        # Handle hyperparameter calibration
        if skip_calibration and calibration_path:
            print("Loading pre-calibrated hyperparameters...")
            optimal_params = self.measurer.load_calibrated_hyperparameters(calibration_path)
            if optimal_params is None:
                print("Failed to load pre-calibrated parameters, falling back to calibration...")
                optimal_params = self.measurer.calibrate_hyperparameters(
                    checkpoints[-1], train_loader, save_path=str(experiment_dir / "calibration")
                )
        else:
            # Calibrate hyperparameters on the final model
            print("Calibrating hyperparameters on final model...")
            optimal_params = self.measurer.calibrate_hyperparameters(
                checkpoints[-1], train_loader, save_path=str(experiment_dir / "calibration")
            )
        
        # Measure LLC trajectory
        print("Measuring LLC trajectory...")
        checkpoint_names = [f"checkpoint_{i}" for i in range(len(checkpoints))]
        
        trajectory_results = self.measurer.measure_llc_trajectory(
            model_checkpoints=checkpoints,
            train_loader=llc_data_loader,  # Use selected data loader (train or test)
            checkpoint_names=checkpoint_names,
            hyperparams=optimal_params,
            save_path=str(experiment_dir / "llc_results")
        )
        
        # Add metadata
        trajectory_results.update({
            'model_name': model_name,
            'dataset_name': dataset_name,
            'defense_method': defense_method,
            'checkpoint_dir': checkpoint_dir,
            'num_checkpoints': len(checkpoints),
            'optimal_params': optimal_params,
            'data_split': data_split,
            'data_loader_size': len(llc_data_loader.dataset)
        })
        
        # Save results
        self._save_trajectory_results(trajectory_results, experiment_dir)
        
        # Note: Trajectory visualization is automatically created by LLCMeasurer.measure_llc_trajectory()
        print(f"Trajectory visualization saved in: {experiment_dir}")
        
        return trajectory_results
    
    def compare_defense_methods(self,
                              model_configs: List[Tuple[str, str, str, str]],
                              dataset_name: str = "CIFAR10") -> Dict:
        """
        Compare LLC across different defense methods.
        
        Args:
            model_configs: List of (model_path, model_name, defense_method, description)
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with comparison results
        """
        print(f"\nComparing defense methods across {len(model_configs)} models")
        
        results = {}
        
        for model_path, model_name, defense_method, description in model_configs:
            print(f"\nProcessing {defense_method} ({description})...")
            
            try:
                result = self.analyze_single_model(
                    model_path=model_path,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    defense_method=defense_method,
                    run_calibration=False  # Use same hyperparams for fair comparison
                )
                results[defense_method] = result
                
            except Exception as e:
                print(f"Error analyzing {defense_method}: {e}")
                results[defense_method] = {'error': str(e)}
        
        # Create comparison plots
        self._create_defense_comparison_plots(results, dataset_name)
        
        return results
    
    def compare_clean_vs_adversarial_llc(self,
                                       checkpoint_dir: str,
                                       model_name: str,
                                       dataset_name: str = "CIFAR10",
                                       defense_method: str = "Unknown",
                                       checkpoint_pattern: str = "*.pth",
                                       max_checkpoints: Optional[int] = None,
                                       adversarial_attack: str = "pgd",
                                       adversarial_eps: float = 8/255,
                                       adversarial_steps: int = 10,
                                       resume_from_checkpoint: int = 0,
                                       output_dir: str = None) -> Dict:
        """
        Compare LLC trajectories on clean vs adversarial data.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            model_name: Name of the model architecture
            dataset_name: Name of the dataset
            defense_method: Defense method used during training
            checkpoint_pattern: Pattern to match checkpoint files
            max_checkpoints: Maximum number of checkpoints to analyze
            adversarial_attack: Type of adversarial attack
            adversarial_eps: Epsilon for adversarial attacks
            adversarial_steps: Number of attack steps
            
        Returns:
            Dictionary with comparison results
        """
        print(f"\nComparing clean vs adversarial LLC: {checkpoint_dir}")
        print(f"Model: {model_name}, Dataset: {dataset_name}, Defense: {defense_method}")
        print(f"Adversarial attack: {adversarial_attack}, ε={adversarial_eps}, steps={adversarial_steps}")
        
        # Create or use existing experiment directory
        if output_dir and resume_from_checkpoint > 0:
            # Use specified output directory for resume
            experiment_dir = Path(output_dir)
            print(f"📁 Using existing experiment directory: {experiment_dir}")
        else:
            # Create new experiment directory
            experiment_name = f"{model_name}_{defense_method}_clean_vs_adv"
            experiment_dir = self.results_dir / experiment_name
            print(f"📁 Creating new experiment directory: {experiment_dir}")
        
        experiment_dir.mkdir(exist_ok=True)
        
        # Load checkpoints
        checkpoints = self._load_checkpoints(checkpoint_dir, model_name, checkpoint_pattern)
        
        if max_checkpoints and len(checkpoints) > max_checkpoints:
            indices = np.linspace(0, len(checkpoints)-1, max_checkpoints, dtype=int)
            checkpoints = [checkpoints[i] for i in indices]
            print(f"Sampled {max_checkpoints} checkpoints from trajectory")
        
        print(f"Loaded {len(checkpoints)} checkpoints")
        
        # Get data
        train_loader, test_loader = self._get_data_loaders(dataset_name)
        
        # Create configurations for clean and adversarial LLC
        clean_config = LLCConfig(
            epsilon=self.llc_config.epsilon,
            gamma=self.llc_config.gamma,
            num_chains=self.llc_config.num_chains,
            num_steps=self.llc_config.num_steps,
            batch_size=self.llc_config.batch_size,
            data_type="clean"
        )
        
        adv_config = LLCConfig(
            epsilon=self.llc_config.epsilon,
            gamma=self.llc_config.gamma,
            num_chains=self.llc_config.num_chains,
            num_steps=self.llc_config.num_steps,
            batch_size=self.llc_config.batch_size,
            data_type="adversarial",
            adversarial_attack=adversarial_attack,
            adversarial_eps=adversarial_eps,
            adversarial_steps=adversarial_steps
        )
        
        # Create measurers
        clean_measurer = LLCMeasurer(clean_config)
        adv_measurer = LLCMeasurer(adv_config)
        
        # Calibrate hyperparameters on final model (using clean data for consistency)
        calibration_results_path = experiment_dir / "calibration" / "calibration_results.json"
        if calibration_results_path.exists() and resume_from_checkpoint > 0:
            print("📂 Loading existing calibration results...")
            with open(calibration_results_path, 'r') as f:
                calibration_data = json.load(f)
                # Extract just the optimal parameters (not the full calibration data)
                optimal_params = calibration_data.get('optimal_params', calibration_data)
            print("✅ Calibration results loaded from existing file")
        else:
            print("Calibrating hyperparameters on final model (clean data)...")
            optimal_params = clean_measurer.calibrate_hyperparameters(
                checkpoints[-1], train_loader, save_path=str(experiment_dir / "calibration")
            )
        
        # Measure LLC trajectories
        checkpoint_names = [f"checkpoint_{i}" for i in range(len(checkpoints))]
        
        # Check if clean trajectory is already complete
        clean_trajectory_path = experiment_dir / "clean_llc_results" / "llc_trajectory.json"
        if clean_trajectory_path.exists() and resume_from_checkpoint > 0:
            print("📂 Loading existing clean LLC trajectory...")
            with open(clean_trajectory_path, 'r') as f:
                clean_trajectory = json.load(f)
            print("✅ Clean trajectory loaded from existing results")
        else:
            print("Measuring clean LLC trajectory...")
            clean_trajectory = clean_measurer.measure_llc_trajectory(
                model_checkpoints=checkpoints,
                train_loader=train_loader,
                checkpoint_names=checkpoint_names,
                hyperparams=optimal_params,
                save_path=str(experiment_dir / "clean_llc_results")
            )
        
        print("Measuring adversarial LLC trajectory...")
        adv_trajectory = adv_measurer.measure_llc_trajectory(
            model_checkpoints=checkpoints,
            train_loader=train_loader,
            checkpoint_names=checkpoint_names,
            hyperparams=optimal_params,
            save_path=str(experiment_dir / "adversarial_llc_results"),
            resume_from_checkpoint=resume_from_checkpoint
        )
        
        # Create comparison results
        comparison_results = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'defense_method': defense_method,
            'checkpoint_dir': checkpoint_dir,
            'num_checkpoints': len(checkpoints),
            'optimal_params': optimal_params,
            'adversarial_config': {
                'attack': adversarial_attack,
                'eps': adversarial_eps,
                'steps': adversarial_steps
            },
            'clean_trajectory': clean_trajectory,
            'adversarial_trajectory': adv_trajectory,
            'checkpoint_names': checkpoint_names
        }
        
        # Save comparison results
        self._save_comparison_results(comparison_results, experiment_dir)
        
        # Create comparison visualization
        self._create_clean_vs_adv_visualization(comparison_results, experiment_dir)
        
        return comparison_results
    
    def _load_model(self, model_path: str, model_name: str) -> nn.Module:
        """Load a pre-trained model"""
        if not os.path.exists(model_path):
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
        
        # When loading MAIR checkpoints, we need to:
        if 'rmodel_state_dict' in checkpoint:
            rmodel_state = checkpoint['rmodel_state_dict']
            # Extract only the base model weights (keys starting with 'model.')
            base_model_state = {}
            for key, value in rmodel_state.items():
                if key.startswith('model.'):
                    base_key = key[6:]  # Remove 'model.' prefix
                    base_model_state[base_key] = value
            model.load_state_dict(base_model_state)
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume the checkpoint is just the state dict
            model.load_state_dict(checkpoint)
        
        print(f"Loaded model from {model_path}")
        return model
    
    def _load_checkpoints(self, checkpoint_dir: str, model_name: str, pattern: str) -> List[nn.Module]:
        """Load multiple checkpoints from a directory"""
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        checkpoint_files = sorted([f for f in checkpoint_path.glob(pattern)])
        if not checkpoint_files:
            raise ValueError(f"No checkpoint files found matching pattern: {pattern}")
        
        checkpoints = []
        for checkpoint_file in checkpoint_files:
            try:
                # Create model instance
                model, config = create_model_and_config(model_name)
                
                # Load checkpoint with weights_only=False for MAIR checkpoints
                try:
                    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
                except Exception as e:
                    print(f"Warning: Failed to load {checkpoint_file} with weights_only=False: {e}")
                    # Try with weights_only=True as fallback
                    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
                
                # Debug: Print checkpoint structure
                # print(f"  Checkpoint keys: {list(checkpoint.keys())}")
                
                # Handle different checkpoint formats
                loaded_successfully = False
                
                # Try 'rmodel' first (your checkpoints have this)
                if 'rmodel' in checkpoint:
                    rmodel = checkpoint['rmodel']
                    
                    # If rmodel is a PyTorch module, get its state_dict
                    if hasattr(rmodel, 'state_dict'):
                        model_state = rmodel.state_dict()
                        model.load_state_dict(model_state)
                        loaded_successfully = True
                    elif isinstance(rmodel, dict):
                        
                        # Check if keys have 'model.' prefix and need stripping
                        model_keys = [k for k in rmodel.keys() if k.startswith('model.')]
                        if model_keys:
                            # Strip 'model.' prefix from keys
                            cleaned_state_dict = {}
                            for key, value in rmodel.items():
                                if key.startswith('model.'):
                                    clean_key = key[6:]  # Remove 'model.' prefix
                                    cleaned_state_dict[clean_key] = value
                            model.load_state_dict(cleaned_state_dict)
                        else:
                            # Use as-is if no 'model.' prefix
                            model.load_state_dict(rmodel)
                        loaded_successfully = True
                
                # Try other common formats if rmodel didn't work
                elif 'rmodel_state_dict' in checkpoint:
                    rmodel_state = checkpoint['rmodel_state_dict']
                    # Extract only the base model weights (keys starting with 'model.')
                    base_model_state = {}
                    for key, value in rmodel_state.items():
                        if key.startswith('model.'):
                            base_key = key[6:]  # Remove 'model.' prefix
                            base_model_state[base_key] = value
                    model.load_state_dict(base_model_state)
                    loaded_successfully = True
                elif 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    loaded_successfully = True
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                    loaded_successfully = True
                else:
                    # Assume the checkpoint is just the state dict
                    try:
                        model.load_state_dict(checkpoint)
                        loaded_successfully = True
                    except Exception as e:
                        print(f"  Failed to load as direct state dict: {e}")
                
                if not loaded_successfully:
                    raise ValueError(f"Could not load model from checkpoint with keys: {list(checkpoint.keys())}")
                
                # Add debugging information
                print(f"  Model device after loading: {next(model.parameters()).device}")
                print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
                
                # Set model to evaluation mode
                model.eval()
                
                checkpoints.append(model)
                print(f"Loaded checkpoint: {checkpoint_file.name}")
                
            except Exception as e:
                print(f"Error loading checkpoint {checkpoint_file}: {e}")
                import traceback
                traceback.print_exc()
        
        if not checkpoints:
            raise ValueError(f"No checkpoints could be loaded from {checkpoint_dir}")
        
        return checkpoints
    
    def _get_data_loaders(self, dataset_name: str) -> Tuple[DataLoader, DataLoader]:
        """Get data loaders for the specified dataset"""
        if dataset_name == "MNIST":
            # MNIST setup (same as AT_replication_complete.py)
            MEAN = [0.1307]
            STD = [0.3081]
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD)
            ])
            
            train_dataset = datasets.MNIST(
                "./data", train=True, transform=transform, download=True
            )
            test_dataset = datasets.MNIST(
                "./data", train=False, transform=transform, download=True
            )
        elif dataset_name == "CIFAR10":
            # CIFAR10 setup (same as AT_replication_complete.py)
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
            
            train_dataset = datasets.CIFAR10(
                "./data", train=True, transform=train_transform, download=True
            )
            test_dataset = datasets.CIFAR10(
                "./data", train=False, transform=test_transform, download=True
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.llc_config.batch_size, 
            shuffle=True,
            num_workers=4
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.llc_config.batch_size, 
            shuffle=False,
            num_workers=4
        )
        
        return train_loader, test_loader
    
    def _save_analysis_results(self, results: Dict, experiment_dir: Path):
        """Save analysis results to file"""
        results_path = experiment_dir / "analysis_results.json"
        
        # Convert numpy arrays and scalar types to JSON-serializable format
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Analysis results saved to: {results_path}")
    
    def _save_trajectory_results(self, results: Dict, experiment_dir: Path):
        """Save trajectory results to file"""
        results_path = experiment_dir / "trajectory_results.json"
        
        # Convert numpy arrays and scalar types to JSON-serializable format
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Trajectory results saved to: {results_path}")
    
    def _save_comparison_results(self, results: Dict, experiment_dir: Path):
        """Save comparison results to file"""
        results_path = experiment_dir / "clean_vs_adv_comparison.json"
        
        # Convert numpy arrays and scalar types to JSON-serializable format
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Comparison results saved to: {results_path}")
    
    
    def _create_defense_comparison_plots(self, results: Dict, dataset_name: str):
        """Create comparison plots for different defense methods"""
        # Filter out failed experiments
        valid_results = {k: v for k, v in results.items() 
                        if 'llc_results' in v and 'error' not in v}
        
        if not valid_results:
            print("No valid results to plot")
            return
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        
        defense_methods = []
        llc_means = []
        llc_stds = []
        
        for defense_method, result in valid_results.items():
            llc_data = result['llc_results']
            defense_methods.append(defense_method)
            llc_means.append(llc_data['llc/mean'])
            llc_stds.append(llc_data['llc/std'])
        
        x_positions = range(len(defense_methods))
        plt.errorbar(x_positions, llc_means, yerr=llc_stds, 
                    marker='o', capsize=5, capthick=2, markersize=8)
        
        plt.xlabel("Defense Method")
        plt.ylabel("Local Learning Coefficient (LLC)")
        plt.title(f"LLC Comparison Across Defense Methods - {dataset_name}")
        plt.xticks(x_positions, defense_methods, rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / f"defense_methods_comparison_{dataset_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Defense methods comparison plot saved to: {plot_path}")
    
    def _create_clean_vs_adv_visualization(self, results: Dict, experiment_dir: Path):
        """Create visualization comparing clean vs adversarial LLC trajectories"""
        clean_means = results['clean_trajectory']['llc_means']
        clean_stds = results['clean_trajectory']['llc_stds']
        adv_means = results['adversarial_trajectory']['llc_means']
        adv_stds = results['adversarial_trajectory']['llc_stds']
        checkpoint_names = results['checkpoint_names']
        
        # Filter out None values
        valid_indices = [i for i, (clean, adv) in enumerate(zip(clean_means, adv_means)) 
                        if clean is not None and adv is not None]
        
        if not valid_indices:
            print("No valid LLC measurements to plot")
            return
        
        valid_clean_means = [clean_means[i] for i in valid_indices]
        valid_clean_stds = [clean_stds[i] for i in valid_indices]
        valid_adv_means = [adv_means[i] for i in valid_indices]
        valid_adv_stds = [adv_stds[i] for i in valid_indices]
        valid_names = [checkpoint_names[i] for i in valid_indices]
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        
        x_positions = range(len(valid_indices))
        
        # Main comparison plot
        plt.errorbar(x_positions, valid_clean_means, yerr=valid_clean_stds, 
                    marker='o', capsize=5, capthick=2, linewidth=2, markersize=8,
                    label='Clean Data', color='blue')
        plt.errorbar(x_positions, valid_adv_means, yerr=valid_adv_stds, 
                    marker='s', capsize=5, capthick=2, linewidth=2, markersize=8,
                    label='Adversarial Data', color='red')
        
        plt.xlabel("Checkpoint")
        plt.ylabel("Local Learning Coefficient (LLC)")
        plt.title(f"Clean vs Adversarial LLC - {results['model_name']} ({results['defense_method']})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(experiment_dir / "clean_vs_adversarial_llc.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Clean vs adversarial comparison plot saved to: {experiment_dir / 'clean_vs_adversarial_llc.png'}")
        
        # Print summary statistics
        print("\n=== Clean vs Adversarial LLC Summary ===")
        print(f"Model: {results['model_name']} ({results['defense_method']})")
        print(f"Adversarial attack: {results['adversarial_config']['attack']}")
        print(f"Number of checkpoints: {len(valid_indices)}")
        
        if valid_clean_means and valid_adv_means:
            avg_clean = np.mean(valid_clean_means)
            avg_adv = np.mean(valid_adv_means)
            avg_diff = avg_adv - avg_clean
            
            print(f"Average Clean LLC: {avg_clean:.4f}")
            print(f"Average Adversarial LLC: {avg_adv:.4f}")
            print(f"Average Difference: {avg_diff:.4f}")
            
            if avg_diff > 0:
                print("→ Adversarial LLC is higher (more complex)")
            else:
                print("→ Clean LLC is higher (more complex)")
        
        print("=" * 40)
    
    def run_calibration_only(self,
                           model_path: str,
                           model_name: str,
                           dataset_name: str = "CIFAR10",
                           defense_method: str = "Unknown") -> Dict:
        """
        Run only hyperparameter calibration on a single model.
        
        Args:
            model_path: Path to the model checkpoint
            model_name: Name of the model architecture
            dataset_name: Name of the dataset
            defense_method: Defense method used
            
        Returns:
            Dictionary with calibration results
        """
        print(f"\nRunning hyperparameter calibration for {model_name} on {dataset_name}")
        print(f"Model path: {model_path}")
        print(f"Defense method: {defense_method}")
        
        # Create experiment directory
        experiment_name = f"{model_name}_{dataset_name}_{defense_method}_calibration"
        experiment_dir = self.results_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        model, config = create_model_and_config(model_name)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats (same logic as trajectory analysis)
        if 'rmodel' in checkpoint:
            rmodel = checkpoint['rmodel']
            if isinstance(rmodel, dict):
                # Check if keys have 'model.' prefix and need stripping
                model_keys = [k for k in rmodel.keys() if k.startswith('model.')]
                if model_keys:
                    # Strip 'model.' prefix from keys
                    cleaned_state_dict = {}
                    for key, value in rmodel.items():
                        if key.startswith('model.'):
                            clean_key = key[6:]  # Remove 'model.' prefix
                            cleaned_state_dict[clean_key] = value
                    model.load_state_dict(cleaned_state_dict)
                else:
                    model.load_state_dict(rmodel)
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Get data loaders
        train_loader, test_loader = self._get_data_loaders(dataset_name)
        
        # Run calibration
        print("Running hyperparameter calibration...")
        optimal_params = self.measurer.calibrate_hyperparameters(
            model, train_loader, save_path=str(experiment_dir)
        )
        
        # Save results
        results = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "defense_method": defense_method,
            "model_path": model_path,
            "optimal_params": optimal_params,
            "experiment_dir": str(experiment_dir)
        }
        
        # Save to JSON
        results_path = experiment_dir / "calibration_only_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n🎯 Calibration Complete!")
        print(f"Optimal parameters: {optimal_params}")
        print(f"Results saved to: {experiment_dir}")
        print(f"Calibration file: {experiment_dir}/calibration_results.json")
        
        return results
    
    def run_adversarial_only_trajectory(self, 
                                      checkpoint_dir: str, 
                                      model_name: str,
                                      dataset_name: str = "Unknown",
                                      defense_method: str = "Unknown",
                                      calibration_path: str = None,
                                      max_checkpoints: int = None,
                                      output_dir: str = None,
                                      resume_from_checkpoint: int = 0) -> Dict:
        """
        Run adversarial-only LLC trajectory analysis using pre-calibrated hyperparameters
        
        Args:
            checkpoint_dir: Directory containing model checkpoints
            model_name: Name of the model architecture
            dataset_name: Name of the dataset
            defense_method: Defense method used
            calibration_path: Path to calibration results JSON file
            max_checkpoints: Maximum number of checkpoints to analyze
            output_dir: Custom output directory (if None, creates new experiment directory)
            
        Returns:
            Dictionary containing analysis results
        """
        print(f"\n🎯 Running adversarial-only LLC trajectory analysis...")
        print(f"Checkpoint dir: {checkpoint_dir}")
        print(f"Model: {model_name}")
        print(f"Dataset: {dataset_name}")
        print(f"Calibration file: {calibration_path}")
        
        # Load pre-calibrated hyperparameters
        if not calibration_path or not Path(calibration_path).exists():
            raise FileNotFoundError(f"Calibration file not found: {calibration_path}")
        
        with open(calibration_path, 'r') as f:
            calibration_data = json.load(f)
        
        # Extract optimal hyperparameters
        if 'optimal_params' not in calibration_data:
            raise ValueError("Calibration file must contain 'optimal_params' key")
        
        optimal_params = calibration_data['optimal_params']
        print(f"📋 Loaded calibrated hyperparameters:")
        print(f"  ε (epsilon): {optimal_params['epsilon']:.2e}")
        print(f"  β (nbeta): {optimal_params['nbeta']:.4f}")
        print(f"  γ (gamma): {optimal_params['gamma']:.2f}")
        
        # Update measurer config with calibrated params
        self.measurer.config.epsilon = optimal_params['epsilon']
        self.measurer.config.nbeta = optimal_params['nbeta'] 
        self.measurer.config.gamma = optimal_params['gamma']
        
        # Enable adversarial mode
        self.measurer.config.data_type = "adversarial"
        
        # Get data loaders
        train_loader, test_loader = self._get_data_loaders(dataset_name)
        
        # Load checkpoints
        checkpoints = self._load_checkpoints(checkpoint_dir, model_name, "*.pth")
        
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {checkpoint_dir}")
        
        # Limit number of checkpoints if specified (sample evenly like other methods)
        if max_checkpoints and len(checkpoints) > max_checkpoints:
            # Sample evenly across the trajectory
            indices = np.linspace(0, len(checkpoints)-1, max_checkpoints, dtype=int)
            checkpoints = [checkpoints[i] for i in indices]
            print(f"Sampled {max_checkpoints} checkpoints from trajectory")
        
        print(f"📦 Found {len(checkpoints)} checkpoints to analyze")
        
        # Create or use specified output directory
        if output_dir:
            experiment_dir = Path(output_dir)
            print(f"📁 Using custom output directory: {experiment_dir}")
        else:
            experiment_name = f"{model_name}_{dataset_name}_{defense_method}_adversarial_only"
            experiment_dir = self.results_dir / experiment_name
            print(f"📁 Creating new experiment directory: {experiment_dir}")
        
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Run adversarial LLC measurement
        print("Measuring adversarial LLC trajectory...")
        checkpoint_names = [f"checkpoint_{i}" for i in range(len(checkpoints))]
        
        results = self.measurer.measure_llc_trajectory(
            model_checkpoints=checkpoints,
            train_loader=train_loader,
            checkpoint_names=checkpoint_names,
            hyperparams=optimal_params,
            save_path=str(experiment_dir),  # Save directly to experiment_dir, not nested
            resume_from_checkpoint=resume_from_checkpoint
        )
        
        # Add metadata for plotting (following pattern from other methods)
        if results:
            results.update({
                'model_name': model_name,
                'dataset_name': dataset_name,
                'defense_method': defense_method,
                'checkpoint_dir': checkpoint_dir,
                'num_checkpoints': len(checkpoints),
                'adversarial_mode': True,
                'calibration_params': optimal_params,
                'analysis_type': 'adversarial_only_trajectory',
                'calibration_source': calibration_path
            })
        
        # Note: llc_trajectory.png is automatically created by measure_llc_trajectory
        print("📊 Trajectory plot automatically saved as llc_trajectory.png")
        
        print(f"✅ Adversarial-only analysis complete!")
        print(f"📁 Results saved to: {experiment_dir}")
        
        return results
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of all analyses"""
        report_path = self.results_dir / "analysis_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("LLC Analysis Summary Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Results Directory: {self.results_dir}\n\n")
            
            # List all experiment directories
            experiment_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]
            f.write(f"Total Experiments: {len(experiment_dirs)}\n\n")
            
            for exp_dir in experiment_dirs:
                f.write(f"Experiment: {exp_dir.name}\n")
                
                # Check for results files
                results_files = list(exp_dir.glob("*.json"))
                for results_file in results_files:
                    f.write(f"  - {results_file.name}\n")
                
                # Check for plots
                plot_files = list(exp_dir.glob("*.png"))
                for plot_file in plot_files:
                    f.write(f"  - {plot_file.name}\n")
                
                f.write("\n")
        
        print(f"Summary report saved to: {report_path}")
        return str(report_path)


def main():
    """Main function to run LLC analysis pipeline"""
    parser = argparse.ArgumentParser(description="LLC Analysis Pipeline for Pre-trained Models")
    parser.add_argument("--mode", choices=["single", "trajectory", "compare", "clean_vs_adv", "calibration", "adversarial_only"], required=True,
                       help="Analysis mode")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, help="Directory containing checkpoints")
    parser.add_argument("--model_name", type=str, default="ResNet18", help="Model architecture")
    parser.add_argument("--dataset", type=str, default="CIFAR10", help="Dataset name")
    parser.add_argument("--defense_method", type=str, default="Unknown", help="Defense method")
    parser.add_argument("--config_file", type=str, help="JSON config file for multiple models")
    
    # Adversarial attack parameters
    parser.add_argument("--adversarial_attack", type=str, default="pgd", 
                       choices=["pgd", "fgsm"], help="Type of adversarial attack")
    parser.add_argument("--adversarial_eps", type=float, default=8/255, 
                       help="Epsilon for adversarial attacks")
    parser.add_argument("--adversarial_steps", type=int, default=10, 
                       help="Number of attack steps")
    
    # New arguments for trajectory analysis
    parser.add_argument("--skip_calibration", action="store_true",
                       help="Skip hyperparameter calibration and use pre-calibrated parameters")
    parser.add_argument("--calibration_path", type=str,
                       help="Path to pre-calibrated hyperparameters JSON file")
    parser.add_argument("--max_checkpoints", type=int, default=None,
                       help="Maximum number of checkpoints to analyze (default: all)")
    parser.add_argument("--output_dir", type=str,
                       help="Custom output directory for results (used with adversarial_only mode)")
    parser.add_argument("--resume_from_checkpoint", type=int, default=0,
                       help="Resume analysis from specific checkpoint index (0-based, default: 0)")
    
    args = parser.parse_args()
    
    # Setup LLC configuration
    llc_config = LLCConfig(
        model_name=args.model_name,
        epsilon=1e-4,
        gamma=1.0,  # Start small per guide recommendations
        num_chains=2,  #8,
        num_steps=500,  #2000,  # Total steps (not draws)
        batch_size=512,
        calibration_epsilons=[1e-6, 1e-5, 1e-4, 1e-3],  # Expanded range
        calibration_gammas=[1.0, 10.0, 100.0, 200.0],  # Include more values
        target_mala_acceptance=0.92  # Target MALA acceptance rate
    )
    
    print(f"Initial LLC Configuration (may change): {llc_config.num_chains} chains, {llc_config.num_steps} total steps, {llc_config.get_effective_draws()} effective draws")
    
    # Initialize pipeline
    pipeline = LLCAnalysisPipeline(llc_config)
    
    if args.mode == "single":
        if not args.model_path:
            raise ValueError("--model_path is required for single model analysis")
        
        # Validate calibration arguments
        if args.skip_calibration and not args.calibration_path:
            raise ValueError("--calibration_path is required when --skip_calibration is used")
        
        pipeline.analyze_single_model(
            model_path=args.model_path,
            model_name=args.model_name,
            dataset_name=args.dataset,
            defense_method=args.defense_method,
            skip_calibration=args.skip_calibration,
            calibration_path=args.calibration_path
        )
    
    elif args.mode == "trajectory":
        if not args.checkpoint_dir:
            raise ValueError("--checkpoint_dir is required for trajectory analysis")
        
        # Validate calibration arguments
        if args.skip_calibration and not args.calibration_path:
            raise ValueError("--calibration_path is required when --skip_calibration is used")
        
        pipeline.analyze_checkpoint_trajectory(
            checkpoint_dir=args.checkpoint_dir,
            model_name=args.model_name,
            dataset_name=args.dataset,
            defense_method=args.defense_method,
            max_checkpoints=args.max_checkpoints,
            skip_calibration=args.skip_calibration,
            calibration_path=args.calibration_path
        )
    
    elif args.mode == "compare":
        if not args.config_file:
            raise ValueError("--config_file is required for comparison analysis")
        
        with open(args.config_file, 'r') as f:
            model_configs = json.load(f)
        
        pipeline.compare_defense_methods(model_configs, args.dataset)
    
    elif args.mode == "clean_vs_adv":
        if not args.checkpoint_dir:
            raise ValueError("--checkpoint_dir is required for clean vs adversarial analysis")
        
        pipeline.compare_clean_vs_adversarial_llc(
            checkpoint_dir=args.checkpoint_dir,
            model_name=args.model_name,
            dataset_name=args.dataset,
            defense_method=args.defense_method,
            max_checkpoints=args.max_checkpoints,
            adversarial_attack=args.adversarial_attack,
            adversarial_eps=args.adversarial_eps,
            adversarial_steps=args.adversarial_steps,
            resume_from_checkpoint=args.resume_from_checkpoint,
            output_dir=args.output_dir
        )
    
    elif args.mode == "calibration":
        if not args.model_path:
            raise ValueError("--model_path is required for calibration mode")
        
        pipeline.run_calibration_only(
            model_path=args.model_path,
            model_name=args.model_name,
            dataset_name=args.dataset,
            defense_method=args.defense_method
        )
    
    elif args.mode == "adversarial_only":
        if not args.checkpoint_dir:
            raise ValueError("--checkpoint_dir is required for adversarial_only mode")
        if not args.calibration_path:
            raise ValueError("--calibration_path is required for adversarial_only mode")
        
        pipeline.run_adversarial_only_trajectory(
            checkpoint_dir=args.checkpoint_dir,
            model_name=args.model_name,
            dataset_name=args.dataset,
            defense_method=args.defense_method,
            calibration_path=args.calibration_path,
            max_checkpoints=args.max_checkpoints,
            output_dir=args.output_dir,
            resume_from_checkpoint=args.resume_from_checkpoint
        )
    
    # Generate summary report
    pipeline.generate_summary_report()
    
    print(f"\nAnalysis complete! Results saved to: {pipeline.results_dir}")


if __name__ == "__main__":
    main() 