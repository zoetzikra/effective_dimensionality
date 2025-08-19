# -*- coding: utf-8 -*-
"""
Local Learning Coefficient (LLC) Measurement Module

This module implements LLC measurement using the devinterp library for adversarial training models.
Based on the methodology from Lau et al. (2023) and the devinterp examples.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
from pathlib import Path

# devinterp imports
from devinterp.optim.sgld import SGLD
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.slt.mala import MalaAcceptanceRate
from devinterp.utils import (
    plot_trace, 
    default_nbeta, 
    evaluate_ce,
    get_init_loss_multi_batch
)
from devinterp.vis_utils import EpsilonBetaAnalyzer

warnings.filterwarnings("ignore")

def convert_to_serializable(obj):
    """Convert numpy arrays and scalar types to JSON-serializable format"""
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

@dataclass
class LLCConfig:
    """Configuration for LLC measurement following empirical guide recommendations"""
    
    # Model information (for size-based scaling)
    model_name: str = "unknown"
    num_parameters: Optional[int] = None  # For automatic scaling
    
    # SGLD hyperparameters
    epsilon: float = 1e-4  # Step size (learning rate) - will be auto-scaled
    gamma: float = 1.0   # Localization strength (start small per guide)
    nbeta: Optional[float] = None  # Inverse temperature (auto-set if None)
    
    # Sampling parameters (following guide recommendations)
    num_chains: int = 8  # 4-20 chains recommended
    num_steps: int = 2000  # Total steps per chain (not draws)
    num_burnin_steps: Optional[int] = None  # Auto-set to 90% of total steps
    num_steps_bw_draws: int = 1  # Steps between draws
    
    # Batch size for SGLD
    batch_size: int = 512
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Calibration parameters (expanded ranges per guide)
    calibration_epsilons: List[float] = None
    calibration_gammas: List[float] = None
    
    # Data type for LLC evaluation
    data_type: str = "clean"  # "clean", "adversarial", or "mixed"
    adversarial_attack: str = "pgd"  # Attack type for adversarial data
    adversarial_eps: float = 8/255  # Epsilon for adversarial attacks
    adversarial_steps: int = 10  # Number of attack steps
    
    # Diagnostic targets
    target_mala_acceptance: float = 0.92  # Target MALA acceptance rate (0.9-0.95)
    
    def __post_init__(self):
        # Set default calibration ranges based on guide recommendations
        if self.calibration_epsilons is None:
            # Expanded range for better coverage
            self.calibration_epsilons = [1e-6, 1e-5, 1e-4, 1e-3]
        if self.calibration_gammas is None:
            # Include both small and large values per guide
            self.calibration_gammas = [1.0, 10.0, 100.0, 200.0]
        
        # Auto-set burn-in to 90% of total steps (guide recommendation)
        if self.num_burnin_steps is None:
            self.num_burnin_steps = int(0.9 * self.num_steps)
        
        # Auto-scale epsilon based on model size if num_parameters provided
        if self.num_parameters is not None:
            self.epsilon = self._scale_epsilon_by_model_size()
        
        # Auto-scale gamma based on model type
        self.gamma = self._scale_gamma_by_model_type()
        
        # Validate data type
        if self.data_type not in ["clean", "adversarial", "mixed"]:
            raise ValueError(f"data_type must be 'clean', 'adversarial', or 'mixed', got {self.data_type}")
        
        # Validate MALA acceptance target
        if not (0.5 <= self.target_mala_acceptance <= 0.95):
            raise ValueError(f"target_mala_acceptance must be between 0.5 and 0.95, got {self.target_mala_acceptance}")
    
    def _scale_epsilon_by_model_size(self) -> float:
        """Scale epsilon based on model size per guide recommendations"""
        if self.num_parameters is None:
            return self.epsilon
        
        # Guide recommendations:
        # Small networks (≤10K params): ε = 1e-3 to 1e-5
        # Medium networks (100K-1M params): ε = 1e-4 to 1e-6
        # Large networks (>1M params): ε = 1e-5 to 1e-7
        
        if self.num_parameters <= 10_000:
            return 1e-4  # Small networks
        elif self.num_parameters <= 1_000_000:
            return 1e-5  # Medium networks
        else:
            return 1e-6  # Large networks
    
    def _scale_gamma_by_model_type(self) -> float:
        """Scale gamma based on model type per guide recommendations"""
        model_lower = self.model_name.lower()
        
        # Guide recommendations:
        # ResNet experiments: γ = 1.0
        # Transformer experiments: γ = 100-300
        # General recommendation: Start with γ = 1.0
        print("Now scaling gamma by model type:")
        if "transformer" in model_lower or "gpt" in model_lower or "bert" in model_lower:
            return 100.0  # Transformers
        elif "resnet" in model_lower or "cnn" in model_lower:
            return 1.0  # ResNet/CNN
        else:
            return 1.0  # Default to small value
    
    def get_effective_draws(self) -> int:
        """Calculate effective number of draws after burn-in"""
        return self.num_steps - self.num_burnin_steps
    
    def validate_for_model(self, model: nn.Module) -> None:
        """Validate configuration for a specific model"""
        if self.num_parameters is None:
            # Count parameters
            self.num_parameters = sum(p.numel() for p in model.parameters())
            print(f"Model has {self.num_parameters:,} parameters")
            
            # Re-scale based on actual model size
            self.epsilon = self._scale_epsilon_by_model_size()
            print(f"Auto-scaled epsilon to {self.epsilon:.2e}")
        
        # Print configuration summary
        print(f"\n=== LLC Configuration Summary ===")
        print(f"Model: {self.model_name} ({self.num_parameters:,} parameters)")
        print(f"Step size (ε): {self.epsilon:.2e}")
        print(f"Localization (γ): {self.gamma:.1f}")
        print(f"Chains: {self.num_chains}")
        print(f"Total steps: {self.num_steps}")
        print(f"Burn-in steps: {self.num_burnin_steps} ({100*self.num_burnin_steps/self.num_steps:.0f}%)")
        print(f"Effective draws: {self.get_effective_draws()}")
        print(f"Batch size: {self.batch_size}")
        print(f"Target MALA acceptance: {self.target_mala_acceptance:.2f}")
        print("=" * 40)


class LLCMeasurer:
    """
    Main class for measuring Local Learning Coefficients (LLC) using devinterp.
    
    This class provides methods for:
    1. Hyperparameter calibration
    2. LLC estimation for individual models
    3. LLC tracking throughout training
    4. Diagnostic monitoring
    """
    
    def __init__(self, config: LLCConfig):
        self.config = config
        self.device = config.device
        self.results_cache = {}
        
    def evaluate_model(self, model: nn.Module, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Evaluation function for LLC estimation.
        
        Args:
            model: PyTorch model
            data: Tuple of (inputs, targets)
            
        Returns:
            Tuple of (loss_tensor, additional_info)
        """
        # Ensure model is in evaluation mode
        model.eval()
        
        # CRITICAL: Ensure all parameters require gradients for LLC estimation
        for param in model.parameters():
            param.requires_grad_(True)
        
        inputs, targets = data
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        # Fix device mismatch: ensure model is on the same device as data
        model_device = next(model.parameters()).device
        target_device = torch.device(self.device)
        
        if model_device != target_device:
            # Only print warning once per model, not every evaluation
            if not hasattr(model, '_device_warning_printed'):
                print(f"  Moving model from {model_device} to {target_device}")
                model._device_warning_printed = True
            model = model.to(target_device)
        
        # Apply adversarial perturbation if configured
        if self.config.data_type in ["adversarial", "mixed"]:
            inputs = self._generate_adversarial_data(model, inputs, targets)
        
        # Remove torch.no_grad() - we need gradients for LLC estimation!
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        
        # Return the loss tensor (not loss.item()) so devinterp can handle it properly
        return loss, {"logits": outputs}
    
    def _generate_adversarial_data(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Generate adversarial examples for LLC evaluation.
        
        Note: This function assumes the model is already in eval mode (as set by evaluate_model).
        The model remains in eval mode throughout adversarial generation.
        
        Args:
            model: Model to attack
            inputs: Clean inputs
            targets: True labels
            
        Returns:
            Adversarial inputs
        """
        model.eval()
        
        if self.config.adversarial_attack.lower() == "pgd":
            return self._pgd_attack(model, inputs, targets)
        elif self.config.adversarial_attack.lower() == "fgsm":
            return self._fgsm_attack(model, inputs, targets)
        else:
            raise ValueError(f"Unsupported attack type: {self.config.adversarial_attack}")
    
    def _pgd_attack(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """PGD attack implementation"""
        inputs_adv = inputs.clone().detach().requires_grad_(True)
        
        # Zero out any existing gradients on model parameters to avoid interference
        model.zero_grad()
        
        for step in range(self.config.adversarial_steps):
            # Zero out any existing gradients on inputs
            if inputs_adv.grad is not None:
                inputs_adv.grad.zero_()
            
            outputs = model(inputs_adv)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            
            # Compute gradients w.r.t. inputs only
            grad_outputs = torch.autograd.grad(
                outputs=loss,
                inputs=inputs_adv,
                create_graph=False,
                retain_graph=False,
                only_inputs=True
            )[0]
            
            with torch.no_grad():
                grad_sign = grad_outputs.sign()
                inputs_adv = inputs_adv + self.config.adversarial_eps / self.config.adversarial_steps * grad_sign
                inputs_adv = torch.clamp(inputs_adv, 0, 1)
                
                # Detach and re-enable gradients for next iteration
                inputs_adv = inputs_adv.detach().requires_grad_(True)
        
        return inputs_adv.detach()
    
    def _fgsm_attack(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """FGSM attack implementation"""
        inputs_adv = inputs.clone().detach().requires_grad_(True)
        
        # Zero out any existing gradients on model parameters to avoid interference
        model.zero_grad()
        
        outputs = model(inputs_adv)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        
        # Compute gradients w.r.t. inputs only
        grad_outputs = torch.autograd.grad(
            outputs=loss,
            inputs=inputs_adv,
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )[0]
        
        with torch.no_grad():
            grad_sign = grad_outputs.sign()
            inputs_adv = inputs_adv + self.config.adversarial_eps * grad_sign
            inputs_adv = torch.clamp(inputs_adv, 0, 1)
        
        return inputs_adv.detach()
    
    def calibrate_hyperparameters(self, 
                                model: nn.Module, 
                                train_loader: DataLoader,
                                save_path: Optional[str] = None) -> Dict[str, float]:
        """
        Calibrate SGLD hyperparameters using grid search.
        
        Args:
            model: Model to calibrate for
            train_loader: Training data loader
            save_path: Path to save calibration results
            
        Returns:
            Dictionary with optimal hyperparameters
        """
        print("Starting hyperparameter calibration...")
        # Create save directory if specified
        if save_path:
            os.makedirs(save_path, exist_ok=True)
    
        # Create calibration data loader
        calibration_loader = DataLoader(
            train_loader.dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        
        # Initialize analyzer
        analyzer = EpsilonBetaAnalyzer()
        
        # Configure sweep
        analyzer.configure_sweep(
            llc_estimator=self._estimate_llc_for_calibration,
            llc_estimator_kwargs=dict(
                model=model,
                evaluate=self.evaluate_model,
                device=self.device,
                loader=calibration_loader,
            ),
            min_epsilon=min(self.config.calibration_epsilons),
            max_epsilon=max(self.config.calibration_epsilons),
            epsilon_samples=len(self.config.calibration_epsilons),
            min_beta=None,
            max_beta=None,
            beta_samples=3,
            dataloader=calibration_loader,
        )
        
        # Run sweep
        print("Running epsilon-beta sweep...")
        analyzer.sweep()
        
        
        # Extract optimal parameters - pass the calibration_loader
        optimal_params = self._extract_optimal_params(analyzer, calibration_loader)
        
        # Estimate LLC with optimal parameters and create final trace
        if save_path and optimal_params:
            print("Estimating LLC with optimal parameters...")
            try:
                # Temporarily update config to match calibration parameters
                original_chains = self.config.num_chains
                original_steps = self.config.num_steps
                original_burnin = self.config.num_burnin_steps
                
                # Use same parameters as calibration for consistency
                self.config.num_chains = 3
                self.config.num_steps = 300 #500  # 50 draws + 450 burn-in
                self.config.num_burnin_steps = 270 #450
                
                print(f"Using calibration-consistent parameters: {self.config.num_chains} chains, {self.config.get_effective_draws()} draws")
                
                llc_results = self.estimate_llc(
                    model, 
                    train_loader, 
                    hyperparams=optimal_params
                )
                
                # Restore original config
                self.config.num_chains = original_chains
                self.config.num_steps = original_steps
                self.config.num_burnin_steps = original_burnin
                
                # Plot the final LLC estimation trace
                if 'loss/trace' in llc_results:
                    final_trace_path = os.path.join(save_path, "final_llc_trace.png")
                    self.plot_sampling_evolution(
                        llc_results, 
                        save_path=final_trace_path, 
                        show=False
                    )
                    print(f"Final LLC trace plot saved to {final_trace_path}")
                    print(f"Final trace shape: {llc_results['loss/trace'].shape}")
                else:
                    print("No trace data found in final results for plotting")
                    
            except Exception as e:
                print(f"Warning: Could not generate plot with optimal parameters: {e}")
                print("Continuing without the optimal trace plot...")
        
        # Save calibration results
        if save_path:
            calibration_results = {
                "optimal_params": optimal_params,
                "calibration_epsilons": self.config.calibration_epsilons,
                "calibration_gammas": self.config.calibration_gammas,
            }
            
            # Convert to serializable format before saving
            serializable_calibration_results = convert_to_serializable(calibration_results)
            with open(os.path.join(save_path, "calibration_results.json"), "w") as f:
                json.dump(serializable_calibration_results, f, indent=2)
        
        # Update config with calibrated values for future use
        if optimal_params:
            print(f"\n📝 Updating config with calibrated hyperparameters...")
            self.config.epsilon = optimal_params['epsilon']
            self.config.nbeta = optimal_params['nbeta']
            self.config.gamma = optimal_params['gamma'] 
            # Note: nbeta is typically calculated from dataloader, so we don't update it in config
            print(f"   Config epsilon: {self.config.epsilon:.2e}")
            print(f"   Config nbeta: {self.config.nbeta:.3f}")
            print(f"   Config gamma: {self.config.gamma:.3f}")
        
        print(f"Calibration complete. Optimal params: {optimal_params}")
        return optimal_params
    
    def _estimate_llc_for_calibration(self, 
                                    model: nn.Module,
                                    loader: DataLoader,
                                    evaluate: callable,
                                    epsilon: float,
                                    beta: float,
                                    **kwargs) -> Dict:
        """Helper function for calibration sweep - used by EpsilonBetaAnalyzer"""
        
        # Create MALA callback for calibration runs too
        mala_callback = MalaAcceptanceRate(
            num_chains=3,  # Match the number of chains used in calibration
            num_draws=50,  # Match the number of draws used in calibration
            nbeta=beta,
            learning_rate=epsilon,
            device=self.device
        )
        
        return estimate_learning_coeff_with_summary(
            model=model,
            loader=loader,
            evaluate=evaluate,
            sampling_method=SGLD,
            optimizer_kwargs=dict(
                lr=epsilon, 
                localization=self.config.gamma,  # Use fixed gamma for calibration
                nbeta=beta
            ),
            num_chains=3,  # Use fewer chains for calibration speed
            num_draws=50,  # Use fewer draws for calibration speed  # DRAWS NOT STEPS
            num_burnin_steps=450,  # 90% burn-in for calibration
            device=self.device,
            online=True,
            callbacks=[mala_callback],  # Add MALA callback for acceptance rate tracking
        )
    
    def plot_sampling_evolution(self,
        llc_stats: Dict[str, Any], 
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (12, 9)
    ) -> plt.Figure:
        """Plot LLC loss traces from LLC estimation.
        
        This recreates the original plot_sampling_evolution from utils_outdated.py
        
        Args:
            llc_stats: Dictionary containing 'loss/trace' or 'llc/trace' and optionally 'llc_average_mean'
            save_path: Path to save figure
            show: Whether to display the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Try to get trace from either 'loss/trace' or 'llc/trace'
        if 'loss/trace' in llc_stats:
            trace = llc_stats['loss/trace']
            trace_type = "Loss"
        elif 'llc/trace' in llc_stats:
            trace = llc_stats['llc/trace']
            trace_type = "LLC"
        else:
            raise ValueError("No 'loss/trace' or 'llc/trace' found in llc_stats")
        
        print(f"{trace_type} trace shape:", trace.shape)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get dimensions correctly
        num_chains, num_draws = trace.shape 
        sgld_step = list(range(num_draws))
        
        # Plot individual chains
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i in range(num_chains):
            draws = trace[i]
            color = colors[i % len(colors)]
            ax.plot(sgld_step, draws, linewidth=1.5, label=f"chain {i}", 
                    color=color, alpha=0.8)

        # Plot mean with black dashed line
        mean = np.mean(trace, axis=0)
        ax.plot(sgld_step, mean, color="black", linestyle="--", 
                linewidth=3, label="mean", zorder=3)

        # Add std shading
        std = np.std(trace, axis=0)
        ax.fill_between(sgld_step, mean - std, mean + std, 
                        color="gray", alpha=0.3, zorder=2)

        # Formatting
        # Try to get llc_average_mean, fallback to calculated mean if not available
        llc_avg = llc_stats.get('llc_average_mean', np.mean(trace))
        ax.set_title(f"{trace_type} Trace, avg LLC = {llc_avg:.2f}",
                    fontsize=16, fontweight='bold')
        ax.set_xlabel("SGLD Step", fontsize=14)
        ax.set_ylabel(trace_type, fontsize=14)
        ax.legend(loc="upper right", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        
        # Save and show
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Trace plot saved to {save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)  # Close figure if not showing to save memory
        
        return fig
    
    
    def _extract_optimal_params(self, analyzer: EpsilonBetaAnalyzer, dataloader: DataLoader, selection_method: str = "mala", use_tuned_beta: bool = True) -> Dict[str, float]:
        """
        Extract optimal parameters using the proven selection criteria from reference file.
        This is the sophisticated logic that actually works!
        """
        print("\n=== HYPERPARAMETER SELECTION ===")
        
        if analyzer.sweep_df is None or len(analyzer.sweep_df) == 0:
            print("No calibration results found, using default parameters")
            return self._get_default_params(analyzer, dataloader)
        
        print(f"Found {len(analyzer.sweep_df)} successful calibration runs")
        
        # Calculate additional statistics if not already present
        if 'llc/final' not in analyzer.sweep_df.columns:
            analyzer.sweep_df['llc/final'] = analyzer.sweep_df['llc/trace'].apply(
                lambda x: x[:, -50:].mean() if len(x.shape) == 2 else x[-50:].mean()
            )
        
        if 'llc/std_over_mean' not in analyzer.sweep_df.columns:
            analyzer.sweep_df['llc/std_over_mean'] = analyzer.sweep_df['llc/trace'].apply(
                lambda x: x[:, -50:].std() / x[:, -50:].mean() if len(x.shape) == 2 else x[-50:].std() / x[-50:].mean()
            )
        
        # Filter out NaN results
        valid_results = analyzer.sweep_df.dropna(subset=['llc/final'])
        
        if len(valid_results) == 0:
            print("No valid results found, using default parameters")
            return self._get_default_params(analyzer, dataloader)
        
        # Debug: Check for negative LLC values
        print(f"LLC value range: {valid_results['llc/final'].min():.4f} to {valid_results['llc/final'].max():.4f}")
        print(f"Stability range: {valid_results['llc/std_over_mean'].min():.4f} to {valid_results['llc/std_over_mean'].max():.4f}")
        
        # Check for negative LLC values (failure mode according to devinterp guide)
        negative_llc_count = (valid_results['llc/final'] < 0).sum()
        if negative_llc_count > 0:
            print(f"⚠️  WARNING: {negative_llc_count}/{len(valid_results)} runs have negative LLC values")
            print("   This indicates failure modes: step size too large, chain too long, or w* not near local minimum")
            print("   Consider: reducing step size, shortening chain, increasing γ, or checking model convergence")
        
        # Filter out results with NaN or extreme values
        clean_results = valid_results.dropna(subset=['llc/final', 'llc/std_over_mean'])
        clean_results = clean_results[clean_results['llc/final'].abs() < 1000]  # Remove extreme values
        
        if len(clean_results) == 0:
            print("Warning: No clean results found, using all results...")
            clean_results = valid_results
        
        # Separate positive and negative LLC results
        positive_llc = clean_results[clean_results['llc/final'] > 0]
        negative_llc = clean_results[clean_results['llc/final'] < 0]
        
        print(f"Results with positive LLC: {len(positive_llc)}")
        print(f"Results with negative LLC: {len(negative_llc)}")
        
        # Debug: Check what columns are available
        print(f"Available result columns: {list(clean_results.columns)}")
        
        # Parameter selection based on chosen method
        if selection_method == "mala":
            print("Using MALA acceptance rate-based parameter selection")
            # Check if MALA data is actually available
            if 'mala_accept/mean' in clean_results.columns:
                best_row = self._select_by_mala_acceptance(clean_results, positive_llc)
            else:
                print("⚠️  MALA selection requested but no MALA data found")
                print("   This likely means EpsilonBetaAnalyzer doesn't preserve callback results")
                print("   Falling back to stability-based selection")
                best_row = self._select_by_stability(clean_results, positive_llc)
        else:  # Default: stability-based
            print("Using stability-based parameter selection")
            best_row = self._select_by_stability(clean_results, positive_llc)
        
        # Choose beta based on use_tuned_beta flag
        if use_tuned_beta and 'beta' in best_row:
            selected_beta = float(best_row['beta'])
            beta_source = "tuned"
        else:
            selected_beta = float(default_nbeta(dataloader))
            beta_source = "default"
        
        # Get llc/stds from results if available (treat like llc/means - take mean of the array)
        if 'llc/stds' in best_row and best_row['llc/stds'] is not None:
            llc_std = float(best_row['llc/stds'].mean())
        else:
            llc_std = None
        
        # Extract MALA acceptance rate if available
        mala_acceptance = None
        if 'mala_accept/mean' in best_row:
            mala_acceptance = float(best_row['mala_accept/mean'])
        
        best_params = {
            'epsilon': float(best_row['epsilon']),
            'gamma': float(self.config.gamma),  # Always from config (not tuned)
            'nbeta': selected_beta,  
            'llc_mean': float(best_row['llc/final']), 
            'llc_std': llc_std,
            'llc_std_over_mean': float(best_row['llc/std_over_mean']),
            'beta_source': beta_source,
            'gamma_source': 'fixed_config',  # Make it clear gamma is not tuned
            'mala_acceptance': mala_acceptance  # Include MALA acceptance rate
        }
        
        print(f"\nSelected parameters ({selection_method}-based selection):")
        print(f"  ε = {best_params['epsilon']:.2e} (tuned via EpsilonBetaAnalyzer)")
        print(f"  γ = {best_params['gamma']:.3f} (FIXED - not tuned)")
        print(f"  β = {best_params['nbeta']:.3f} ({beta_source})")
        print(f"  LLC mean = {best_params['llc_mean']:.4f}")
        print(f"  LLC std/mean = {best_params['llc_std_over_mean']:.4f}")
        if 'llc_std' in best_params and best_params['llc_std'] is not None:
            print(f"  LLC std = {best_params['llc_std']:.4f}")
        if best_params['mala_acceptance'] is not None:
            print(f"  MALA acceptance = {best_params['mala_acceptance']:.3f}")
        
        if best_params['llc_mean'] < 0:
            print(f"  ⚠️  WARNING: Negative LLC selected - may need parameter adjustment")
        
        # Also print top 5 results for comparison
        print(f"\nTop 5 parameter combinations (by stability):")
        clean_results['abs_stability'] = clean_results['llc/std_over_mean'].abs()
        top_5 = clean_results.nsmallest(5, 'abs_stability')
        for i, (_, row) in enumerate(top_5.iterrows()):
            llc_sign = "⚠️" if row['llc/final'] < 0 else "✓"
            print(f"  {i+1}. {llc_sign} ε={row['epsilon']:.2e}, LLC={row['llc/final']:.4f}, stability={row['llc/std_over_mean']:.4f}")
        
        print("=" * 50)
        return best_params
    
    def _select_by_stability(self, clean_results, positive_llc):
        """Select parameters based on stability (std/mean ratio)"""
        # Prefer positive LLC with good stability
        if len(positive_llc) > 0:
            # Find the most stable positive LLC result
            best_stability_idx = positive_llc['llc/std_over_mean'].abs().idxmin()
            best_row = positive_llc.loc[best_stability_idx]
            print("✓ Using positive LLC result with best stability")
        else:
            # Fall back to best stability among all results
            best_stability_idx = clean_results['llc/std_over_mean'].abs().idxmin()
            best_row = clean_results.loc[best_stability_idx]
            print("⚠️  No positive LLC found, using best stability among all results")
        return best_row
    
    def _select_by_mala_acceptance(self, clean_results, positive_llc):
        """Select parameters based on MALA acceptance rate (assumes data is available)"""
        print(f"✓ MALA acceptance data found in calibration results")
        mala_rates = clean_results['mala_accept/mean']
        print(f"  MALA acceptance range: {mala_rates.min():.3f} - {mala_rates.max():.3f}")
        
        # Target acceptance rate: 0.90-0.95 is ideal
        target_acceptance = 0.92
        
        # Calculate distance from target acceptance rate
        clean_results['mala_distance'] = (clean_results['mala_accept/mean'] - target_acceptance).abs()
        
        # Prefer positive LLC results
        if len(positive_llc) > 0:
            # Update positive_llc to include the new mala_distance column
            positive_llc_with_distance = clean_results[clean_results['llc/final'] > 0].copy()
            # Among positive LLC, find closest to target MALA acceptance
            best_mala_idx = positive_llc_with_distance['mala_distance'].idxmin()
            best_row = positive_llc_with_distance.loc[best_mala_idx]
            acceptance_rate = best_row['mala_accept/mean']
            print(f"✓ Selected positive LLC with MALA acceptance = {acceptance_rate:.3f} (target: {target_acceptance:.3f})")
        else:
            # Fall back to best MALA acceptance among all results
            best_mala_idx = clean_results['mala_distance'].idxmin()
            best_row = clean_results.loc[best_mala_idx]
            acceptance_rate = best_row['mala_accept/mean']
            print(f"⚠️  No positive LLC found, using best MALA acceptance = {acceptance_rate:.3f}")
        
        return best_row
    
    def _get_default_params(self, analyzer: EpsilonBetaAnalyzer, dataloader: DataLoader) -> Dict[str, float]:
        """Get default parameters when calibration fails"""
        return {
            "epsilon": float(np.mean(self.config.calibration_epsilons)),  # Convert to Python float
            "gamma": float(self.config.gamma),  # Convert to Python float
            "nbeta": float(default_nbeta(dataloader))  # Convert to Python float
        }

    def load_calibrated_hyperparameters(self, calibration_path: str) -> Dict[str, float]:
        """
        Load pre-calibrated hyperparameters from a previous calibration run.
        
        Args:
            calibration_path: Path to the calibration results JSON file
            
        Returns:
            Dictionary with calibrated hyperparameters
        """
        try:
            print(f"Attempting to load calibration from: {calibration_path}")
            
            # Check if file exists
            if not os.path.exists(calibration_path):
                print(f"File does not exist: {calibration_path}")
                return None
            
            # Read file content first
            with open(calibration_path, 'r') as f:
                content = f.read()
            
            print(f"File size: {len(content)} characters")
            print(f"First 100 characters: {repr(content[:100])}")
            
            # Try to parse JSON
            calibration_results = json.loads(content)
            
            if 'optimal_params' in calibration_results:
                optimal_params = calibration_results['optimal_params']
                print(f"✓ Successfully loaded pre-calibrated hyperparameters from {calibration_path}")
                print(f"  ε = {optimal_params['epsilon']:.2e}")
                print(f"  γ = {optimal_params['gamma']:.3f}")
                print(f"  β = {optimal_params['nbeta']:.3f}")
                if 'llc_mean' in optimal_params:
                    print(f"  Previous LLC = {optimal_params['llc_mean']:.4f}")
                return optimal_params
            else:
                print("No 'optimal_params' found in calibration results")
                print(f"Available keys: {list(calibration_results.keys())}")
                return None
                
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Error at line {e.lineno}, column {e.colno}")
            # Try to show the problematic line
            try:
                with open(calibration_path, 'r') as f:
                    lines = f.readlines()
                    if e.lineno <= len(lines):
                        print(f"Problematic line {e.lineno}: {repr(lines[e.lineno-1])}")
            except:
                pass
            return None
        except Exception as e:
            print(f"Error loading calibrated hyperparameters: {e}")
            print(f"Error type: {type(e).__name__}")
            return None
    
    def estimate_llc(self, 
                    model: nn.Module, 
                    train_loader: DataLoader,
                    hyperparams: Optional[Dict[str, float]] = None,
                    run_diagnostics: bool = True) -> Dict[str, Any]:
        """
        Estimate LLC for a single model.
        
        Args:
            model: Model to estimate LLC for
            train_loader: Training data loader
            hyperparams: Optional hyperparameters (uses config defaults if None)
            run_diagnostics: Whether to run diagnostic checks
            
        Returns:
            Dictionary containing LLC estimates and diagnostics
        """
        # Validate configuration for this model
        self.config.validate_for_model(model)
        
        if hyperparams is None:
            hyperparams = {
                "epsilon": self.config.epsilon,
                "gamma": self.config.gamma,
                "nbeta": self.config.nbeta or default_nbeta(train_loader)
            }
        
        print(f"Estimating LLC with hyperparams: {hyperparams}")
        print(f"  Model device: {next(model.parameters()).device}")
        print(f"  Target device: {self.device}")
        
        # Move model to correct device if needed (only once)
        model_device = next(model.parameters()).device
        target_device = torch.device(self.device)
        
        if model_device != target_device:
            print(f"  Moving model to {self.device}")
            model = model.to(target_device)
        
        # Prepare data loader
        llc_loader = DataLoader(
            train_loader.dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        
        # Setup callbacks for diagnostics
        callbacks = []
        if run_diagnostics:
            mala_estimator = MalaAcceptanceRate(
                num_chains=self.config.num_chains,
                num_draws=self.config.get_effective_draws(),  # Use effective draws
                nbeta=hyperparams["nbeta"],
                learning_rate=hyperparams["epsilon"],
                device=self.device,
            )
            callbacks.append(mala_estimator)
        
        print(f"  Starting LLC estimation with {self.config.num_chains} chains, {self.config.get_effective_draws()} draws")
        
        try:
            # Run LLC estimation
            results = estimate_learning_coeff_with_summary(
                model=model,
                loader=llc_loader,
                evaluate=self.evaluate_model,
                sampling_method=SGLD,
                optimizer_kwargs=dict(
                    lr=hyperparams["epsilon"],
                    localization=hyperparams["gamma"],
                    nbeta=hyperparams["nbeta"],
                ),
                num_chains=self.config.num_chains,
                num_draws=self.config.get_effective_draws(),  # Use effective draws
                num_burnin_steps=self.config.num_burnin_steps,
                num_steps_bw_draws=self.config.num_steps_bw_draws,
                callbacks=callbacks,
                device=self.device,
                online=True,
            )
            
            print(f"  LLC estimation completed successfully")
            print(f"  Results keys: {list(results.keys())}")
            
            # Add diagnostic information
            if run_diagnostics and callbacks:
                for callback in callbacks:
                    if hasattr(callback, "get_results"):
                        callback_results = callback.get_results()
                        print(f"  Callback results keys: {list(callback_results.keys())}")
                        results.update(callback_results)
        
            # Add llc/mean and llc/std for compatibility (these are the mean/std across all chains)
            if 'llc/means' in results:
                results['llc/mean'] = float(results['llc/means'].mean())
                results['llc/std'] = float(results['llc/means'].std())
            
            return results
            
        except Exception as e:
            print(f"  ERROR in LLC estimation: {e}")
            print(f"  Error type: {type(e).__name__}")
            import traceback
            print(f"  Full traceback:")
            traceback.print_exc()
            raise
    
    def measure_llc_trajectory(self, 
                             model_checkpoints: List[nn.Module],
                             train_loader: DataLoader,
                             checkpoint_names: Optional[List[str]] = None,
                             hyperparams: Optional[Dict[str, float]] = None,
                             save_path: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Measure LLC across multiple model checkpoints.
        
        Args:
            model_checkpoints: List of model checkpoints
            train_loader: Training data loader
            checkpoint_names: Optional names for checkpoints
            hyperparams: Optional hyperparameters
            save_path: Path to save results
            
        Returns:
            Dictionary with LLC trajectory data
        """
        # Create save directory if specified
        if save_path:
            os.makedirs(save_path, exist_ok=True)
        
        if checkpoint_names is None:
            checkpoint_names = [f"checkpoint_{i}" for i in range(len(model_checkpoints))]
        
        print(f"Measuring LLC trajectory across {len(model_checkpoints)} checkpoints...")
        
        llc_means = []
        llc_stds = []
        mala_rates = []
        
        for i, (model, name) in enumerate(zip(model_checkpoints, checkpoint_names)):
            print(f"Processing checkpoint {i+1}/{len(model_checkpoints)}: {name}")
            
            try:
                # Add some debugging information
                print(f"  Model device: {next(model.parameters()).device}")
                print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
                
                results = self.estimate_llc(model, train_loader, hyperparams)
                
                # Debug: Check what keys are in results
                print(f"  LLC estimation completed. Available keys: {list(results.keys())}")
                
                if "llc/mean" not in results:
                    print(f"  ERROR: 'llc/mean' not found in results!")
                    print(f"  Results content: {results}")
                    raise KeyError(f"'llc/mean' not found in LLC results for {name}")
                
                llc_means.append(results["llc/mean"])
                llc_stds.append(results["llc/std"])
                
                # Extract MALA acceptance rate if available
                if "mala_accept/mean" in results:
                    mala_rates.append(results["mala_accept/mean"])
                else:
                    mala_rates.append(None)
                
                # Save individual checkpoint results
                if save_path:
                    checkpoint_path = os.path.join(save_path, f"{name}_llc_results.json")
                    checkpoint_results = {
                        "llc_mean": results["llc/mean"],
                        "llc_std": results["llc/std"],
                        "hyperparams": hyperparams or {
                            "epsilon": self.config.epsilon,
                            "gamma": self.config.gamma,
                            "nbeta": self.config.nbeta
                        }
                    }
                    if "mala_accept/mean" in results:
                        checkpoint_results["mala_accept_rate"] = results["mala_accept/mean"]
                    
                    # Convert to serializable format before saving
                    serializable_checkpoint_results = convert_to_serializable(checkpoint_results)
                    with open(checkpoint_path, "w") as f:
                        json.dump(serializable_checkpoint_results, f, indent=2)
                
            except Exception as e:
                print(f"Error processing checkpoint {name}: {e}")
                print(f"  Error type: {type(e).__name__}")
                import traceback
                print(f"  Full traceback:")
                traceback.print_exc()
                llc_means.append(None)
                llc_stds.append(None)
                mala_rates.append(None)
        
        trajectory_data = {
            "checkpoint_names": checkpoint_names,
            "llc_means": llc_means,
            "llc_stds": llc_stds,
            "mala_rates": mala_rates,
            "hyperparams": hyperparams
        }
        
        # Save trajectory results
        if save_path:
            trajectory_path = os.path.join(save_path, "llc_trajectory.json")
            # Convert to serializable format before saving
            serializable_trajectory_data = convert_to_serializable(trajectory_data)
            with open(trajectory_path, "w") as f:
                json.dump(serializable_trajectory_data, f, indent=2)
            
            # Create trajectory plot
            self._plot_llc_trajectory(trajectory_data, save_path)
        
        return trajectory_data
    
    def _plot_llc_trajectory(self, trajectory_data: Dict, save_path: str):
        """Plot LLC trajectory"""
        checkpoint_names = trajectory_data["checkpoint_names"]
        llc_means = trajectory_data["llc_means"]
        llc_stds = trajectory_data["llc_stds"]
        
        # Filter out None values
        valid_indices = [i for i, mean in enumerate(llc_means) if mean is not None]
        valid_means = [llc_means[i] for i in valid_indices]
        valid_stds = [llc_stds[i] for i in valid_indices]
        valid_names = [checkpoint_names[i] for i in valid_indices]
        
        if not valid_means:
            print("No valid LLC measurements to plot")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot LLC means with error bars
        x_positions = range(len(valid_means))
        plt.errorbar(x_positions, valid_means, yerr=valid_stds, 
                    marker='o', capsize=5, capthick=2)
        
        plt.xlabel("Checkpoint")
        plt.ylabel("Local Learning Coefficient (LLC)")
        plt.title("LLC Trajectory Throughout Training")
        plt.xticks(x_positions, valid_names, rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_path, "llc_trajectory.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_diagnostics(self, 
                       model: nn.Module, 
                       train_loader: DataLoader,
                       hyperparams: Optional[Dict[str, float]] = None,
                       save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive diagnostics for LLC estimation.
        
        Args:
            model: Model to diagnose
            train_loader: Training data loader
            hyperparams: Optional hyperparameters
            save_path: Path to save diagnostic plots
            
        Returns:
            Dictionary with diagnostic results
        """
        print("Running LLC diagnostics...")
        
        # Create save directory if specified
        if save_path:
            os.makedirs(save_path, exist_ok=True)
        
        if hyperparams is None:
            hyperparams = {
                "epsilon": self.config.epsilon,
                "gamma": self.config.gamma,
                "nbeta": self.config.nbeta or default_nbeta(train_loader)
            }
        
        # Run LLC estimation with diagnostics
        results = self.estimate_llc(model, train_loader, hyperparams, run_diagnostics=True)
        
        # Create diagnostic plots
        if save_path:
            self._create_diagnostic_plots(results, save_path)
        
        # Print diagnostic summary
        self._print_diagnostic_summary(results)
        
        return results
    
    def _create_diagnostic_plots(self, results: Dict[str, Any], save_path: str):
        """Create diagnostic plots"""
        # Loss trace plot
        if "loss/trace" in results:
            try:
                # Try to use devinterp's plot_trace first
                plt.figure(figsize=(12, 8))
                
                # Get LLC mean for title, handling both key formats
                llc_mean = None
                if 'llc/mean' in results:
                    llc_mean = results['llc/mean']
                elif 'llc/means' in results:
                    llc_mean = float(results['llc/means'].mean())
                
                title = f"Loss Trace, avg LLC = {llc_mean:.2f}" if llc_mean is not None else "Loss Trace"
                
                plot_trace(
                    results["loss/trace"],
                    "Loss",
                    x_axis="Step",
                    title=title,
                    plot_mean=False,
                    plot_std=False,
                    fig_size=(12, 8),
                    true_lc=None,
                )
                plt.savefig(os.path.join(save_path, "loss_trace.png"), dpi=300, bbox_inches='tight')
                plt.close()
                print("Loss trace plot created using devinterp plot_trace")
            except Exception as e:
                print(f"Warning: devinterp plot_trace failed: {e}")
                print("Falling back to custom plotting function...")
                
                # Fallback to our custom plotting function
                try:
                    self.plot_sampling_evolution(
                        results,
                        save_path=os.path.join(save_path, "loss_trace.png"),
                        show=False,
                        figsize=(12, 8)
                    )
                    print("Loss trace plot created using custom plotting function")
                except Exception as e2:
                    print(f"Error: Custom plotting also failed: {e2}")
                    print("Skipping loss trace plot")
        
        # MALA acceptance rate plot
        if "mala_accept/trace" in results:
            try:
                # Try to use devinterp's plot_trace first
                plt.figure(figsize=(12, 8))
                plot_trace(
                    results["mala_accept/trace"],
                    "MALA Acceptance Rate",
                    x_axis="Step",
                    title="MALA Acceptance Rate Trace",
                    plot_mean=True,
                    plot_std=True,
                    fig_size=(12, 8),
                )
                plt.savefig(os.path.join(save_path, "mala_acceptance_trace.png"), dpi=300, bbox_inches='tight')
                plt.close()
                print("MALA acceptance rate plot created using devinterp plot_trace")
            except Exception as e:
                print(f"Warning: devinterp plot_trace failed for MALA: {e}")
                print("Skipping MALA acceptance rate plot")
    
    def _print_diagnostic_summary(self, results: Dict[str, Any]):
        """Print comprehensive diagnostic summary"""
        print("\n=== LLC Diagnostic Summary ===")
        
        # Get LLC mean and std, handling both key formats
        llc_mean = None
        llc_std = None
        
        if 'llc/mean' in results:
            llc_mean = results['llc/mean']
            llc_std = results['llc/std']
        elif 'llc/means' in results:
            llc_mean = float(results['llc/means'].mean())
            llc_std = float(results['llc/means'].std())
        
        if llc_mean is not None and llc_std is not None:
            print(f"LLC Mean: {llc_mean:.4f}")
            print(f"LLC Std: {llc_std:.4f}")
            
            # Calculate stability metric
            if llc_std != 0 and llc_mean != 0:
                stability = abs(llc_std / llc_mean)
                print(f"Stability (std/mean): {stability:.4f}")
                
                if stability < 0.1:
                    print("✅ Excellent stability")
                elif stability < 0.2:
                    print("✅ Good stability")
                elif stability < 0.5:
                    print("⚠️  Moderate stability - consider more chains/draws")
                else:
                    print("❌ Poor stability - check hyperparameters")
        else:
            print("❌ Could not extract LLC mean/std from results")
        
        # MALA acceptance rate analysis
        if "mala_accept/mean" in results:
            mala_rate = results["mala_accept/mean"]
            print(f"MALA Acceptance Rate: {mala_rate:.4f}")
            
            if mala_rate < 0.5:
                print("❌ Low MALA acceptance rate (< 0.5)")
                print("   → Step size too large, reduce epsilon")
            elif mala_rate > 0.95:
                print("⚠️  Very high MALA acceptance rate (> 0.95)")
                print("   → Step size too small, increase epsilon")
            else:
                print("✅ MALA acceptance rate in good range (0.5-0.95)")
        
        # LLC value analysis
        if llc_mean is not None:
            if llc_mean < 0:
                print("❌ Negative LLC detected")
                print("   → Possible causes: step size too large, model not converged")
                print("   → Solutions: reduce epsilon, increase gamma, check model training")
            elif llc_mean > 100:
                print("⚠️  Very high LLC detected (> 100)")
                print("   → May indicate numerical instability or extreme complexity")
            else:
                print("✅ LLC is positive and reasonable")
        
        print("=" * 50)
        
        # Overall assessment
        print("\n=== Overall Assessment ===")
        issues = []
        
        if llc_mean is not None and llc_mean < 0:
            issues.append("Negative LLC")
        if "mala_accept/mean" in results and (results["mala_accept/mean"] < 0.5 or results["mala_accept/mean"] > 0.95):
            issues.append("Poor MALA acceptance rate")
        if "llc/trace" in results and len(results["llc/trace"].shape) == 2:
            trace = results["llc/trace"]
            final_portion = trace[:, -50:]
            early_portion = trace[:, :50]
            final_mean = final_portion.mean()
            early_mean = early_portion.mean()
            convergence_ratio = abs(final_mean - early_mean) / abs(early_mean) if early_mean != 0 else float('inf')
            if convergence_ratio > 0.3:
                issues.append("Poor convergence")
        
        if not issues:
            print("✅ All diagnostics passed - LLC measurement appears reliable")
        else:
            print(f"⚠️  Issues detected: {', '.join(issues)}")
            print("   Consider adjusting hyperparameters or increasing sampling budget")
        
        print("=" * 50)


# def main_llc_measurement_example():
#     """
#     Example usage of the LLC measurement module.
#     """
#     # Example configuration
#     config = LLCConfig(
#         epsilon=1e-4,
#         gamma=100.0,
#         num_chains=8,
#         num_draws=2000,
#         batch_size=512
#     )
    
#     # Initialize measurer
#     measurer = LLCMeasurer(config)
    
#     print("LLC Measurement Module initialized successfully!")
#     print("Use this module to measure LLC for your adversarial training models.")
#     print("\nKey methods:")
#     print("- calibrate_hyperparameters(): Tune SGLD hyperparameters")
#     print("- estimate_llc(): Measure LLC for a single model")
#     print("- measure_llc_trajectory(): Track LLC across training")
#     print("- run_diagnostics(): Comprehensive diagnostic checks")


# if __name__ == "__main__":
#     main_llc_measurement_example() 