# Local Learning Coefficient (LLC) Measurement for Adversarial Training

This document provides comprehensive documentation for the Local Learning Coefficient (LLC) measurement system, which analyzes model complexity evolution during adversarial training using the devinterp library and advanced SGLD sampling techniques.

## 🎯 Overview

The Local Learning Coefficient (LLC) is a **singularity-aware complexity measure** that quantifies the degeneracy of the loss landscape around a given parameter configuration. This implementation enables:

1. **📊 LLC Measurement for Pre-trained Models** - Analyze model complexity after training
2. **📈 LLC Trajectory Analysis** - Track complexity evolution across training checkpoints
3. **🛡️ Defense Method Comparison** - Compare how different adversarial training methods affect complexity
4. **🔧 Advanced Hyperparameter Calibration** - Stability-based parameter selection with comprehensive diagnostics
5. **🔍 Diagnostic Monitoring** - Monitor sampling quality, convergence, and detect issues
6. **⚔️ Clean vs Adversarial LLC** - Compare complexity under different data conditions
7. **🎯 Cross-Attack Analysis** - Analyze LLC signatures across different attack types

## 🚀 Quick Start

### Installation

```bash
# Install core dependencies
pip install -r requirements_llc.txt

# Install devinterp library
pip install devinterp

# Verify installation
python -c "import devinterp; print('✅ devinterp installed successfully')"
```

### Basic LLC Measurement

```python
from llc_measurement import LLCMeasurer, LLCConfig
from AT_replication_complete import create_model_and_config, setup_cifar10_data

# Create model and data
model, config = create_model_and_config("ResNet18")
train_loader, _, _, _ = setup_cifar10_data(batch_size=512)

# Configure LLC measurement
llc_config = LLCConfig(
    epsilon=1e-4,          # SGLD step size
    gamma=100.0,           # Localization strength
    num_chains=3,          # Number of SGLD chains
    num_draws=500,        # Samples per chain
    batch_size=512         # SGLD batch size
)

# Initialize measurer and estimate LLC
measurer = LLCMeasurer(llc_config)
results = measurer.estimate_llc(model, train_loader)

print(f"LLC: {results['llc/mean']:.4f} ± {results['llc/std']:.4f}")
```

### Command Line Interface

```bash
# Single model analysis
python llc_analysis_pipeline.py --mode single \
    --model_path ./models/ResNet18_AT/best.pth \
    --model_name ResNet18 \
    --dataset CIFAR10 \
    --defense_method AT

# Trajectory analysis across checkpoints
python llc_analysis_pipeline.py --mode trajectory \
    --checkpoint_dir ./models/ResNet18_AT/epoch_iter/ \
    --model_name ResNet18 \
    --dataset CIFAR10 \
    --defense_method AT

# Clean vs adversarial comparison
python llc_analysis_pipeline.py --mode clean_vs_adv \
    --checkpoint_dir ./models/ResNet18_AT/epoch_iter/ \
    --model_name ResNet18 \
    --dataset CIFAR10 \
    --defense_method AT \
    --adversarial_attack pgd \
    --adversarial_eps 8/255
```

## 🏗️ Architecture & Components

### 1. LLCConfig - Configuration Management

```python
@dataclass
class LLCConfig:
    # SGLD Hyperparameters
    epsilon: float = 1e-4              # Step size (most critical parameter)
    gamma: float = 100.0               # Localization strength
    nbeta: Optional[float] = None      # Inverse temperature (auto-calculated)
    
    # Sampling Parameters
    num_chains: int = 8                # Number of parallel SGLD chains
    num_draws: int = 2000              # Samples per chain
    num_burnin_steps: int = 0          # Burn-in steps (usually 0)
    num_steps_bw_draws: int = 1        # Steps between draws
    
    # Data Configuration
    batch_size: int = 512              # SGLD batch size
    data_type: str = "clean"           # "clean", "adversarial", or "mixed"
    
    # Adversarial Settings (for adversarial data_type)
    adversarial_attack: str = "pgd"    # "pgd" or "fgsm"
    adversarial_eps: float = 8/255     # Attack strength
    adversarial_steps: int = 10        # Attack steps
    
    # Calibration Parameters
    calibration_epsilons: List[float] = field(default_factory=lambda: [1e-5, 1e-4, 1e-3])
    calibration_gammas: List[float] = field(default_factory=lambda: [1.0, 10.0, 100.0])
    
    # System Configuration
    device: str = "cuda"               # "cuda" or "cpu"
    verbose: bool = True               # Enable verbose output
```

### 2. LLCMeasurer - Core Measurement Engine

```python
class LLCMeasurer:
    """Main class for LLC measurement operations"""
    
    def __init__(self, config: LLCConfig):
        self.config = config
        self.device = config.device
        self.results_cache = {}
    
    # Core Methods
    def estimate_llc(self, model, data_loader, hyperparams=None) -> Dict
    def calibrate_hyperparameters(self, model, data_loader, save_path=None) -> Dict
    def measure_llc_trajectory(self, model_checkpoints, data_loader, save_path=None) -> Dict
    def run_diagnostics(self, model, data_loader, hyperparams=None) -> Dict
```

**Key Methods:**
- **`estimate_llc()`** - Measure LLC for a single model state
- **`calibrate_hyperparameters()`** - Advanced stability-based parameter tuning
- **`measure_llc_trajectory()`** - Track LLC across multiple checkpoints
- **`run_diagnostics()`** - Comprehensive quality assessment

### 3. LLCAnalysisPipeline - High-Level Interface

```python
class LLCAnalysisPipeline:
    """Comprehensive analysis pipeline for pre-trained models"""
    
    def __init__(self, llc_config: LLCConfig, base_save_dir: str = "./llc_analysis"):
        self.llc_config = llc_config
        self.measurer = LLCMeasurer(llc_config)
        self.results_dir = self._create_timestamped_dir()
    
    # Analysis Methods
    def analyze_single_model(self, model_path, model_name, dataset_name, defense_method)
    def analyze_checkpoint_trajectory(self, checkpoint_dir, model_name, dataset_name, defense_method)
    def compare_defense_methods(self, model_configs, dataset_name)
    def compare_clean_vs_adversarial_llc(self, checkpoint_dir, model_name, dataset_name, defense_method)
    def generate_summary_report(self, results)
```

## 🔧 Advanced Hyperparameter Calibration

The LLC measurement system uses a **sophisticated stability-based calibration approach**:

### Calibration Process

1. **📊 Grid Search** - Systematic sweep across epsilon and gamma values
2. **🎯 Stability Analysis** - Calculate coefficient of variation (std/mean) for each combination
3. **🔍 Quality Filtering** - Remove NaN, extreme values, and negative LLC results
4. **⭐ Optimal Selection** - Choose parameters with best stability among valid results
5. **📋 Comprehensive Reporting** - Detailed analysis with top 5 parameter combinations

### Example Calibration

```python
# Automatic calibration
optimal_params = measurer.calibrate_hyperparameters(
    model=model,
    data_loader=train_loader,
    save_path="./calibration_results"
)

# Use calibrated parameters
results = measurer.estimate_llc(
    model=model,
    data_loader=train_loader,
    hyperparams=optimal_params
)
```

### Calibration Output

```
📊 Calibration Results:
✅ Optimal Parameters:
   - Epsilon: 1e-4
   - Gamma: 100.0  
   - Beta: 0.0012
   - Stability: 0.085 (excellent)
   - LLC: 45.32 ± 3.87

📈 Top 5 Parameter Combinations:
1. eps=1e-4, gamma=100.0 → LLC=45.32±3.87, stability=0.085
2. eps=5e-5, gamma=100.0 → LLC=44.91±4.12, stability=0.092
3. eps=1e-4, gamma=50.0  → LLC=46.15±4.89, stability=0.106
4. eps=2e-4, gamma=100.0 → LLC=43.78±5.23, stability=0.119
5. eps=1e-4, gamma=200.0 → LLC=44.65±5.67, stability=0.127
```

## 📊 Data Types for LLC Evaluation

### 1. Clean Data (Default)
```python
config = LLCConfig(data_type="clean")
```
- Evaluates LLC on original, unperturbed training data
- Standard approach for measuring intrinsic model complexity

### 2. Adversarial Data
```python
config = LLCConfig(
    data_type="adversarial",
    adversarial_attack="pgd",      # "pgd" or "fgsm"
    adversarial_eps=8/255,         # L∞ perturbation budget
    adversarial_steps=10           # PGD steps
)
```
- Evaluates LLC on adversarially perturbed data
- Measures complexity under adversarial conditions
- Supports PGD and FGSM attacks with configurable parameters

### 3. Mixed Data
```python
config = LLCConfig(data_type="mixed")
```
- Evaluates LLC on 50/50 mix of clean and adversarial data
- Provides balanced complexity assessment

## 🎯 Analysis Modes

### Mode 1: Single Model Analysis
```bash
python llc_analysis_pipeline.py --mode single \
    --model_path ./models/ResNet18_AT/best.pth \
    --model_name ResNet18 \
    --dataset CIFAR10 \
    --defense_method AT
```
**Output:**
- LLC measurement with calibration
- Diagnostic plots (MALA acceptance, loss traces)
- Comprehensive analysis report

### Mode 2: Trajectory Analysis
```bash
python llc_analysis_pipeline.py --mode trajectory \
    --checkpoint_dir ./models/ResNet18_AT/epoch_iter/ \
    --model_name ResNet18 \
    --dataset CIFAR10 \
    --defense_method AT \
    --max_checkpoints 20
```
**Output:**
- LLC evolution across training epochs
- Trajectory visualization with trend analysis
- Checkpoint-by-checkpoint detailed results

### Mode 3: Defense Method Comparison
```bash
# Create configuration file
cat > model_comparison.json << EOF
[
  ["./models/ResNet18_Standard/best.pth", "ResNet18", "Standard", "Standard Training"],
  ["./models/ResNet18_AT/best.pth", "ResNet18", "AT", "Adversarial Training"],
  ["./models/ResNet18_TRADES/best.pth", "ResNet18", "TRADES", "TRADES Defense"],
  ["./models/ResNet18_MART/best.pth", "ResNet18", "MART", "MART Defense"]
]
EOF

python llc_analysis_pipeline.py --mode compare \
    --config_file model_comparison.json \
    --dataset CIFAR10
```
**Output:**
- Side-by-side LLC comparison
- Statistical significance testing
- Comparative visualization

### Mode 4: Clean vs Adversarial Comparison
```bash
python llc_analysis_pipeline.py --mode clean_vs_adv \
    --checkpoint_dir ./models/ResNet18_AT/epoch_iter/ \
    --model_name ResNet18 \
    --dataset CIFAR10 \
    --defense_method AT \
    --adversarial_attack pgd \
    --adversarial_eps 8/255 \
    --adversarial_steps 10
```
**Output:**
- Clean vs adversarial LLC trajectories
- Comparative analysis and divergence detection
- Data-type specific complexity insights

## 🔍 Diagnostic Tools & Quality Control

### 1. MALA Acceptance Rate Monitoring
```python
# Target acceptance rate: 0.9-0.95
acceptance_rate = results['diagnostics']['mala_acceptance_rate']

if acceptance_rate < 0.5:
    print("⚠️  Step size too large - reduce epsilon")
elif acceptance_rate > 0.95:
    print("⚠️  Step size too small - increase epsilon")
else:
    print("✅ Good acceptance rate")
```

### 2. Stability Metrics
```python
stability = results['llc/std'] / results['llc/mean']

if stability < 0.1:
    print("✅ Excellent stability")
elif stability < 0.2:
    print("✅ Good stability")  
elif stability < 0.5:
    print("⚠️  Moderate stability - consider more chains/draws")
else:
    print("❌ Poor stability - check hyperparameters")
```

### 3. Convergence Analysis
- **Early vs Late Comparison** - Compare first and second half of traces
- **Chain Consistency** - Agreement between different SGLD chains
- **Trace Visualization** - Visual inspection of LLC and loss traces

### 4. Failure Mode Detection
- **Negative LLC Values** - Indicates step size too large or model issues
- **Extreme Values** - LLC > 1000 may indicate numerical instability
- **NaN/Inf Values** - Gradient explosion or numerical issues

## 📈 Output Structure & Results

### Directory Organization
```
llc_analysis/
└── llc_analysis_20240101_120000/
    ├── ResNet18_AT_single/
    │   ├── calibration/
    │   │   ├── calibration_sweep.png
    │   │   ├── calibration_sweep_normalized.png
    │   │   ├── calibration_results.json
    │   │   └── llc_calibration_detailed_results.csv
    │   ├── diagnostics/
    │   │   ├── loss_trace.png
    │   │   ├── mala_acceptance_trace.png
    │   │   └── llc_trace.png
    │   └── analysis_results.json
    │
    ├── ResNet18_AT_trajectory/
    │   ├── calibration/
    │   ├── llc_results/
    │   │   ├── llc_trajectory.png
    │   │   ├── llc_trajectory_enhanced.png
    │   │   ├── llc_trajectory.json
    │   │   └── checkpoint_*_llc_results.json
    │   └── trajectory_results.json
    │
    ├── ResNet18_AT_clean_vs_adv/
    │   ├── clean_vs_adversarial_llc.png
    │   ├── clean_vs_adv_comparison.json
    │   └── divergence_analysis.json
    │
    └── analysis_summary.txt
```

### Key Output Files

1. **`analysis_results.json`** - Complete numerical results
2. **`calibration_results.json`** - Optimal hyperparameters
3. **`llc_calibration_detailed_results.csv`** - Full calibration grid
4. **`llc_trajectory.json`** - Trajectory data for plotting
5. **`analysis_summary.txt`** - Human-readable summary report

## ⚔️ Cross-Attack Analysis

Analyze LLC signatures across different attack types to investigate whether LLC can distinguish between different adversarial perturbations.

### Usage
```bash
python cross_attack_llc_analysis.py \
    --model_path ./models/LeNet_AT/best.pth \
    --model_name LeNet \
    --dataset MNIST \
    --calibration_path ./calibration_results.json \
    --checkpoint_dir ./models/LeNet_AT/epoch_iter/ \
    --max_checkpoints 10 \
    --output_dir ./cross_attack_analysis
```

### Supported Attack Types
- **L∞ Attacks**: PGD-L∞, FGSM-L∞
- **L2 Attacks**: PGD-L2
- **L1 Attacks**: PGD-L1
- **Clean**: No perturbation baseline

### Research Questions
- Do different attack norms produce distinct LLC signatures?
- How do LLC trajectories vary across attack types during training?
- Can LLC be used as a robustness indicator for specific attack types?

## 🛠️ Advanced Usage Patterns

### Custom Model Integration
```python
from llc_measurement import LLCMeasurer, LLCConfig

# Load your custom model
model = YourCustomModel()
model.load_state_dict(torch.load('your_model.pth'))

# Setup data loader
data_loader = YourDataLoader(batch_size=512)

# Configure and measure LLC
config = LLCConfig(epsilon=1e-4, gamma=100.0, num_chains=8)
measurer = LLCMeasurer(config)
results = measurer.estimate_llc(model, data_loader)
```

### Batch Processing Multiple Models
```python
from llc_analysis_pipeline import LLCAnalysisPipeline

# Define model configurations
models = [
    ("./models/model1.pth", "ResNet18", "AT", "Model 1"),
    ("./models/model2.pth", "ResNet18", "TRADES", "Model 2"),
    ("./models/model3.pth", "ResNet18", "MART", "Model 3"),
]

# Run batch analysis
pipeline = LLCAnalysisPipeline(LLCConfig())
results = pipeline.compare_defense_methods(models, "CIFAR10")
```

### Skip Calibration for Speed
```bash
# First run: calibrate and save results
python llc_analysis_pipeline.py --mode single \
    --model_path ./model.pth \
    --model_name ResNet18

# Subsequent runs: skip calibration
python llc_analysis_pipeline.py --mode trajectory \
    --checkpoint_dir ./checkpoints/ \
    --model_name ResNet18 \
    --skip_calibration \
    --calibration_path ./calibration_results.json
```

## 🚨 Troubleshooting Guide

### Common Issues & Solutions

#### 1. Negative LLC Values
**Symptoms:** LLC estimates are negative
**Causes:** Step size too large, model not converged, chains escaped local minimum
**Solutions:**
- Reduce epsilon (step size): try 1e-5 instead of 1e-4
- Increase gamma (localization): try 200.0 instead of 100.0
- Check model training convergence
- Verify MALA acceptance rate < 0.5

#### 2. Poor MALA Acceptance Rate
**Symptoms:** Acceptance rate < 0.5 or > 0.95
**Solutions:**
- **Low acceptance (<0.5):** Reduce epsilon
- **High acceptance (>0.95):** Increase epsilon
- Use calibration to find optimal parameters automatically

#### 3. High LLC Variance
**Symptoms:** Large standard deviation relative to mean
**Solutions:**
- Increase number of chains: `num_chains=16`
- Increase number of draws: `num_draws=4000`
- Run calibration to find more stable parameters
- Check for gradient explosion or numerical issues

#### 4. Memory Issues
**Symptoms:** CUDA out of memory errors
**Solutions:**
- Reduce batch size: `batch_size=256` or `batch_size=128`
- Reduce number of chains: `num_chains=4`
- Use CPU instead of GPU: `device="cpu"`
- Process checkpoints in smaller batches

#### 5. Slow Performance
**Solutions:**
- Use GPU acceleration: `device="cuda"`
- Skip calibration for repeated runs
- Reduce number of draws for initial exploration
- Use checkpoint sampling: `max_checkpoints=10`

### Performance Optimization

```python
# Fast configuration for exploration
fast_config = LLCConfig(
    num_chains=3,      # Fewer chains
    num_draws=500,    # Fewer draws
    batch_size=256     # Smaller batches
)

# High-precision configuration for final results
precise_config = LLCConfig(
    num_chains=8,     # More chains
    num_draws=2000,    # More draws
    batch_size=512     # Larger batches
)
```

## 📊 Integration with Training Pipeline

### Workflow Integration
```python
# 1. Train model with checkpointing
from AT_replication_complete import train_model

trained_model = train_model(
    model_name="ResNet18",
    defense_method="AT",
    checkpoint_frequency=10  # Save every 10 epochs
)

# 2. Analyze LLC trajectory
from llc_analysis_pipeline import LLCAnalysisPipeline

pipeline = LLCAnalysisPipeline(LLCConfig())
trajectory_results = pipeline.analyze_checkpoint_trajectory(
    checkpoint_dir="./models/ResNet18_AT/epoch_iter/",
    model_name="ResNet18",
    dataset_name="CIFAR10",
    defense_method="AT"
)

# 3. Generate visualization
from generate_comparison_plots import generate_training_llc_plot

generate_training_llc_plot(
    model_dir="./models/ResNet18_AT/",
    llc_results=trajectory_results,
    output_path="./ResNet18_AT_training_dynamics.png"
)
```

### Result Interpretation
- **Compare relative values** rather than absolute
- **Consider confidence intervals** from multiple chains
- **Validate with multiple random seeds**
- **Correlate with other complexity measures**


## 📚 Theoretical Background

### Local Learning Coefficient (LLC)
The LLC measures the **effective dimensionality** of the loss landscape around a parameter configuration θ*:

```
LLC(θ*) = E[tr(H^(-1)(θ*)G(θ*)G(θ*)^T)]
```

Where:
- **H(θ*)** is the Hessian matrix at θ*
- **G(θ*)** is the gradient vector at θ*
- The expectation is over the data distribution

### SGLD Sampling
The LLC is estimated using **Stochastic Gradient Langevin Dynamics**:

```
θ_{t+1} = θ_t - ε∇L(θ_t) + √(2ε/β)ξ_t
```

Where:
- **ε** is the step size (epsilon)
- **β** is the inverse temperature
- **ξ_t** is Gaussian noise
- **∇L(θ_t)** is the stochastic gradient

### Hyperparameter Relationships
- **Epsilon (ε)**: Controls step size and sampling quality
- **Gamma (γ)**: Localization strength, prevents chains from wandering
- **Beta (β)**: Temperature parameter, typically set to 1/log(n)


## 🔗 References

1. **Local Learning Coefficient** - Lau et al. (2023): "The Local Learning Coefficient: A Singularity-Aware Complexity Measure"
2. **devinterp Library** - https://github.com/timaeus-research/devinterp
3. **SGLD Methodology** - Welling & Teh (2011): "Bayesian Learning via Stochastic Gradient Langevin Dynamics"
4. **MAIR Framework** - Kim et al. (2023): "Fantastic Robustness Measures: The Secrets of Robust Generalization"

---

**🎯 Ready to measure Local Learning Coefficients? Start with the Quick Start section and explore the comprehensive analysis capabilities!**