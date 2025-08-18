# Local Learning Coefficient (LLC) Measurement for Adversarial Training

This project implements Local Learning Coefficient (LLC) measurement using the devinterp library to analyze how model complexity evolves during adversarial training with different defense methods.

## Overview

The Local Learning Coefficient (LLC) is a singularity-aware complexity measure that quantifies the degeneracy of the loss landscape around a given parameter configuration. This implementation allows you to:

1. **Measure LLC for pre-trained models** - Analyze model complexity after training
2. **Track LLC trajectories** - Measure LLC across training checkpoints
3. **Compare defense methods** - Analyze how different adversarial training methods affect model complexity
4. **Hyperparameter calibration** - Advanced stability-based parameter selection with comprehensive diagnostics
5. **Diagnostic monitoring** - Monitor sampling quality, convergence, and detect potential issues

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_llc.txt
```

2. Install devinterp (if not already installed):
```bash
pip install devinterp
```

## Quick Start

### Basic LLC Measurement

```python
from llc_measurement import LLCMeasurer, LLCConfig
from AT_replication_complete import create_model_and_config

# Create a model
model, config = create_model_and_config("ResNet18")

# Setup LLC measurement
llc_config = LLCConfig(
    epsilon=1e-4,
    gamma=100.0,
    num_chains=8,
    num_draws=2000,
    batch_size=512
)

measurer = LLCMeasurer(llc_config)

# Measure LLC
llc_results = measurer.estimate_llc(model, train_loader)
print(f"LLC: {llc_results['llc/mean']:.4f} ± {llc_results['llc/std']:.4f}")
```

### Hyperparameter Calibration

```python
# Calibrate hyperparameters
optimal_params = measurer.calibrate_hyperparameters(
    model, 
    train_loader,
    save_path="./calibration_results"
)

# Use optimal parameters
llc_results = measurer.estimate_llc(
    model, 
    train_loader, 
    hyperparams=optimal_params
)
```

### LLC Trajectory Measurement

```python
# Measure LLC across multiple checkpoints
trajectory_results = measurer.measure_llc_trajectory(
    model_checkpoints=checkpoints,
    train_loader=train_loader,
    save_path="./llc_trajectory"
)
```

## Key Components

### 1. LLCConfig

Configuration class for LLC measurement parameters:

```python
@dataclass
class LLCConfig:
    epsilon: float = 1e-4          # SGLD step size
    gamma: float = 100.0           # Localization strength
    nbeta: Optional[float] = None  # Inverse temperature (auto-set)
    num_chains: int = 8            # Number of SGLD chains
    num_draws: int = 2000          # Steps per chain
    batch_size: int = 512          # SGLD batch size
    device: str = "cuda"           # Device to use
    data_type: str = "clean"       # "clean", "adversarial", or "mixed"
```

### 2. LLCMeasurer

Main class for LLC measurement operations:

- `estimate_llc()` - Measure LLC for a single model
- `calibrate_hyperparameters()` - Tune SGLD parameters
- `measure_llc_trajectory()` - Track LLC across checkpoints
- `run_diagnostics()` - Comprehensive diagnostic checks

### 3. LLCAnalysisPipeline

High-level pipeline for comprehensive analysis:

- `analyze_single_model()` - Analyze LLC for a single pre-trained model
- `analyze_checkpoint_trajectory()` - Analyze LLC across training checkpoints
- `compare_defense_methods()` - Compare different defense methods
- `compare_clean_vs_adversarial_llc()` - Compare LLC on clean vs adversarial data
- `generate_summary_report()` - Generate comprehensive reports

## Usage

### Single Model Analysis

```bash
# Analyze a single pre-trained model
python llc_analysis_pipeline.py --mode single \
    --model_path ./models/ResNet18_AT/best_Clean\(Val\).pth \
    --model_name ResNet18 \
    --dataset CIFAR10 \
    --defense_method AT
```

### Checkpoint Trajectory Analysis

```bash
# Analyze LLC trajectory across training checkpoints
python llc_analysis_pipeline.py --mode trajectory \
    --checkpoint_dir ./models/ResNet18_AT/ \
    --model_name ResNet18 \
    --dataset CIFAR10 \
    --defense_method AT
```

### Clean vs Adversarial LLC Comparison

```bash
# Compare LLC on clean vs adversarial data across checkpoints
python llc_analysis_pipeline.py --mode clean_vs_adv \
    --checkpoint_dir ./models/ResNet18_AT/ \
    --model_name ResNet18 \
    --dataset CIFAR10 \
    --defense_method AT \
    --adversarial_attack pgd \
    --adversarial_eps 8/255 \
    --adversarial_steps 10
```

### Defense Method Comparison

First, create a configuration file `model_config.json`:
```json
[
  [
    "./models/ResNet18_AT/best_Clean(Val).pth",
    "ResNet18",
    "AT",
    "Adversarial Training"
  ],
  [
    "./models/ResNet18_TRADES/best_Clean(Val).pth",
    "ResNet18",
    "TRADES",
    "TRADES Defense"
  ],
  [
    "./models/ResNet18_MART/best_Clean(Val).pth",
    "ResNet18",
    "MART",
    "MART Defense"
  ]
]
```

Then run the comparison:
```bash
python llc_analysis_pipeline.py --mode compare \
    --config_file model_config.json \
    --dataset CIFAR10
```

### Custom Analysis

```python
from llc_analysis_pipeline import LLCAnalysisPipeline
from llc_measurement import LLCConfig

# Setup configuration
llc_config = LLCConfig(
    epsilon=1e-4,
    gamma=100.0,
    num_chains=8,
    num_draws=2000,
    batch_size=512
)

# Initialize pipeline
pipeline = LLCAnalysisPipeline(llc_config)

# Analyze single model
results = pipeline.analyze_single_model(
    model_path="./models/ResNet18_AT/best_Clean(Val).pth",
    model_name="ResNet18",
    dataset_name="CIFAR10",
    defense_method="AT"
)

# Analyze checkpoint trajectory
trajectory = pipeline.analyze_checkpoint_trajectory(
    checkpoint_dir="./models/ResNet18_AT/",
    model_name="ResNet18",
    dataset_name="CIFAR10",
    defense_method="AT"
)

# Compare clean vs adversarial LLC
comparison = pipeline.compare_clean_vs_adversarial_llc(
    checkpoint_dir="./models/ResNet18_AT/",
    model_name="ResNet18",
    dataset_name="CIFAR10",
    defense_method="AT",
    adversarial_attack="pgd",
    adversarial_eps=8/255,
    adversarial_steps=10
)
```

## Workflow

### Step 1: Train Models with Checkpointing

First, train your models using your existing adversarial training pipeline with frequent checkpointing:

```python
# Train with frequent checkpointing
trained_model = train_model(
    model_name="ResNet18",
    defense_method="AT",
    checkpoint_frequency=10  # Save every 10 epochs
)
```

### Step 2: Analyze LLC

After training, use the LLC analysis pipeline to measure complexity:

```bash
# For single model analysis
python llc_analysis_pipeline.py --mode single --model_path your_model.pth

# For trajectory analysis
python llc_analysis_pipeline.py --mode trajectory --checkpoint_dir your_checkpoints/

# For comparison
python llc_analysis_pipeline.py --mode compare --config_file model_config.json
```

## Hyperparameter Guidelines

### Sophisticated Parameter Selection

The pipeline uses an advanced **stability-based selection** approach:

1. **Grid Search**: Sweeps across epsilon and beta values
2. **Stability Analysis**: Calculates `std/mean` ratio for each parameter combination
3. **Quality Filtering**: Removes NaN, extreme values, and negative LLC results
4. **Optimal Selection**: Chooses parameters with best stability among positive LLC results
5. **Comprehensive Reporting**: Provides detailed analysis and top 5 parameter combinations

### Calibration Process

```python
# The calibration process automatically:
# 1. Runs epsilon-beta sweep
# 2. Calculates stability metrics (std/mean ratio)
# 3. Filters out failure modes (negative LLC, extreme values)
# 4. Selects optimal parameters based on stability
# 5. Saves detailed results for analysis

optimal_params = measurer.calibrate_hyperparameters(model, train_loader)
```

**Output includes:**
- Optimal epsilon, gamma, and beta values
- Stability metrics and quality assessment
- Top 5 parameter combinations
- Detailed CSV results for further analysis
- Calibration plots and diagnostics

### Calibration Strategy

The pipeline uses a **single calibration strategy** for efficiency and consistency:

1. **Single Model Analysis**: Calibrates on the model being analyzed
2. **Trajectory Analysis**: Calibrates on the **final checkpoint** and uses those parameters for all checkpoints
3. **Comparison Analysis**: Uses the same hyperparameters across all models (for fair comparison)
4. **Clean vs Adversarial**: Calibrates on clean data from the final checkpoint, then applies to both clean and adversarial evaluation

This approach ensures:
- **Consistency**: Same hyperparameters across related analyses
- **Efficiency**: Avoids expensive calibration for each checkpoint
- **Fairness**: Enables direct comparison between different conditions

## Critical Parameters

### 1. ε (epsilon) - Step Size
- **Most important parameter** for LLC estimation
- **Range**: 1e-5 to 1e-3
- **Target MALA acceptance rate**: 0.9-0.95
- **Larger models require smaller step sizes**
- **Too large**: Chains escape local minimum, negative LLC
- **Too small**: Inefficient sampling, very high acceptance rate

### 2. γ (gamma) - Localization Strength
- **Controls proximity to initialization**
- **Range**: 1.0 to 200.0
- **Higher values for larger models**
- **Start with γ = 1.0, increase if chains escape**
- **Prevents chains from wandering too far from w***

### 3. β (beta) - Inverse Temperature
- **Usually set to 1/log(n) where n is dataset size**
- **Auto-calculated by default_nbeta()**
- **Controls the "temperature" of the sampling process**

### Sampling Parameters

- **Number of chains**: 4-20 (more chains = better estimates)
- **Number of draws**: 200-5000 (more draws = better convergence)
- **Batch size**: 32-2048 (balance between memory and stability)

## Data Types for LLC Evaluation

The pipeline supports three data types for LLC evaluation:

### 1. Clean Data (Default)
```python
llc_config = LLCConfig(data_type="clean")
```
- Evaluates LLC on clean, unperturbed data
- Standard approach for measuring model complexity

### 2. Adversarial Data
```python
llc_config = LLCConfig(
    data_type="adversarial",
    adversarial_attack="pgd",
    adversarial_eps=8/255,
    adversarial_steps=10
)
```
- Evaluates LLC on adversarially perturbed data
- Measures complexity under adversarial conditions
- Supports PGD and FGSM attacks

### 3. Mixed Data
```python
llc_config = LLCConfig(data_type="mixed")
```
- Evaluates LLC on a mix of clean and adversarial data
- Useful for comprehensive complexity assessment

### Research Applications

**Clean vs Adversarial Comparison** enables several research directions:

1. **Robustness Analysis**: How does model complexity change under adversarial conditions?
2. **Training Dynamics**: Do clean and adversarial LLC trajectories evolve differently?
3. **Defense Method Comparison**: Which defense methods maintain consistent complexity?
4. **Architecture Analysis**: How do different architectures handle adversarial complexity?

**Example Research Questions**:
- Does adversarial training increase or decrease model complexity?
- Are there specific training phases where clean and adversarial complexity diverge?
- Which defense methods maintain the most consistent complexity across data types?

## Diagnostic Tools

### 1. MALA Acceptance Rate
- **Target**: 0.9-0.95
- **< 0.5**: Step size too large, reduce epsilon
- **> 0.95**: Step size too small, increase epsilon
- **Purpose**: Quality control for SGLD sampling

### 2. Stability Metrics
- **Stability (std/mean)**: Measures consistency of LLC estimates
- **Excellent**: < 0.1
- **Good**: < 0.2
- **Moderate**: < 0.5
- **Poor**: > 0.5

### 3. Loss Traces
- Should converge to stable values
- Avoid negative dips
- Monitor for spikes or divergence

### 4. LLC Traces
- Should flatten to stable values
- Check for convergence within sampling budget
- Monitor for negative values (failure mode)

### 5. Chain Consistency
- Measures agreement between different SGLD chains
- Low coefficient of variation indicates good sampling
- High variation suggests hyperparameter issues

### 6. Convergence Analysis
- Compares early vs late portions of LLC traces
- Identifies if sampling has converged
- Guides decisions on number of draws needed

## Output and Results

### Directory Structure
```
llc_analysis/
└── llc_analysis_20241201_143022/
    ├── ResNet18_AT_single/
    │   ├── calibration/
    │   │   ├── calibration_sweep.png
    │   │   ├── calibration_sweep_normalized.png
    │   │   ├── calibration_results.json
    │   │   └── llc_calibration_detailed_results.csv
    │   ├── diagnostics/
    │   │   ├── loss_trace.png
    │   │   └── mala_acceptance_trace.png
    │   └── analysis_results.json
    ├── ResNet18_AT_trajectory/
    │   ├── calibration/
    │   ├── llc_results/
    │   │   ├── llc_trajectory.png
    │   │   └── llc_trajectory_enhanced.png
    │   └── trajectory_results.json
    ├── ResNet18_AT_clean_vs_adv/
    │   ├── clean_vs_adversarial_llc.png
    │   └── clean_vs_adv_comparison.json
    ├── defense_methods_comparison_CIFAR10.png
    └── analysis_summary.txt
```

### Key Outputs

1. **LLC Trajectory Plots** - Show how LLC evolves during training
2. **MALA Acceptance Rate Plots** - Monitor sampling quality
3. **Calibration Results** - Optimal hyperparameters with detailed analysis
4. **Enhanced Visualizations** - Trend analysis and change detection
5. **Detailed CSV Results** - Comprehensive data for further analysis
6. **Summary Reports** - Comprehensive analysis overview

## Troubleshooting

### Common Issues

1. **Negative LLC Estimates**
   - **Cause**: Step size too large, model not converged, chains escaped
   - **Solution**: Reduce epsilon, increase gamma, check model training
   - **Diagnostic**: Check MALA acceptance rate < 0.5

2. **Divergent SGLD Chains**
   - **Cause**: Step size too large, gradient explosion
   - **Solution**: Reduce step size significantly, check gradient computation
   - **Diagnostic**: Monitor loss traces for spikes

3. **Non-converged Estimates**
   - **Cause**: Insufficient sampling budget, poor hyperparameters
   - **Solution**: Increase number of chains/draws, reduce step size
   - **Diagnostic**: Check convergence analysis, stability metrics

4. **Low MALA Acceptance Rate**
   - **Cause**: Step size too large
   - **Solution**: Reduce epsilon
   - **Diagnostic**: MALA rate < 0.5

5. **Very High MALA Acceptance Rate**
   - **Cause**: Step size too small
   - **Solution**: Increase epsilon
   - **Diagnostic**: MALA rate > 0.95

6. **Poor Stability**
   - **Cause**: Insufficient sampling, poor hyperparameters
   - **Solution**: Increase chains/draws, recalibrate hyperparameters
   - **Diagnostic**: High std/mean ratio

### Performance Tips

1. **Use GPU acceleration** - Set device to "cuda" when available
2. **Adjust batch size** - Balance memory usage and stability
3. **Use fewer chains for calibration** - Speed up hyperparameter tuning
4. **Sample checkpoints** - Use max_checkpoints parameter for large trajectories
5. **Separate training and analysis** - Avoid memory overload
6. **Monitor diagnostics** - Use comprehensive diagnostic tools to catch issues early

## Integration with Existing Code

The LLC measurement system integrates seamlessly with your existing adversarial training pipeline:

```python
# Your existing training code
from AT_replication_complete import train_model

# Train model with checkpointing
trained_model = train_model(
    model_name="ResNet18",
    defense_method="AT",
    checkpoint_frequency=10  # Save every 10 epochs
)

# Later, analyze LLC using the pipeline
from llc_analysis_pipeline import LLCAnalysisPipeline

pipeline = LLCAnalysisPipeline(LLCConfig())
results = pipeline.analyze_checkpoint_trajectory(
    checkpoint_dir="./models/ResNet18_AT/",
    model_name="ResNet18",
    defense_method="AT"
)
```

## Research Applications

This implementation enables several research directions:

1. **Stage Detection** - Identify training phases through LLC changes
2. **Defense Method Comparison** - Compare complexity evolution across methods
3. **Robustness Analysis** - Correlate LLC with adversarial robustness
4. **Architecture Analysis** - Study how different architectures affect complexity
5. **Training Dynamics** - Understand how complexity evolves during training
6. **Adversarial Complexity** - Study how adversarial conditions affect model complexity

## Command Line Interface

The `llc_analysis_pipeline.py` provides a comprehensive command-line interface:

```bash
# Single model analysis
python llc_analysis_pipeline.py --mode single --model_path model.pth --model_name ResNet18

# Trajectory analysis
python llc_analysis_pipeline.py --mode trajectory --checkpoint_dir checkpoints/ --model_name ResNet18

# Comparison analysis
python llc_analysis_pipeline.py --mode compare --config_file models.json --dataset CIFAR10

# Clean vs adversarial comparison
python llc_analysis_pipeline.py --mode clean_vs_adv --checkpoint_dir checkpoints/ --model_name ResNet18

# View all options
python llc_analysis_pipeline.py --help
```

## Files

### Core Files
- `llc_measurement.py` - Core LLC measurement functionality with advanced calibration
- `llc_analysis_pipeline.py` - Comprehensive analysis pipeline
- `example_model_config.json` - Example configuration for comparison mode
- `requirements_llc.txt` - Required dependencies

### Output
- Results are saved in timestamped directories under `llc_analysis/`
- Each analysis creates its own subdirectory with results, plots, and diagnostics
- Detailed CSV files for further analysis and research

## References

- Lau et al. (2023): "The Local Learning Coefficient: A Singularity-Aware Complexity Measure"
- devinterp library: https://github.com/timaeus-research/devinterp
- SGLD methodology: Stochastic Gradient Langevin Dynamics for LLC estimation
- MALA methodology: Metropolis-Adjusted Langevin Algorithm for quality control


## License

This implementation follows the same license as the underlying devinterp library and the MAIR adversarial training code. 