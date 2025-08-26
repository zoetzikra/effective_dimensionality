# Effective Dimensionality Analysis Results

This directory contains results from effective dimensionality analysis using Hessian eigenvalue computation via Lanczos iteration. The analysis tracks model complexity evolution during adversarial training.

## 🎯 Overview

Effective dimensionality provides a complementary measure to LLC by analyzing the spectrum of the Hessian matrix. This approach:
- Computes top eigenvalues of the loss Hessian using Lanczos iteration
- Calculates effective dimensionality as a function of eigenvalue distribution
- Tracks complexity evolution across training checkpoints
- Compares complexity patterns across defense methods

## 📁 Directory Structure

```
eff_dim_analysis/
├── batch_analysis_YYYYMMDD_HHMMSS/    # Timestamped batch analysis runs
│   ├── ModelName_analysis/             # Per-model analysis results
│   │   ├── DefenseMethod_effective_dim.json      # Numerical results
│   │   ├── DefenseMethod_effective_dim_plot.png  # Trajectory plot
│   │   └── comparison_plot.png         # Multi-method comparison
│   ├── batch_analysis_summary.json    # Summary of all analyzed models
│   └── master_comparison_plot.png     # Cross-model comparison
└── README.md                          # This file
```

## 🚀 Usage

### Single Model Analysis
```bash
# Analyze effective dimensionality for single model/defense
python effective_dimensionality_analysis.py \
    --model ResNet18 \
    --defense AT \
    --max_checkpoints 10 \
    --nsteps 50
```

### Multi-Method Comparison
```bash
# Compare multiple defense methods for single model
python effective_dimensionality_analysis.py \
    --model ResNet18 \
    --compare \
    --max_checkpoints 10
```


## 📊 Analysis Results

### Individual Model Results

#### JSON Format (`DefenseMethod_effective_dim.json`)
```json
{
  "model_name": "ResNet18",
  "defense_method": "AT",
  "checkpoints": [
    {
      "checkpoint": 0,
      "epoch": 1,
      "eigenvalues": [145.23, 89.45, 67.12, ...],
      "effective_dim_s1": 23.45,
      "effective_dim_s10": 18.92,
      "effective_dim_s100": 12.34,
      "converged_eigs": 45,
      "total_eigs": 50
    },
    ...
  ],
  "analysis_summary": {
    "final_effective_dim": 15.67,
    "complexity_trend": "decreasing",
    "convergence_rate": 0.89
  }
}
```

#### Key Metrics
- **`eigenvalues`** - Top eigenvalues of the Hessian matrix
- **`effective_dim_s1`** - Effective dimensionality with s=1.0
- **`effective_dim_s10`** - Effective dimensionality with s=10.0  
- **`effective_dim_s100`** - Effective dimensionality with s=100.0
- **`converged_eigs`** - Number of converged eigenvalues
- **`complexity_trend`** - Overall trend in complexity evolution

### Batch Analysis Summary

#### Summary Format (`batch_analysis_summary.json`)
```json
{
  "timestamp": "2024-01-01T12:00:00",
  "models_analyzed": ["LeNet", "ResNet18", "VGG11"],
  "defense_methods": {
    "LeNet": ["Standard", "AT", "TRADES"],
    "ResNet18": ["Standard", "AT", "MART", "ATAWP"],
    "VGG11": ["Standard", "AT"]
  },
  "results_paths": {
    "LeNet": "./LeNet_analysis/",
    "ResNet18": "./ResNet18_analysis/",
    "VGG11": "./VGG11_analysis/"
  },
  "analysis_parameters": {
    "max_checkpoints": 10,
    "nsteps": 50,
    "eigenvalue_threshold": 1e-6
  }
}
```

## 📈 Visualization Outputs

### 1. Individual Trajectory Plots
**File**: `{DefenseMethod}_effective_dim_plot.png`

Shows evolution across training of:
- Effective dimensionality (multiple s values)
- Top eigenvalues
- Convergence indicators
- Training epoch markers

### 2. Multi-Method Comparison
**File**: `comparison_plot.png`

Compares different defense methods for same model:
- Side-by-side effective dimensionality trajectories
- Statistical significance indicators
- Performance correlation overlays
- Critical training phases

### 3. Cross-Model Analysis
**File**: `master_comparison_plot.png`

Compares across different architectures:
- Architecture-specific complexity patterns
- Defense method effectiveness by model
- Scaling relationships
- Complexity-performance trade-offs

## 🔧 Configuration Parameters

### Core Parameters
```python
# Analysis configuration
max_checkpoints = 10        # Maximum checkpoints to analyze
nsteps = 50                # Lanczos iteration steps
s_values = [1.0, 10.0, 100.0]  # Effective dimensionality scales

# Eigenvalue computation
eigenvalue_threshold = 1e-6  # Convergence threshold
max_iterations = 1000       # Maximum Lanczos iterations
random_seed = 42           # Reproducibility seed
```

### Model-Specific Settings
```python
# Checkpoint selection
checkpoint_pattern = "*.pth"           # Checkpoint file pattern
checkpoint_sorting = "numerical"       # Sorting method
skip_checkpoints = []                 # Checkpoints to skip

# Hessian computation
batch_size = 256                      # Batch size for Hessian-vector products
device = "cuda"                       # Computation device
precision = "float32"                 # Numerical precision
```

## 🔍 Effective Dimensionality Theory

### Definition
The effective dimensionality is calculated as:
```
eff_dim(λ, s) = Σᵢ λᵢ / (λᵢ + s)
```

Where:
- **λᵢ** are the eigenvalues of the Hessian
- **s** is a scale parameter controlling sensitivity
- Higher s values emphasize larger eigenvalues

### Interpretation
- **High effective dimensionality** → Complex loss landscape, many active directions
- **Low effective dimensionality** → Simple loss landscape, few dominant directions
- **Decreasing trend** → Model converging to simpler solution
- **Increasing trend** → Model becoming more complex (potential overfitting)

### Scale Parameter Effects
- **s = 1.0** - Sensitive to all eigenvalues, emphasizes fine structure
- **s = 10.0** - Balanced view of eigenvalue spectrum
- **s = 100.0** - Emphasizes only largest eigenvalues, captures dominant structure


## 🔗 Integration with LLC Analysis

### Combined Analysis
```python
# Correlate effective dimensionality with LLC
from effective_dimensionality_analysis import analyze_checkpoints
from llc_analysis_pipeline import LLCAnalysisPipeline

# Get effective dimensionality results
eff_dim_results = analyze_checkpoints("ResNet18", "AT")

# Get LLC trajectory results  
llc_pipeline = LLCAnalysisPipeline(LLCConfig())
llc_results = llc_pipeline.analyze_checkpoint_trajectory(
    checkpoint_dir="./models/ResNet18_AT/epoch_iter/",
    model_name="ResNet18",
    defense_method="AT"
)

# Create combined visualization
create_combined_complexity_plot(eff_dim_results, llc_results)
```

### Cross-Validation
- **Complexity trends** - Do LLC and effective dimensionality show similar patterns?
- **Critical points** - Do both measures identify same training phases?
- **Method ranking** - Do both measures rank defense methods similarly?
- **Architecture effects** - How do different architectures affect both measures?


---

