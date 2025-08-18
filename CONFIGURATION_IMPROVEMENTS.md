# LLC Configuration Improvements

## Overview
Updated the `LLCConfig` class to better align with the empirical LLC measurement guide recommendations.

## Key Improvements

### 1. **Model Size Scaling**
- **Before**: Fixed epsilon regardless of model size
- **After**: Auto-scales epsilon based on model parameters:
  - Small networks (≤10K params): ε = 1e-4
  - Medium networks (100K-1M params): ε = 1e-5  
  - Large networks (>1M params): ε = 1e-6

### 2. **Localization Strength (γ)**
- **Before**: γ = 100.0 (too large for most models)
- **After**: Auto-scales based on model type:
  - ResNet/CNN: γ = 1.0 (per guide)
  - Transformers: γ = 100.0 (per guide)
  - Default: γ = 1.0 (start small)

### 3. **Burn-in Period**
- **Before**: `num_burnin_steps = 0` (no burn-in)
- **After**: `num_burnin_steps = 90% of total steps` (guide recommendation)

### 4. **Sampling Parameters**
- **Before**: `num_draws = 2000` (confusing terminology)
- **After**: `num_steps = 2000` with `get_effective_draws()` method
- **Result**: 2000 total steps, 1800 burn-in, 200 effective draws

### 5. **Calibration Ranges**
- **Before**: Limited ranges `[1e-5, 1e-4, 1e-3]` and `[1.0, 10.0, 100.0]`
- **After**: Expanded ranges `[1e-6, 1e-5, 1e-4, 1e-3]` and `[1.0, 10.0, 100.0, 200.0]`

### 6. **Diagnostic Targets**
- **New**: `target_mala_acceptance = 0.92` (guide recommendation: 0.9-0.95)

### 7. **Model Validation**
- **New**: `validate_for_model()` method that:
  - Counts model parameters
  - Auto-scales hyperparameters
  - Prints configuration summary

## Configuration Summary Example

For a LeNet model (small network):
```
=== LLC Configuration Summary ===
Model: LeNet (61,706 parameters)
Step size (ε): 1.00e-05
Localization (γ): 1.0
Chains: 8
Total steps: 2000
Burn-in steps: 1800 (90%)
Effective draws: 200
Batch size: 512
Target MALA acceptance: 0.92
========================================
```

## Benefits

1. **Better Convergence**: Proper burn-in period improves LLC estimates
2. **Model-Appropriate Scaling**: Hyperparameters scale with model size
3. **Guide Compliance**: Follows empirical recommendations
4. **Improved Diagnostics**: MALA acceptance rate targets
5. **Automatic Validation**: Configuration validation for each model

## Usage

The improved configuration is automatically applied when you run the pipeline:

```bash
python llc_analysis_pipeline.py --mode trajectory \
    --checkpoint_dir /path/to/checkpoints \
    --model_name LeNet \
    --dataset MNIST \
    --defense_method AT
```

The system will automatically:
- Detect model size and scale epsilon appropriately
- Set gamma based on model type
- Configure proper burn-in periods
- Validate the configuration for your specific model 