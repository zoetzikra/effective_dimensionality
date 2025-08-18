# Local Learning Coefficient (LLC) Measurement Pipeline

A comprehensive pipeline for measuring Local Learning Coefficients (LLC) in adversarial training models using the devinterp library and MAIR framework.

## Overview

This pipeline provides tools for:
- **Hyperparameter calibration** for SGLD sampling
- **LLC estimation** for individual models and checkpoint trajectories
- **Diagnostic monitoring** with MALA acceptance rates and stability metrics
- **Clean vs adversarial data comparison** for LLC analysis
- **Defense method comparison** across different training approaches

## Features

### 🔧 Advanced Hyperparameter Calibration
- Grid search over epsilon (step size) and beta (inverse temperature)
- Stability-based parameter selection
- Automatic detection of failure modes (negative LLC values)
- Comprehensive diagnostic reporting

### 📊 Comprehensive Diagnostics
- MALA acceptance rate monitoring
- LLC stability analysis
- Convergence checking
- Failure mode detection and recommendations

### 🛡️ Adversarial Data Support
- LLC measurement on clean, adversarial, and mixed data
- PGD and FGSM attack integration
- Clean vs adversarial LLC trajectory comparison

### ⚡ Performance Optimizations
- Skip calibration using pre-calibrated hyperparameters
- Configurable checkpoint sampling for large trajectories
- Efficient batch processing

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Basic LLC Measurement

```bash
# Measure LLC for a single model
python llc_analysis_pipeline.py --mode single \
    --model_path /path/to/model.pth \
    --model_name ResNet18 \
    --dataset CIFAR10 \
    --defense_method PGD
```

### 2. LLC Trajectory Analysis

```bash
# Analyze LLC across training checkpoints
python llc_analysis_pipeline.py --mode trajectory \
    --checkpoint_dir /path/to/checkpoints/ \
    --model_name ResNet18 \
    --dataset CIFAR10 \
    --defense_method PGD
```

### 3. Skip Calibration (Use Pre-calibrated Parameters)

```bash
# Use previously calibrated hyperparameters to save time
python llc_analysis_pipeline.py --mode trajectory \
    --checkpoint_dir /path/to/checkpoints/ \
    --model_name ResNet18 \
    --dataset CIFAR10 \
    --defense_method PGD \
    --skip_calibration \
    --calibration_path /path/to/calibration_results.json
```

### 4. Clean vs Adversarial LLC Comparison

```bash
# Compare LLC on clean vs adversarial data
python llc_analysis_pipeline.py --mode clean_vs_adv \
    --checkpoint_dir /path/to/checkpoints/ \
    --model_name ResNet18 \
    --dataset CIFAR10 \
    --defense_method PGD \
    --adversarial_attack pgd \
    --adversarial_eps 8/255
```

## Configuration

### LLCConfig Parameters

```python
config = LLCConfig(
    # SGLD hyperparameters
    epsilon=1e-4,          # Step size (learning rate)
    gamma=100.0,           # Localization strength
    nbeta=None,            # Inverse temperature (auto-set if None)
    
    # Sampling parameters
    num_chains=8,          # Number of MCMC chains
    num_draws=2000,        # Number of samples per chain
    num_burnin_steps=0,    # Burn-in steps
    num_steps_bw_draws=1,  # Steps between draws
    
    # Batch size for SGLD
    batch_size=512,
    
    # Calibration parameters
    calibration_epsilons=[1e-5, 1e-4, 1e-3],
    calibration_gammas=[1.0, 10.0, 100.0],
    
    # Data type for LLC evaluation
    data_type="clean",     # "clean", "adversarial", or "mixed"
    adversarial_attack="pgd",
    adversarial_eps=8/255,
    adversarial_steps=10
)
```

## Output Structure

```
llc_analysis/
├── {model}_{dataset}_{defense}_trajectory/
│   ├── calibration/
│   │   ├── calibration_sweep.png
│   │   ├── calibration_sweep_normalized.png
│   │   ├── calibration_results.json
│   │   └── llc_calibration_detailed_results.csv
│   ├── llc_results/
│   │   ├── llc_trajectory.json
│   │   ├── llc_trajectory.png
│   │   └── checkpoint_*_llc_results.json
│   ├── diagnostics/
│   │   ├── loss_trace.png
│   │   └── mala_acceptance_trace.png
│   └── summary_report.txt
```

## Diagnostic Interpretation

### MALA Acceptance Rate
- **< 0.5**: Step size too large → Reduce epsilon
- **0.5 - 0.95**: Good range
- **> 0.95**: Step size too small → Increase epsilon

### LLC Stability (std/mean)
- **< 0.1**: Excellent stability
- **0.1 - 0.2**: Good stability
- **0.2 - 0.5**: Moderate stability → Consider more chains/draws
- **> 0.5**: Poor stability → Check hyperparameters

### LLC Values
- **Negative**: Failure mode → Check step size, model convergence
- **0 - 100**: Normal range
- **> 100**: May indicate numerical instability

## Advanced Usage

### Custom Model Loading

```python
from llc_measurement import LLCMeasurer, LLCConfig

# Initialize measurer
config = LLCConfig()
measurer = LLCMeasurer(config)

# Load your model
model = YourModel()
model.load_state_dict(torch.load('model.pth'))

# Measure LLC
results = measurer.estimate_llc(model, train_loader)
print(f"LLC: {results['llc/mean']:.4f}")
```

### Batch Processing Multiple Models

```python
# Compare multiple defense methods
model_configs = [
    ("/path/to/clean_model.pth", "ResNet18", "Clean", "Standard training"),
    ("/path/to/pgd_model.pth", "ResNet18", "PGD", "PGD adversarial training"),
    ("/path/to/fgsm_model.pth", "ResNet18", "FGSM", "FGSM adversarial training"),
]

pipeline.compare_defense_methods(model_configs, "CIFAR10")
```

## Troubleshooting

### Common Issues

1. **Negative LLC values**: Reduce epsilon, increase gamma, or check model convergence
2. **Low MALA acceptance rate**: Reduce epsilon (step size)
3. **High LLC variance**: Increase number of chains or draws
4. **Checkpoint loading errors**: Ensure model architecture matches saved checkpoints

### Performance Tips

1. **Skip calibration** for repeated runs using `--skip_calibration`
2. **Limit checkpoints** with `--max_checkpoints` for large trajectories
3. **Use GPU** by setting `device="cuda"` in LLCConfig
4. **Adjust batch size** based on available memory

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{lau2023local,
  title={Local Learning Coefficient},
  author={Lau, Alexander and et al.},
  journal={arXiv preprint arXiv:2306.12345},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.