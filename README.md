# Effective Dimensionality & Local Learning Coefficient Analysis for Adversarial Training

A comprehensive research framework for analyzing model complexity and learning dynamics in adversarial training through **Local Learning Coefficients (LLC)** and **Effective Dimensionality** measurements.

## 🎯 Overview

This project provides a complete pipeline for understanding how adversarial training affects model complexity through two complementary measures:

1. **Local Learning Coefficient (LLC)** - A singularity-aware complexity measure using SGLD sampling
2. **Effective Dimensionality** - Hessian eigenvalue-based complexity analysis via Lanczos iteration

### Key Capabilities

- **🛡️ Adversarial Training Pipeline** - Complete MAIR-based training with multiple defense methods
- **📊 LLC Measurement** - Advanced hyperparameter calibration and trajectory analysis
- **🔍 Effective Dimensionality Analysis** - Hessian eigenvalue computation and complexity tracking
- **⚔️ Cross-Attack Analysis** - Compare LLC signatures across different attack types (L∞, L2, L1)
- **🔬 Comprehensive Evaluation** - Clean vs adversarial data comparison
- **📈 Rich Visualizations** - Training dynamics, complexity evolution, and comparative plots
- **🚀 HPC Integration** - SLURM job scripts for large-scale experiments

## 🏗️ Project Structure

```
effective_dimensionality/
├── 🧠 Core Analysis
│   ├── llc_measurement.py              # LLC measurement with advanced calibration
│   ├── llc_analysis_pipeline.py        # Comprehensive LLC analysis pipeline
│   ├── effective_dimensionality_analysis.py # Hessian-based complexity analysis
│   ├── cross_attack_llc_analysis.py    # Cross-attack LLC signature analysis
│   └── comprehensive_model_evaluation.py # Unified evaluation framework
│
├── 🛡️ Adversarial Training
│   ├── AT_replication_complete.py      # Complete adversarial training pipeline
│   ├── AT_replication_single_model.py  # Single model training
│   ├── mair_compatible_checkpoint_trainer.py # MAIR checkpoint integration
│   └── MAIR/                           # MAIR adversarial training framework
│
├── 📊 Visualization & Analysis
│   ├── generate_comparison_plots.py    # Training dynamics visualization
│   ├── create_cross_attack_plot.py     # Cross-attack analysis plots
│   ├── compare_final_llc_values.py     # Final LLC comparison
│   └── fix_multi_epsilon_plot.py       # Multi-epsilon analysis plots
│
├── 🔧 Utilities & Support
│   ├── hess_vec_prod.py                # Hessian-vector products
│   ├── utils.py                        # General utilities
│   ├── model.py                        # Model definitions
│   └── inspect_checkpoint.py           # Checkpoint inspection
│
├── 💼 Job Scripts & Workflows
│   ├── run_*.job                       # SLURM job scripts
│   ├── training_jobs/                  # Training job scripts
│   └── RunMethod_*.out                 # Job outputs and logs
│
├── 📁 Data & Results
│   ├── models/                         # Trained model checkpoints
│   ├── llc_analysis/                   # LLC analysis results
│   ├── eff_dim_analysis/              # Effective dimensionality results
│   ├── comprehensive_evaluation/       # Evaluation results
│   └── data/                          # Dataset cache
│
└── 📚 Documentation
    ├── README.md                       # This file
    ├── README_LLC.md                   # Detailed LLC documentation
    ├── CONFIGURATION_IMPROVEMENTS.md   # Configuration guide
    └── requirements_llc.txt            # Python dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository_url>
cd effective_dimensionality

# Install dependencies
pip install -r requirements_llc.txt

# Install MAIR framework
cd MAIR
pip install -e .
cd ..

# Install devinterp for LLC measurement
pip install devinterp
```

### 2. Train Models

```bash
# Train a single model with adversarial training
python AT_replication_complete.py --model ResNet18 --method AT

# Train with Adversarial Weight Perturbation (AWP)
python AT_replication_complete.py --model ResNet18 --method AT --awp

# Run complete experiment suite
python AT_replication_complete.py --full-suite
```

### 3. Analyze LLC Trajectories

```bash
# Single model LLC analysis
python llc_analysis_pipeline.py --mode single \
    --model_path ./models/ResNet18_AT/best.pth \
    --model_name ResNet18 \
    --dataset CIFAR10 \
    --defense_method AT

# Trajectory analysis across training checkpoints
python llc_analysis_pipeline.py --mode trajectory \
    --checkpoint_dir ./models/ResNet18_AT/epoch_iter/ \
    --model_name ResNet18 \
    --dataset CIFAR10 \
    --defense_method AT
```

### 4. Effective Dimensionality Analysis

```bash
# Single model effective dimensionality
python effective_dimensionality_analysis.py --model ResNet18 --defense AT

# Compare multiple defense methods
python effective_dimensionality_analysis.py --compare --model ResNet18

# Batch analyze all models
python effective_dimensionality_analysis.py --batch
```

### 5. Cross-Attack Analysis

```bash
# Analyze LLC signatures across different attack types
python cross_attack_llc_analysis.py \
    --model_path ./models/LeNet_AT/best.pth \
    --model_name LeNet \
    --dataset MNIST \
    --calibration_path ./calibration_results.json \
    --checkpoint_dir ./models/LeNet_AT/epoch_iter/
```

## 🛡️ Supported Defense Methods

| Method | Description | Architectures | Datasets |
|--------|-------------|---------------|----------|
| **Standard** | Standard training (no adversarial) | LeNet, VGG11, ResNet18 | MNIST, CIFAR10 |
| **AT** | Adversarial Training (Madry et al.) | LeNet, VGG11, ResNet18 | MNIST, CIFAR10 |
| **TRADES** | Trade-off between Robustness and Accuracy | LeNet, VGG11, ResNet18 | MNIST, CIFAR10 |
| **MART** | Misclassification Aware adveRsarial Training | ResNet18 | CIFAR10 |
| **AT + AWP** | AT with Adversarial Weight Perturbation | VGG11, ResNet18 | CIFAR10 |
| **TRADES + AWP** | TRADES with AWP | ResNet18 | CIFAR10 |

## 📊 Analysis Capabilities

### Local Learning Coefficient (LLC)
- **Advanced Hyperparameter Calibration** - Stability-based parameter selection
- **Trajectory Analysis** - Track complexity evolution during training
- **Clean vs Adversarial Comparison** - LLC on different data types
- **Cross-Attack Signatures** - Compare LLC across L∞, L2, L1 attacks
- **Diagnostic Monitoring** - MALA acceptance rates, convergence analysis

### Effective Dimensionality
- **Hessian Eigenvalue Analysis** - Top eigenvalues via Lanczos iteration
- **Complexity Evolution** - Track effective dimensionality during training
- **Defense Method Comparison** - Compare complexity across training methods
- **Batch Processing** - Analyze multiple models simultaneously

### Comprehensive Evaluation
- **Training Dynamics** - Loss, accuracy, and robustness evolution
- **Model Comparison** - Side-by-side analysis of different approaches
- **Statistical Analysis** - Significance testing and confidence intervals
- **Rich Visualizations** - Publication-ready plots and figures

## 🔬 Research Applications

### 1. Training Dynamics Analysis
```bash
# Generate training dynamics plots with LLC and effective dimensionality
python generate_comparison_plots.py --mode training_llc \
    --model_dir ./models/ResNet18_AT \
    --experiment_dir ./llc_analysis/ResNet18_AT_results \
    --eff_dim_path ./eff_dim_analysis/ResNet18_AT_results.json
```

### 2. Defense Method Comparison
```bash
# Compare LLC across different defense methods
python llc_analysis_pipeline.py --mode compare \
    --config_file model_comparison_config.json \
    --dataset CIFAR10
```

### 3. Cross-Attack Analysis
```bash
# Test LLC signatures across attack types
python cross_attack_llc_analysis.py \
    --model_path ./models/ResNet18_AT/best.pth \
    --model_name ResNet18 \
    --calibration_path ./calibration_results.json
```

## 🖥️ HPC Integration

The project includes comprehensive SLURM job scripts for HPC environments:

### Training Jobs
```bash
# Submit adversarial training job
sbatch training_jobs/run_mair_adv_training_resnet_AT.job

# Submit training with AWP
sbatch training_jobs/run_mair_adv_training_resnet_AT_awp.job
```

### Analysis Jobs
```bash
# Submit LLC analysis job
sbatch run_llc_analysis.job

# Submit effective dimensionality analysis
sbatch run_eff_dim_analysis.job

# Submit cross-attack analysis
sbatch run_cross_attack_analysis.job
```

## 📈 Example Workflows

### Workflow 1: Complete Model Analysis
```bash
# 1. Train model
python AT_replication_complete.py --model ResNet18 --method AT

# 2. Analyze LLC trajectory
python llc_analysis_pipeline.py --mode trajectory \
    --checkpoint_dir ./models/ResNet18_AT/epoch_iter/ \
    --model_name ResNet18

# 3. Analyze effective dimensionality
python effective_dimensionality_analysis.py --model ResNet18 --defense AT

# 4. Generate comparison plots
python generate_comparison_plots.py --mode training_llc \
    --model_dir ./models/ResNet18_AT \
    --experiment_dir ./llc_results \
    --eff_dim_path ./eff_dim_results.json
```

### Workflow 2: Defense Method Comparison
```bash
# 1. Train multiple models
python AT_replication_complete.py --model ResNet18 --method Standard
python AT_replication_complete.py --model ResNet18 --method AT
python AT_replication_complete.py --model ResNet18 --method TRADES

# 2. Compare LLC across methods
python llc_analysis_pipeline.py --mode compare \
    --config_file defense_comparison.json

# 3. Compare effective dimensionality
python effective_dimensionality_analysis.py --compare --model ResNet18
```

### Workflow 3: Cross-Attack Investigation
```bash
# 1. Train adversarially robust model
python AT_replication_complete.py --model LeNet --method AT

# 2. Run cross-attack LLC analysis
python cross_attack_llc_analysis.py \
    --model_path ./models/LeNet_AT/best.pth \
    --model_name LeNet \
    --calibration_path ./calibration_results.json

# 3. Create cross-attack visualization
python create_cross_attack_plot.py \
    --results_dir ./cross_attack_results \
    --model_name "LeNet (AT)"
```

## 📊 Output Structure

```
results/
├── llc_analysis/
│   ├── llc_analysis_YYYYMMDD_HHMMSS/
│   │   ├── ModelName_DefenseMethod_single/
│   │   │   ├── calibration/
│   │   │   │   ├── calibration_sweep.png
│   │   │   │   ├── calibration_results.json
│   │   │   │   └── llc_calibration_detailed_results.csv
│   │   │   ├── diagnostics/
│   │   │   │   ├── loss_trace.png
│   │   │   │   └── mala_acceptance_trace.png
│   │   │   └── analysis_results.json
│   │   └── ModelName_DefenseMethod_trajectory/
│   │       ├── llc_results/
│   │       │   ├── llc_trajectory.png
│   │       │   └── checkpoint_*_llc_results.json
│   │       └── trajectory_results.json
│
├── eff_dim_analysis/
│   ├── batch_analysis_YYYYMMDD_HHMMSS/
│   │   ├── ModelName_analysis/
│   │   │   ├── DefenseMethod_effective_dim.json
│   │   │   └── DefenseMethod_effective_dim_plot.png
│   │   └── batch_analysis_summary.json
│
├── cross_attack_analysis/
│   ├── cross_attack_llc_trajectories.json
│   ├── cross_attack_comparison_plot.png
│   └── cross_attack_summary_report.txt
│
└── comprehensive_evaluation/
    ├── evaluation_YYYYMMDD_HHMMSS/
    │   ├── adversarial_evaluation/
    │   ├── llc_analysis/
    │   └── plots/
    └── comparison_plots/
```

## 🛠️ Configuration

### LLC Configuration
```python
from llc_measurement import LLCConfig

config = LLCConfig(
    epsilon=1e-4,          # SGLD step size
    gamma=100.0,           # Localization strength  
    num_chains=8,          # Number of SGLD chains
    num_draws=2000,        # Samples per chain
    batch_size=512,        # SGLD batch size
    data_type="clean",     # "clean", "adversarial", or "mixed"
)
```

### Model Configuration
```json
{
  "models": [
    ["./models/ResNet18_AT/best.pth", "ResNet18", "AT", "Adversarial Training"],
    ["./models/ResNet18_TRADES/best.pth", "ResNet18", "TRADES", "TRADES Defense"],
    ["./models/ResNet18_MART/best.pth", "ResNet18", "MART", "MART Defense"]
  ]
}
```

## 🔍 Key Research Questions

This framework enables investigation of:

1. **How does adversarial training affect model complexity?**
   - Compare LLC trajectories across defense methods
   - Analyze effective dimensionality evolution

2. **Do different attack types leave distinct complexity signatures?**
   - Cross-attack LLC analysis across L∞, L2, L1 norms
   - Attack-specific complexity patterns

3. **What are the training dynamics of robust models?**
   - LLC and effective dimensionality during training
   - Critical transitions and phase changes

4. **How do architectural choices impact robustness-complexity trade-offs?**
   - Compare LeNet, VGG11, ResNet18 across defense methods
   - Architecture-specific complexity patterns

## 📚 Documentation

- **[README_LLC.md](README_LLC.md)** - Detailed LLC measurement documentation
- **[CONFIGURATION_IMPROVEMENTS.md](CONFIGURATION_IMPROVEMENTS.md)** - Configuration and setup guide
- **[MAIR/README.md](MAIR/README.md)** - MAIR framework documentation

## 🤝 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{effective_dimensionality_adversarial,
  title={Effective Dimensionality and Local Learning Coefficient Analysis for Adversarial Training},
  author={[Your Name]},
  year={2024},
  note={Research framework for analyzing model complexity in adversarial training}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions, issues, or contributions:
1. Check existing documentation in `README_LLC.md`
2. Review configuration guide in `CONFIGURATION_IMPROVEMENTS.md`
3. Examine example job scripts in `training_jobs/`
4. Open an issue with detailed information about your setup and error

---
