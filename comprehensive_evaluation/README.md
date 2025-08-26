# Comprehensive Model Evaluation Framework

This directory contains results and utilities for comprehensive **inference-time** evaluation of adversarial training models, combining multiple analysis methods including LLC measurement, effective dimensionality analysis, and adversarial robustness evaluation.

## 🎯 Overview

The comprehensive evaluation framework provides a unified approach to analyze **inference-time model complexity** across training checkpoints:
- **Training dynamics** through **test-time LLC trajectories** (LLC measured on test data)
- **Model complexity evolution** via effective dimensionality during training
- **Adversarial robustness** across different attack types at inference
- **Clean vs adversarial performance** comparison on test data

> **🔍 Key Point**: All LLC measurements are performed at **inference time** using test data, not training data. This provides insights into how model complexity evolves during training as measured by the model's behavior on unseen test examples.

## 📋 Inference-Time vs Training-Time Analysis

**Why Inference-Time LLC Matters:**
- 🎯 **Generalization Analysis**: Shows how complex the model appears to unseen data
- 🔍 **Model Comparison**: Fair comparison across different training methods  
- 📊 **Robustness Insights**: How complexity changes under adversarial test conditions
- 🚀 **Practical Relevance**: Reflects real-world model deployment scenarios

## 📁 Directory Structure

```
comprehensive_evaluation/
├── evaluation_YYYYMMDD_HHMMSS/     # Timestamped evaluation runs
│   ├── adversarial_evaluation/      # Adversarial attack results
│   ├── data/                        # Cached datasets and preprocessing
│   ├── llc_analysis/               # LLC measurement results
│   │   └── temp_trajectory_*/       # Temporary LLC trajectory data
│   └── plots/                      # Generated visualizations
│       ├── *_clean_llc_with_training.pdf
│       ├── *_clean_llc_with_training.png
│       ├── *_adversarial_llc_with_training.pdf
│       └── *_adversarial_llc_with_training.png
└── README.md                       # This file
```

## 🚀 Usage

The comprehensive evaluation is primarily driven by the `comprehensive_model_evaluation.py` script in the parent directory.

### Basic Evaluation (Inference-Time LLC)
```bash
# Measure LLC on CLEAN TEST DATA across training checkpoints
python comprehensive_model_evaluation.py \
    --mode clean-test-llc \
    --checkpoint_dir ./models/LeNet_Standard/epoch_iter/ \
    --model_name LeNet \
    --dataset MNIST \
    --defense_method Standard \
    --max_checkpoints 30

# Measure LLC on ADVERSARIAL TEST DATA across training checkpoints
python comprehensive_model_evaluation.py \
    --mode adv-data-llc \
    --checkpoint_dir ./models/LeNet_AT/epoch_iter/ \
    --model_name LeNet \
    --dataset MNIST \
    --defense_method AT \
    --max_checkpoints 30
```

### Comparative Analysis
```bash
# Compare clean vs adversarial LLC trajectories
python comprehensive_model_evaluation.py \
    --mode plot-compare-trajectories \
    --model_path ./models/LeNet_AT/best.pth \
    --model_name LeNet \
    --dataset MNIST \
    --defense_method AT \
    --clean_trajectory_path ./clean_results/clean_test_llc_trajectory.json \
    --adversarial_trajectory_path ./adv_results/adversarial_test_llc_trajectory.json
```

## 📊 Evaluation Results

### Timestamped Evaluation Directories
Each evaluation run creates a timestamped directory containing:

#### 1. LLC Analysis Results
- **`llc_analysis/`** - Core LLC measurement data
  - Trajectory JSON files with epoch-by-epoch LLC values
  - Checkpoint-specific LLC measurements
  - Calibration results and diagnostics

#### 2. Adversarial Evaluation
- **`adversarial_evaluation/`** - Robustness assessment results
  - Attack success rates across different perturbation budgets
  - Clean vs adversarial accuracy comparisons
  - Per-checkpoint robustness metrics

#### 3. Generated Plots
- **`plots/`** - Visualization outputs
  - **Clean LLC plots** - LLC evolution on clean data
  - **Adversarial LLC plots** - LLC evolution on adversarial data
  - **Training dynamics** - Combined training metrics
  - **Comparative analysis** - Multi-model comparisons

### Example Results Structure
```
evaluation_20240101_120000/
├── llc_analysis/
│   └── temp_trajectory_clean/
│       └── llc_analysis_20240101_120000/
│           └── LeNet_MNIST_Standard_trajectory/
│               ├── llc_results/
│               │   ├── checkpoint_0_llc_results.json
│               │   ├── checkpoint_1_llc_results.json
│               │   ├── ...
│               │   ├── llc_trajectory.json
│               │   └── llc_trajectory.png
│               └── trajectory_results.json
├── plots/
│   ├── LeNet_clean_llc_with_training.pdf
│   ├── LeNet_clean_llc_with_training.png
│   ├── LeNet_adversarial_llc_with_training.pdf
│   └── LeNet_adversarial_llc_with_training.png
└── data/
    └── [cached datasets]
```

## 📈 Visualization Outputs

### 1. Clean LLC with Training Dynamics
**File**: `{ModelName}_clean_llc_with_training.png/pdf`

Shows the evolution of:
- **Test-time LLC values** measured on clean test data across training epochs
- Training loss progression (from training logs)
- Validation accuracy (from training logs)
- Clean test accuracy (from training logs)

### 2. Adversarial LLC with Training Dynamics  
**File**: `{ModelName}_adversarial_llc_with_training.png/pdf`

Shows the evolution of:
- **Test-time LLC values** measured on adversarial test data across training epochs
- Adversarial training loss (from training logs)
- Robust validation accuracy (from training logs)  
- Adversarial test accuracy (from training logs)

### 3. Comparative Analysis Plots
Side-by-side comparison of:
- Clean vs adversarial LLC trajectories
- Training dynamics differences
- Performance trade-offs
- Critical training phases

## 🔧 Configuration Options

### Evaluation Modes

#### `clean-test-llc`
- **Measures LLC on clean TEST data** across training checkpoints
- Evaluates how model complexity (as seen by test examples) evolves during training
- **Inference-time analysis**: Shows complexity from the perspective of unseen test data
- Requires: checkpoint directory, model configuration

#### `adv-data-llc`
- **Measures LLC on adversarially perturbed TEST data** across training checkpoints  
- Reveals how adversarial conditions affect **inference-time complexity**
- **Test-time robustness analysis**: Shows how complexity changes under adversarial test conditions
- Requires: checkpoint directory, attack configuration

#### `plot-compare-trajectories`
- Creates comparative visualizations
- Combines clean and adversarial results
- Requires: both clean and adversarial trajectory data

### Key Parameters

```bash
--mode                    # Evaluation mode (see above)
--checkpoint_dir         # Directory with training checkpoints
--model_name            # Model architecture (LeNet, VGG11, ResNet18)
--dataset              # Dataset (MNIST, CIFAR10)
--defense_method       # Training method (Standard, AT, TRADES, MART)
--max_checkpoints      # Maximum checkpoints to analyze
--calibration_path     # Path to LLC calibration results
--skip_calibration     # Skip hyperparameter calibration
--output_dir          # Custom output directory
```

## 🔍 Analysis Insights

### Training Phase Identification
The comprehensive evaluation helps identify training phases through **inference-time complexity**:
1. **Early Training** - High test-time LLC, rapid complexity changes as seen by test data
2. **Middle Training** - Test-time LLC stabilization, performance gains on test data
3. **Late Training** - Test-time LLC convergence, potential overfitting visible in test complexity
4. **Critical Transitions** - Sharp test-time LLC changes indicating phase shifts in generalization

### Robustness-Complexity Trade-offs
Compare how different defense methods affect:
- **LLC evolution patterns** - Complexity trajectories
- **Training stability** - Variance in LLC measurements
- **Final complexity** - End-of-training LLC values
- **Performance correlation** - LLC vs accuracy/robustness

### Clean vs Adversarial Insights (Inference-Time Analysis)
Analyze differences between test-time complexity on different data types:
- **Inference complexity differences** - How adversarial test data affects LLC vs clean test data
- **Generalization dynamics** - How complexity evolution differs between clean and adversarial test conditions
- **Robustness-complexity relationship** - How test-time complexity correlates with adversarial robustness
- **Generalization patterns** - Clean vs robust generalization as seen through test-time complexity


## 🔗 Integration with Main Pipeline

The comprehensive evaluation integrates with:

### Training Pipeline
```bash
# 1. Train model with checkpointing
python AT_replication_complete.py --model LeNet --method AT

# 2. Run comprehensive evaluation
python comprehensive_model_evaluation.py \
    --mode clean-test-llc \
    --checkpoint_dir ./models/LeNet_AT/epoch_iter/
```

-- --