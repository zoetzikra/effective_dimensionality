# Training Jobs - SLURM Scripts for Adversarial Training

This directory contains SLURM job scripts for training adversarial models on HPC systems with GPU support.

## 🎯 Overview

These job scripts automate the training of various model architectures using different adversarial defense methods through the MAIR framework. Each script is configured for H100 GPU systems with appropriate resource allocation.

## 📁 Available Training Jobs

### LeNet (MNIST)
- **`run_mair_adv_training_lenet_Stand.job`** - Standard training (no adversarial)
- **`run_mair_adv_training_lenet_AT.job`** - Adversarial Training (AT)
- **`run_mair_adv_training_lenet_TRADES.job`** - TRADES defense method

### VGG11 (CIFAR10)
- **`run_mair_adv_training_vgg_Stand.job`** - Standard training
- **`run_mair_adv_training_vgg_AT.job`** - Adversarial Training (AT)
- **`run_mair_adv_training_vgg_AT_awp.job`** - AT with Adversarial Weight Perturbation

### ResNet18 (CIFAR10)
- **`run_mair_adv_training_resnet_TRADES.job`** - TRADES defense
- **`run_mair_adv_training_resnet_TRADES_awp.job`** - TRADES with AWP
- **`run_mair_adv_training_resnet_MART.job`** - MART defense
- **`run_mair_adv_training_resnet_AT_awp.job`** - AT with AWP

## 🚀 Usage

### Submit a Training Job
```bash
# Submit LeNet AT training
sbatch run_mair_adv_training_lenet_AT.job

# Submit ResNet18 TRADES training
sbatch run_mair_adv_training_resnet_TRADES.job

# Submit VGG11 AT with AWP
sbatch run_mair_adv_training_vgg_AT_awp.job
```

### Monitor Job Status
```bash
# Check job queue
squeue -u $USER

# Check specific job
squeue -j <job_id>

# View job output (while running)
tail -f RunMethod_<job_id>_*.out
```

### Cancel Job
```bash
scancel <job_id>
```

## ⚙️ Job Configuration

### Standard SLURM Parameters
```bash
#!/bin/bash
#SBATCH --partition=gpu_h100        # H100 GPU partition
#SBATCH --gpus=1                    # Single GPU
#SBATCH --cpus-per-task=8           # 8 CPU cores
#SBATCH --job-name=<job_name>       # Job identifier
#SBATCH --output=RunMethod_%A_*.out # Output file
#SBATCH --time=24:00:00             # 24 hour time limit
#SBATCH --mem=40G                   # 40GB RAM
#SBATCH --hint=nomultithread        # Disable hyperthreading
```

### Environment Setup
```bash
# Load required modules
module purge
module load 2023
module load Anaconda3/2023.07-2
module load CUDA/12.1.1

# Activate conda environment
source activate devinterp_env
```

## 📊 Training Configurations

### Model-Specific Settings

#### LeNet (MNIST)
- **Epochs**: 100
- **Batch Size**: 128
- **Attack**: PGD with ε=0.3, α=0.01, steps=40
- **Optimizer**: SGD with lr=0.01, momentum=0.9
- **Scheduler**: MultiStepLR at epochs [50, 75]

#### VGG11 (CIFAR10)
- **Epochs**: 200
- **Batch Size**: 128
- **Attack**: PGD with ε=8/255, α=2/255, steps=10
- **Optimizer**: SGD with lr=0.1, momentum=0.9, weight_decay=5e-4
- **Scheduler**: MultiStepLR at epochs [100, 150]

#### ResNet18 (CIFAR10)
- **Epochs**: 200
- **Batch Size**: 128
- **Attack**: PGD with ε=8/255, α=2/255, steps=10
- **Optimizer**: SGD with lr=0.1, momentum=0.9, weight_decay=5e-4
- **Scheduler**: MultiStepLR at epochs [100, 150]

### Defense Method Parameters

#### Adversarial Training (AT)
- Standard PGD adversarial training
- Uses PGD-10 during training

#### TRADES
- Trade-off parameter β=6.0
- Balances clean accuracy and robustness

#### MART
- Misclassification-aware training
- Uses boosted CE loss for hard examples

#### AWP (Adversarial Weight Perturbation)
- Weight perturbation strength ρ=5e-3
- Applied in addition to input perturbations

## 📁 Output Structure

Each training job creates the following structure:
```
models/<ModelName>_<DefenseMethod>/
├── best.pth              # Best model checkpoint
├── last.pth              # Final epoch checkpoint  
├── init.pth              # Initial model state
├── log.txt               # Training log
└── epoch_iter/           # Per-epoch checkpoints
    ├── 00001_00000.pth
    ├── 00002_00000.pth
    └── ...
```

## 🔍 Monitoring & Debugging

### Check Job Progress
```bash
# View real-time output
tail -f RunMethod_<job_id>_*.out

# Check GPU utilization
nvidia-smi

# Monitor job resources
sstat -j <job_id>
```

### Common Issues

#### 1. Out of Memory
- Reduce batch size in the job script
- Check GPU memory usage with `nvidia-smi`

#### 2. Module Load Errors
- Verify module availability: `module avail`
- Check conda environment: `conda env list`

#### 3. CUDA Errors
- Verify CUDA/GPU availability
- Check driver compatibility

#### 4. Job Time Limit
- Monitor training progress
- Extend time limit if needed: `--time=48:00:00`

## 🎯 Best Practices

### Resource Management
1. **Monitor resource usage** during initial runs
2. **Adjust memory allocation** based on model size
3. **Use appropriate time limits** (typically 24-48 hours)
4. **Check GPU utilization** to ensure efficient usage

### Job Organization
1. **Use descriptive job names** for easy identification
2. **Organize output files** by timestamp and model
3. **Save intermediate checkpoints** for recovery
4. **Log comprehensive training metrics**

### Experiment Planning
1. **Start with smaller models** (LeNet) for testing
2. **Validate configurations** before large-scale runs
3. **Plan job dependencies** for analysis pipelines
4. **Archive completed results** to prevent data loss

## 🔗 Integration with Analysis Pipeline

After training completion, use the analysis scripts:

```bash
# LLC trajectory analysis
python llc_analysis_pipeline.py --mode trajectory \
    --checkpoint_dir ./models/ResNet18_AT/epoch_iter/ \
    --model_name ResNet18 \
    --defense_method AT

# Effective dimensionality analysis  
python effective_dimensionality_analysis.py \
    --model ResNet18 \
    --defense AT

# Cross-attack analysis
python cross_attack_llc_analysis.py \
    --model_path ./models/ResNet18_AT/best.pth \
    --model_name ResNet18
```

## 📝 Customization

To create custom training jobs:

1. **Copy existing job script** as template
2. **Modify model/defense parameters** as needed
3. **Adjust resource allocation** (time, memory, GPUs)
4. **Update job name and output paths**
5. **Test with small configuration** first

### Example Custom Job
```bash
#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=Custom_Training
#SBATCH --output=RunMethod_%A_custom.out
#SBATCH --time=12:00:00
#SBATCH --mem=32G

# Load environment
module purge
module load 2023 Anaconda3/2023.07-2 CUDA/12.1.1
source activate devinterp_env

# Run custom training
python AT_replication_complete.py \
    --model YourModel \
    --method YourMethod \
    --custom-param value
```

---

**🎯 Ready to train adversarial models? Choose the appropriate job script and submit to the HPC system!**
