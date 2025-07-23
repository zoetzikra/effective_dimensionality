#!/usr/bin/env python3
"""
MAIR-Compatible Checkpoint Training Function

This function extends the original MAIR trainer.fit() functionality 
while adding checkpoint saving every N epochs. It preserves all the 
sophisticated MAIR features including:
- Record management and evaluation
- Best model selection by validation metrics  
- AWP/minimizer support
- Proper scheduler handling
- All MAIR state tracking
"""

import os
import torch
from collections import OrderedDict

def save_checkpoint_mair(trainer, epoch, checkpoint_dir, training_loss=None):
    """Save MAIR-compatible checkpoint with all trainer state"""
    
    # Create checkpoint with all MAIR state
    checkpoint = {
        # Training progress
        'epoch': epoch,
        'accumulated_epoch': trainer.accumulated_epoch,
        'accumulated_iter': trainer.accumulated_iter, 
        'curr_epoch': trainer.curr_epoch,
        'curr_iter': trainer.curr_iter,
        
        # Model and optimizer states
        'rmodel_state_dict': trainer.rmodel.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict() if trainer.optimizer else None,
        'scheduler_state_dict': trainer.scheduler.state_dict() if trainer.scheduler else None,
        'scheduler_type': trainer.scheduler_type,
        'minimizer_state_dict': trainer.minimizer.state_dict() if trainer.minimizer else None,
        
        # MAIR record manager (contains all metrics history)
        'record_manager': trainer.rm,
        'dict_record': trainer.dict_record.copy() if trainer.dict_record else None,
        
        # Additional info
        'training_loss': training_loss,
    }
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as latest
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    torch.save(checkpoint, latest_path)
    
    print(f"MAIR checkpoint saved: {checkpoint_path}")
    return checkpoint_path

def load_checkpoint_mair(trainer, checkpoint_path):
    """Load MAIR-compatible checkpoint and restore all trainer state"""
    
    print(f"Loading MAIR checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Restore training progress
    trainer.accumulated_epoch = checkpoint.get('accumulated_epoch', 0)
    trainer.accumulated_iter = checkpoint.get('accumulated_iter', 0)
    trainer.curr_epoch = checkpoint.get('curr_epoch', 0)
    trainer.curr_iter = checkpoint.get('curr_iter', 0)
    
    # Restore model state
    trainer.rmodel.load_state_dict(checkpoint['rmodel_state_dict'])
    
    # Restore optimizer state
    if trainer.optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    # Restore scheduler state  
    if trainer.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    # Restore minimizer state (AWP etc.)
    if trainer.minimizer and 'minimizer_state_dict' in checkpoint and checkpoint['minimizer_state_dict']:
        trainer.minimizer.load_state_dict(checkpoint['minimizer_state_dict'])
        
    # Restore record manager (metrics history)
    if 'record_manager' in checkpoint and checkpoint['record_manager']:
        trainer.rm = checkpoint['record_manager']
        
    # Restore current record dict
    if 'dict_record' in checkpoint and checkpoint['dict_record']:
        trainer.dict_record = checkpoint['dict_record']
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}")
    return start_epoch

def fit_with_checkpoints(
    trainer,
    train_loader,
    n_epochs,
    checkpoint_frequency=5,
    resume_checkpoint=None,
    n_iters=None,
    record_type="Epoch",
    save_path=None,
    save_type="Epoch", 
    save_best=None,
    save_overwrite=False,
    refit=False,
):
    """
    Extended MAIR trainer.fit() with checkpoint saving functionality.
    
    This function preserves ALL original MAIR functionality while adding:
    - Automatic checkpoint saving every N epochs
    - Ability to resume from any checkpoint
    - Emergency checkpoint saving on interruption
    
    Args:
        trainer: MAIR trainer object (AT, TRADES, MART, etc.)
        train_loader: Training data loader
        n_epochs: Total number of epochs
        checkpoint_frequency: Save checkpoint every N epochs (default: 5)
        resume_checkpoint: Path to checkpoint to resume from (optional)
        ... (all other args same as MAIR trainer.fit())
    """
    
    # Setup checkpoint directory
    checkpoint_dir = None
    if save_path:
        checkpoint_dir = os.path.join(save_path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved every {checkpoint_frequency} epochs to: {checkpoint_dir}")
    
    # Resume from checkpoint if specified
    start_epoch_override = None
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        start_epoch_override = load_checkpoint_mair(trainer, resume_checkpoint)
        # Set refit=True to continue from checkpoint
        refit = True
        print(f"Resuming training from epoch {start_epoch_override}")
    
    # === Original MAIR trainer.fit() logic starts here ===
    
    # Check Save and Record Values
    trainer._check_valid_options(save_type)
    trainer._check_valid_options(record_type)
    trainer._check_valid_save_path(save_path, save_type, save_overwrite)

    if refit:
        if trainer.rm.count == 0 and not resume_checkpoint:
            raise ValueError("Please call load_dict for refitting.")
        # Update record and init
        trainer.rm.update(
            record_type=record_type, save_path=save_path, best_option=save_best
        )
        record_type = trainer.rm.record_type
        trainer.accumulated_epoch += -1
        
        # Use checkpoint epoch if resuming, otherwise use trainer state
        if start_epoch_override is not None:
            start_epoch = start_epoch_override - 1  # fit() expects 0-based
            start_iter = 0
        else:
            start_epoch = trainer.curr_epoch - 1
            start_iter = trainer.curr_iter
    else:
        # Start record and save init
        trainer.rm.initialize(
            record_type=record_type, save_path=save_path, best_option=save_best
        )
        if (save_path is not None) and (trainer.accumulated_iter == 0):
            trainer.save_dict(save_path, is_init=True)
        start_epoch = 0
        start_iter = 0

    # Print train information
    trainer.rm.print(record_type, "[%s]" % trainer.__class__.__name__)
    trainer.rm.print(record_type, "Training Information.")
    trainer.rm.print(record_type, "-Epochs: %s" % n_epochs)
    trainer.rm.print(record_type, "-Optimizer: %s" % trainer.optimizer)
    trainer.rm.print(record_type, "-Scheduler: %s" % trainer.scheduler)
    trainer.rm.print(record_type, "-Minimizer: %s" % trainer.minimizer)
    trainer.rm.print(record_type, "-Save Path: %s" % save_path)
    trainer.rm.print(record_type, "-Save Type: %s" % str(save_type))
    trainer.rm.print(record_type, "-Record Type: %s" % str(record_type))
    trainer.rm.print(record_type, "-Device: %s" % trainer.device)
    trainer.rm.print(record_type, "-Checkpoint Frequency: %s epochs" % checkpoint_frequency)

    # Start training
    try:
        epoch_losses = []  # Track losses for checkpoint saving
        
        for epoch in range(start_epoch, n_epochs):
            # Update current epoch and n_iters
            trainer.curr_epoch = epoch + 1
            trainer.accumulated_epoch += 1
            if n_iters is None:
                n_iters = len(train_loader)
            
            epoch_loss = 0.0
            batch_count = 0

            for i, train_data in enumerate(train_loader):
                # Update current iteration
                trainer.curr_iter = i + 1
                if trainer.curr_iter <= start_iter:  # For refit
                    continue
                trainer.accumulated_iter += 1
                is_last_batch = trainer.curr_iter == n_iters

                # Init records and dicts
                trainer._init_dicts_record_save()
                trainer.add_record_item("Epoch", trainer.accumulated_epoch)
                trainer.add_record_item("Iter", trainer.curr_iter)

                # Set train mode
                trainer.rmodel.train()

                # Update weight (THIS IS THE CRITICAL MAIR TRAINING STEP)
                trainer.rm.progress_start()
                trainer._update_weight(train_data)  # This handles AWP, minimizers, etc.
                trainer.rm.progress_end()
                
                # Track loss for checkpoint saving
                if hasattr(trainer, 'dict_record') and 'Loss' in trainer.dict_record:
                    epoch_loss += trainer.dict_record['Loss']
                    batch_count += 1

                # Eval mode (THIS IS WHERE CLEAN/PGD/FGSM ACCURACY IS CALCULATED)
                if trainer._check_run_condition(record_type, is_last_batch):
                    if type(trainer).record_during_eval != trainer.record_during_eval:
                        trainer.rmodel.eval()
                        trainer.record_during_eval()  # Calculates all robustness metrics
                    trainer.add_record_item("lr", trainer.optimizer.param_groups[0]["lr"])
                    trainer.rm.add(trainer.dict_record)
                    
                    # If record added, save dicts (BEST MODEL SELECTION BY VALIDATION METRICS)
                    if trainer.rm.check_best(trainer.dict_record):
                        trainer.save_dict(save_path, save_type, is_best=True)

                # Save dicts
                if save_path is not None:
                    is_save_condition = trainer._check_run_condition(
                        save_type, is_last_batch
                    )
                    # Save if condition is satisfied or best or end of the epoch.
                    if is_save_condition or is_last_batch:
                        trainer.save_dict(save_path, save_type, is_best=False)

                # Update scheduler (PROPER SCHEDULER HANDLING)
                if trainer._check_run_condition(trainer.scheduler_type, is_last_batch):
                    trainer.scheduler.step()

                # Check number of iterations
                if is_last_batch:
                    break

            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
            epoch_losses.append(avg_epoch_loss)
            
            # === CHECKPOINT SAVING LOGIC ===
            current_epoch = trainer.curr_epoch
            
            # Save checkpoint at specified intervals or final epoch
            if checkpoint_dir and (current_epoch % checkpoint_frequency == 0 or current_epoch == n_epochs):
                save_checkpoint_mair(trainer, current_epoch, checkpoint_dir, avg_epoch_loss)
                print(f"Checkpoint saved at epoch {current_epoch}")
            
            start_iter = 0  # For refit

        # Generate final summary (MAIR METRICS SUMMARY)
        if (save_path is not None) and (record_type is not None):
            trainer.rm.generate_summary()
            
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving emergency checkpoint...")
        if checkpoint_dir:
            save_checkpoint_mair(trainer, trainer.curr_epoch, checkpoint_dir, epoch_losses[-1] if epoch_losses else None)
            print("Emergency checkpoint saved. You can resume training later.")
        raise
    
    return checkpoint_dir

def quick_train_with_mair_checkpoints(model_name="ResNet18", defense_method="AT", 
                                    checkpoint_frequency=5, resume_from=None):
    """
    Quick training function using full MAIR functionality with checkpoints.
    This replaces our previous simplified version.
    """
    
    print(f"Training {model_name} with {defense_method} (MAIR-compatible)")
    print(f"Checkpoints every {checkpoint_frequency} epochs")
    
    # Import required modules
    import torchvision.datasets as dsets
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import mair
    from mair.defenses import AT, TRADES, MART, Standard
    from mair.utils.models import load_model
    
    # Setup data (CIFAR10 example)
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2023, 0.1994, 0.2010]
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    
    train_data = dsets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_data = dsets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=2)
    
    # Create model
    model = load_model(model_name, n_classes=10)
    rmodel = mair.RobModel(model, n_classes=10, normalization_used={'mean': MEAN, 'std': STD}).cuda()
    
    # Setup trainer
    EPS = 8/255
    ALPHA = 2/255
    STEPS = 10
    
    if defense_method == "AT":
        trainer = AT(rmodel, eps=EPS, alpha=ALPHA, steps=STEPS)
    elif defense_method == "TRADES":
        trainer = TRADES(rmodel, eps=EPS, alpha=ALPHA, steps=STEPS, beta=6.0)
    elif defense_method == "MART":
        trainer = MART(rmodel, eps=EPS, alpha=ALPHA, steps=STEPS, beta=6.0)
    else:
        trainer = Standard(rmodel)
    
    # CRITICAL: Setup robustness recording (this enables Clean/PGD/FGSM evaluation)
    trainer.record_rob(train_loader, test_loader, eps=EPS, alpha=ALPHA, steps=STEPS, std=0.1,
                       n_train_limit=1000, n_val_limit=1000)
    
    # Setup training parameters (exactly as MAIR paper)
    trainer.setup(optimizer="SGD(lr=0.1, momentum=0.9, weight_decay=0.0005)",
                  scheduler="Step(milestones=[100, 150], gamma=0.1)", 
                  scheduler_type="Epoch",
                  minimizer=None,  # Set to "AWP(rho=5e-3)" for AWP
                  n_epochs=200)
    
    # Train with checkpoints using full MAIR functionality
    save_path = f"./models/{model_name}_{defense_method}_mair"
    
    checkpoint_dir = fit_with_checkpoints(
        trainer=trainer,
        train_loader=train_loader,
        n_epochs=200,
        checkpoint_frequency=checkpoint_frequency,
        resume_checkpoint=resume_from,
        record_type="Epoch",
        save_path=save_path,
        save_best={"Clean(Val)": "HBO", "PGD(Val)": "HB"},  # Best model by validation metrics!
        save_type="Epoch",
        save_overwrite=True
    )
    
    print("Training completed with full MAIR functionality!")
    print(f"Checkpoints saved in: {checkpoint_dir}")
    
    # Final evaluation
    clean_acc = rmodel.eval_accuracy(test_loader)
    pgd_acc = rmodel.eval_rob_accuracy_pgd(test_loader, eps=EPS, alpha=ALPHA, steps=STEPS)
    
    print(f"Final Clean Accuracy: {clean_acc:.2f}%")
    print(f"Final PGD Accuracy: {pgd_acc:.2f}%")
    
    return rmodel, trainer

if __name__ == "__main__":
    # Example usage
    print("MAIR-Compatible Checkpoint Training")
    print("1. Train new model")
    print("2. Resume from checkpoint")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        checkpoint_path = input("Enter checkpoint path: ").strip()
        rmodel, trainer = quick_train_with_mair_checkpoints(
            model_name="ResNet18", 
            defense_method="AT",
            resume_from=checkpoint_path
        )
    else:
        rmodel, trainer = quick_train_with_mair_checkpoints(
            model_name="ResNet18", 
            defense_method="AT",
            checkpoint_frequency=5
        )


