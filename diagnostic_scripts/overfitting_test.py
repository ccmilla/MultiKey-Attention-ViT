import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from PyTorchLgtAttTemplate import LitNetwork

"""
OVERFITTING TEST
================
Goal: Train on just 100 samples for 20 epochs
Expected: Model should reach 90%+ accuracy (proving it CAN learn)
If this fails, something is still fundamentally broken
If this works, we can move to full training
"""

print("="*60)
print("OVERFITTING TEST - Training on 100 samples")
print("="*60)

# Setup
dataset_dir = "../data/"
b = 16  # Smaller batch for tiny dataset
width = 224
height = 224

transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop((height, width)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("\nLoading dataset...")
full_dataset = torchvision.datasets.Food101(
    root=dataset_dir, 
    split='train', 
    transform=transforms, 
    download=True
)

# Create tiny subset - just 100 samples
print("Creating tiny subset of 100 samples...")
tiny_dataset = Subset(full_dataset, range(100))
print(f"✓ Tiny dataset size: {len(tiny_dataset)}")

# Use 80 for train, 20 for val
train_size = 80
val_size = 20
tiny_train, tiny_val = torch.utils.data.random_split(
    tiny_dataset, 
    [train_size, val_size]
)

train_loader = DataLoader(
    tiny_train, 
    batch_size=b, 
    shuffle=True, 
    num_workers=0  # Set to 0 for easier debugging
)
val_loader = DataLoader(
    tiny_val, 
    batch_size=b, 
    shuffle=False, 
    num_workers=0
)

print(f"Train samples: {len(tiny_train)}")
print(f"Val samples: {len(tiny_val)}")
print(f"Batches per epoch: {len(train_loader)}")

# Check what we're training on
print("\nSampling labels from training set...")
sample_batch = next(iter(train_loader))
print(f"Sample labels: {sample_batch[1].tolist()}")

# Create model
print("\n" + "="*60)
print("Creating model...")
print("="*60)

model_name = "ViTLayerReduction"

# IMPORTANT: Using simplified learning rate for this test
model = LitNetwork(
    model_name=model_name,
    pretrained=True,  # We want pretrained if possible
    freeze_backbone=False,
    lr=1e-4,  # Simple constant LR for this test
    num_epochs=20  # Just 20 epochs for quick test
)

print(f"✓ Model created: {model_name}")
print(f"Learning rate: {model.hparams.lr}")

# Setup training
checkpoint = pl.callbacks.ModelCheckpoint(
    monitor='val_acc_epoch',
    save_top_k=1,
    mode='max',
    filename='overfit-test-{epoch:02d}-{val_acc_epoch:.2f}'
)

logger = pl_loggers.TensorBoardLogger(
    save_dir="logs_overfit_test",
    name="tiny_100_samples"
)

# Train
print("\n" + "="*60)
print("Starting training (20 epochs on 100 samples)...")
print("="*60)
print("\nExpected behavior:")
print("- Loss should decrease steadily")
print("- Train accuracy should reach 90%+ by epoch 20")
print("- Val accuracy should reach 60%+ (some overfitting expected)")
print("\nIf you see this, the model CAN learn!")
print("="*60 + "\n")

device = "gpu" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

torch.set_float32_matmul_precision('medium')

trainer = pl.Trainer(
    max_epochs=20,
    accelerator=device,
    callbacks=[checkpoint],
    logger=logger,
    log_every_n_steps=1,  # Log every step since we have so few
    enable_progress_bar=True
)

# Train
trainer.fit(model, train_loader, val_loader)

# Print final results
print("\n" + "="*60)
print("OVERFITTING TEST RESULTS")
print("="*60)

# Get final metrics from logger
metrics = trainer.callback_metrics
final_train_acc = metrics.get('train_acc_epoch', 0)
final_val_acc = metrics.get('val_acc_epoch', 0)
final_train_loss = metrics.get('train_loss', 0)

print(f"\nFinal train accuracy: {final_train_acc*100:.2f}%")
print(f"Final val accuracy: {final_val_acc*100:.2f}%")
print(f"Final train loss: {final_train_loss:.4f}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

if final_train_acc > 0.9:
    print("✅ SUCCESS! Model can learn (>90% train accuracy)")
    print("   Your custom attention mechanism is working!")
    print("   Ready to train on full dataset.")
elif final_train_acc > 0.5:
    print("⚠️  PARTIAL SUCCESS - Model is learning but slowly")
    print(f"   Reached {final_train_acc*100:.1f}% accuracy")
    print("   Possible issues:")
    print("   - Learning rate might be too low")
    print("   - Custom attention might be too restrictive")
    print("   - Need more epochs")
elif final_train_acc > 0.1:
    print("⚠️  WEAK LEARNING - Model is learning very slowly")
    print(f"   Only reached {final_train_acc*100:.1f}% accuracy")
    print("   Possible issues:")
    print("   - Learning rate definitely too low")
    print("   - Model capacity issues (only 6 blocks?)")
    print("   - Custom attention might be broken")
else:
    print("❌ FAILURE - Model is not learning (<10% accuracy)")
    print("   Something is still fundamentally broken")
    print("   Check:")
    print("   - Is pretrained=True actually working?")
    print("   - Are masks applied correctly?")
    print("   - Is optimizer configured right?")

print("\nNext steps:")
if final_train_acc > 0.7:
    print("1. Try full training on entire dataset")
    print("2. Monitor for overfitting")
    print("3. Consider increasing to 10-12 transformer blocks")
else:
    print("1. Debug learning rate and optimizer")
    print("2. Try baseline model without custom attention")
    print("3. Verify pretrained weights are loading")

print("\nView training curves:")
print(f"  tensorboard --logdir=logs_overfit_test")