'''
Baseline comparison experiment

Compare custom spatial attention ViT against multiple baseline architectures.

Experiments:
1) Standard ViT-Small - baseline transformer
2) ResNet18 - baseline CNN
3) ResNet50 - deeper CNN baseline
4) EfficientNet-B0 - efficient architecture
5) Custom ViT 12 blocks - your full model
6) Custom ViT 10 blocks - your current model
7) Custom ViT 6 blocks - reduced capacity

Optimized for 16GB GPU (RTX 5080)
'''
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
from AttTemplate import LitNetwork


# Configuration
dataset_dir = "../data/"
b = 64  # Optimal for 16GB GPU
num_epochs = 100  # Reduced from 150 for faster comparison
device = "gpu"

# Shared transforms
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop((224, 224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225]),
])

val_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225]),
])

# Load datasets
print("Loading datasets...")
train_dataset = torchvision.datasets.Food101(
    root=dataset_dir, 
    split='train', 
    transform=train_transforms, 
    download=True
)

# Split train into train/val
train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset, 
    [int(len(train_dataset)*0.8), len(train_dataset) - int(len(train_dataset)*0.8)]
)

# Use actual test split for final testing
test_dataset = torchvision.datasets.Food101(
    root=dataset_dir, 
    split='test', 
    transform=val_transforms, 
    download=True
)

train_loader = DataLoader(train_dataset, batch_size=b, shuffle=True, 
                         num_workers=12, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=b, shuffle=False, 
                       num_workers=12, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=b, shuffle=False, num_workers=4)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

torch.set_float32_matmul_precision('medium')

# ============================================================================
# GPU MEMORY TEST
# ============================================================================
print("\n" + "="*80)
print("GPU MEMORY TEST")
print("="*80)

import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Quick memory test
    test_model = LitNetwork(
        model_name="vit_small_patch16_224",
        pretrained=True,
        freeze_backbone=False,
        lr=1e-4,
        peak_lr=1e-4,
        weight_decay=0.01,
        warmup_epochs=3,
        rampup_epochs=5,
        num_epochs=num_epochs
    ).to('cuda')
    
    try:
        test_batch = torch.randn(b, 3, 224, 224).to('cuda')
        test_labels = torch.randint(0, 101, (b,)).to('cuda')
        test_model.train()
        output = test_model(test_batch)
        loss = torch.nn.functional.cross_entropy(output, test_labels)
        loss.backward()
        
        print(f"✓ Batch size {b} works!")
        print(f"  Peak memory: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"✗ Batch size {b} is TOO LARGE!")
            print(f"  Reduce to batch_size=32")
        else:
            raise e
    finally:
        del test_model
        torch.cuda.empty_cache()

print("="*80 + "\n")

# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

experiments = [
    {
        "name": "1_baseline_vit_small",
        "model_name": "vit_small_patch16_224",
        "description": "Baseline: Standard ViT-Small (12 blocks, standard attention)",
        "expected_acc": "70-75%"
    },
    {
        "name": "2_resnet18",
        "model_name": "resnet18tv",
        "description": "ResNet18 (CNN baseline)",
        "expected_acc": "68-73%"
    },
    {
        "name": "3_resnet50",
        "model_name": "resnet50tv",
        "description": "ResNet50 (deeper CNN baseline)",
        "expected_acc": "72-76%"
    },
    {
        "name": "4_efficientnet_b0",
        "model_name": "efficientnet_b0",
        "description": "EfficientNet-B0 (efficient architecture)",
        "expected_acc": "73-77%"
    },
    {
        "name": "5_custom_12blocks",
        "model_name": "ViTLayerReduction",
        "num_blocks": 12,
        "description": "Your model: Custom 5-way spatial attention (12 blocks)",
        "expected_acc": "65-72%"
    },
    {
        "name": "6_custom_10blocks",
        "model_name": "ViTLayerReduction", 
        "num_blocks": 10,
        "description": "Your current: Custom 5-way spatial attention (10 blocks)",
        "expected_acc": "60-68%"
    },
    {
        "name": "7_custom_6blocks",
        "model_name": "ViTLayerReduction",
        "num_blocks": 6,
        "description": "Reduced: Custom 5-way spatial attention (6 blocks)",
        "expected_acc": "50-60%"
    }
]

# Store results
results = []

# ============================================================================
# RUN EXPERIMENTS
# ============================================================================

for exp in experiments:
    print("\n" + "="*80)
    print(f"EXPERIMENT: {exp['name']}")
    print(f"Description: {exp['description']}")
    print(f"Expected accuracy: {exp['expected_acc']}")
    print("="*80 + "\n")
    
    # Get num_blocks if it exists
    num_blocks = exp.get("num_blocks", 12)

    # Training config for logging
    pretrained = True
    lr = 1e-4
    weight_decay = 0.01
    scheduler_type = "warmup_cosine"
    
    # Create model
    model = LitNetwork(
        model_name=exp["model_name"],
        pretrained=True,
        freeze_backbone=False,
        lr=lr,
        peak_lr=lr,
        weight_decay=weight_decay,
        warmup_epochs=3,
        rampup_epochs=5,
        num_epochs=num_epochs,
        num_blocks=num_blocks,
        scheduler_type=scheduler_type
    )
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        monitor='val_acc_epoch',
        save_top_k=1,
        mode='max',
        filename=f'{exp["name"]}-{{epoch:02d}}-{{val_acc_epoch:.3f}}'
    )
    
    early_stop = EarlyStopping(
        monitor='val_acc_epoch',
        patience=15,  # Reduced from 20 for faster stopping
        mode='max',
        verbose=True
    )
    
    #Create descriptive version string with all metadata
    pretrained_str = "pretrained" if pretrained else "scratch"
    version_str = (
        f"{exp['name']}_"
        f"{pretrained_str}_"
        f"blocks{num_blocks}_"
        f"{scheduler_type}"
    )

    logger = pl_loggers.TensorBoardLogger(
        save_dir="logs_comparison",
        name=exp["name"],         #Group by model type
        version=version_str       #detailed config in version
    )
    
    # Trainer with mixed precision for memory efficiency
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator=device,
        callbacks=[checkpoint, early_stop],
        logger=logger,
        log_every_n_steps=50,
        precision="16-mixed"  # Saves memory and speeds up training
    )
    
    # Train
    print(f"Training {exp['name']}...")
    trainer.fit(model, train_loader, val_loader)
    
    # Test with best checkpoint
    print(f"Testing {exp['name']}...")
    test_results = trainer.test(ckpt_path="best", dataloaders=test_loader)
    
    # Extract metrics
    final_train_acc = trainer.callback_metrics.get('train_acc_epoch', 0).item()
    final_val_acc = trainer.callback_metrics.get('val_acc_epoch', 0).item()
    test_acc = test_results[0]['test_acc']
    
    # Store results
    result = {
        "Experiment": exp["name"],
        "Model": exp["model_name"],
        "Description": exp["description"],
        "Train Acc": f"{final_train_acc*100:.2f}%",
        "Val Acc": f"{final_val_acc*100:.2f}%",
        "Test Acc": f"{test_acc*100:.2f}%",
        "Epochs Trained": trainer.current_epoch,
        "Expected": exp["expected_acc"]
    }
    results.append(result)
    
    print(f"\n{exp['name']} Results:")
    print(f"  Train Acc: {final_train_acc*100:.2f}%")
    print(f"  Val Acc: {final_val_acc*100:.2f}%")
    print(f"  Test Acc: {test_acc*100:.2f}%")
    
    # Clear GPU memory
    del model
    torch.cuda.empty_cache()

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "="*80)
print("COMPARISON RESULTS SUMMARY")
print("="*80 + "\n")

df = pd.DataFrame(results)
print(df.to_string(index=False))

# Save to CSV
df.to_csv("baseline_comparison_results.csv", index=False)
print("\n✓ Results saved to 'baseline_comparison_results.csv'")

# Analysis
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

# Extract validation accuracies
baseline_vit_val = float(results[0]["Val Acc"].strip('%'))
resnet18_val = float(results[1]["Val Acc"].strip('%'))
resnet50_val = float(results[2]["Val Acc"].strip('%'))
efficientnet_val = float(results[3]["Val Acc"].strip('%'))
custom_12_val = float(results[4]["Val Acc"].strip('%'))
custom_10_val = float(results[5]["Val Acc"].strip('%'))
custom_6_val = float(results[6]["Val Acc"].strip('%'))

print(f"\n=== BASELINE MODELS ===")
print(f"Standard ViT-Small: {baseline_vit_val:.2f}%")
print(f"ResNet18: {resnet18_val:.2f}%")
print(f"ResNet50: {resnet50_val:.2f}%")
print(f"EfficientNet-B0: {efficientnet_val:.2f}%")

print(f"\n=== YOUR CUSTOM MODELS ===")
print(f"Custom ViT (12 blocks): {custom_12_val:.2f}%")
print(f"Custom ViT (10 blocks): {custom_10_val:.2f}%")
print(f"Custom ViT (6 blocks): {custom_6_val:.2f}%")

# Compare against best baseline
best_baseline = max(baseline_vit_val, resnet18_val, resnet50_val, efficientnet_val)
best_baseline_name = ["ViT-Small", "ResNet18", "ResNet50", "EfficientNet-B0"][
    [baseline_vit_val, resnet18_val, resnet50_val, efficientnet_val].index(best_baseline)
]

print(f"\n=== COMPARISON TO BEST BASELINE ({best_baseline_name}: {best_baseline:.2f}%) ===")

diff_12 = custom_12_val - best_baseline
diff_10 = custom_10_val - best_baseline
diff_6 = custom_6_val - best_baseline

print(f"Custom 12-block vs Best: {diff_12:+.2f}%")
print(f"Custom 10-block vs Best: {diff_10:+.2f}%")
print(f"Custom 6-block vs Best: {diff_6:+.2f}%")

print("\n=== INTERPRETATION ===")
if diff_12 > 2:
    print("✓ Custom spatial attention HELPS! It improves over baseline.")
    print("  → Your 5-way spatial inductive bias is beneficial")
    print("  → Strong contribution for your research paper")
elif diff_12 > -2:
    print("≈ Custom spatial attention has NEUTRAL effect (within 2% of baseline)")
    print("  → Similar performance with added structure")
    print("  → Could argue for interpretability benefits")
    print("  → Analyze where it helps vs hurts")
else:
    print("✗ Custom spatial attention HURTS performance")
    print("  → Spatial restrictions may be too limiting")
    print("  → Consider:")
    print("    - Using custom attention on fewer blocks (hybrid model)")
    print("    - Softening the masks (weighted instead of binary)")
    print("    - Analyzing which spatial directions are problematic")

# Capacity analysis
capacity_loss_12_10 = custom_12_val - custom_10_val
capacity_loss_10_6 = custom_10_val - custom_6_val

print(f"\n=== CAPACITY ANALYSIS ===")
print(f"12 blocks → 10 blocks: {capacity_loss_12_10:+.2f}%")
print(f"10 blocks → 6 blocks: {capacity_loss_10_6:+.2f}%")

if capacity_loss_12_10 > 3:
    print("  → Significant loss from removing 2 blocks")
    print("  → Recommend using 12 blocks in final model")
else:
    print("  → Minimal capacity loss")
    print("  → 10 blocks sufficient (faster inference)")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("\n1. View training curves:")
print("   tensorboard --logdir=logs_comparison")
print("\n2. If custom attention helps:")
print("   → Analyze attention patterns")
print("   → Try hybrid models (custom on some blocks)")
print("   → Write strong motivation in paper")
print("\n3. If custom attention neutral/hurts:")
print("   → Analyze failure cases")
print("   → Consider modifications:")
print("     - Learnable masks")
print("     - Softer spatial biases")
print("     - Task-specific mask design")
print("\n4. For your paper:")
print("   → Include this comparison table")
print("   → Discuss trade-offs")
print("   → Show where spatial attention helps most")
print("\n5. Architecture insights:")
print("   → Compare CNN (ResNet) vs Transformer (ViT) performance")
print("   → Analyze efficiency (EfficientNet) vs accuracy trade-offs")