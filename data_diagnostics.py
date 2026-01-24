import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

dataset_dir = "data/"
b = 64
width = 224
height = 224

transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop((height,width)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

print("Loading datasets...")
train_dataset = torchvision.datasets.Food101(root=dataset_dir, split='train', transform=transforms, download=True)
train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset, 
    [int(len(train_dataset)*0.8), len(train_dataset) - int(len(train_dataset)*0.8)]
)
test_dataset = torchvision.datasets.Food101(root=dataset_dir, split='test', transform=transforms, download=True)

print("\n" + "="*60)
print("DATASET SIZES")
print("="*60)
print(f"Train dataset: {len(train_dataset)} samples")
print(f"Val dataset: {len(val_dataset)} samples")
print(f"Test dataset: {len(test_dataset)} samples")
print(f"Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=b, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=b, shuffle=False, num_workers=0)

# Get one batch
print("\n" + "="*60)
print("BATCH INSPECTION")
print("="*60)
batch = next(iter(train_loader))
images, labels = batch

print(f"Batch image shape: {images.shape}")  # Should be [64, 3, 224, 224]
print(f"Batch labels shape: {labels.shape}")  # Should be [64]
print(f"Image dtype: {images.dtype}")
print(f"Labels dtype: {labels.dtype}")
print(f"\nFirst 16 labels in batch: {labels[:16].tolist()}")
print(f"Unique labels in batch: {len(torch.unique(labels))}")
print(f"Label range: [{labels.min().item()}, {labels.max().item()}]")

# Check image statistics
print("\n" + "="*60)
print("IMAGE STATISTICS (after normalization)")
print("="*60)
print(f"Image min: {images.min().item():.3f}")
print(f"Image max: {images.max().item():.3f}")
print(f"Image mean: {images.mean().item():.3f}")
print(f"Image std: {images.std().item():.3f}")

# IMPORTANT: Check if images are all zeros or have no variation
if images.std().item() < 0.01:
    print("⚠️  WARNING: Images have very low variance - possible data loading issue!")

# Visualize some samples (denormalize first)
print("\n" + "="*60)
print("VISUALIZING SAMPLES")
print("="*60)

def denormalize(tensor):
    """Reverse the normalization for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

# Plot 16 samples
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
axes = axes.flatten()

for i in range(16):
    img = denormalize(images[i]).permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)  # Clip to valid range
    axes[i].imshow(img)
    axes[i].set_title(f"Label: {labels[i].item()}", fontsize=10)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('data_diagnostic_samples.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to 'data_diagnostic_samples.png'")
print("  Check this image to verify:")
print("  - Images look like food")
print("  - Labels seem correct")
print("  - Images have variety (not all similar)")

# Label distribution across entire training set
print("\n" + "="*60)
print("LABEL DISTRIBUTION (checking for class imbalance)")
print("="*60)
print("Sampling 1000 examples to check distribution...")

sample_labels = []
sample_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=0)
for i, (imgs, lbls) in enumerate(sample_loader):
    sample_labels.extend(lbls.tolist())
    if len(sample_labels) >= 1000:
        break

label_counts = {}
for lbl in sample_labels[:1000]:
    label_counts[lbl] = label_counts.get(lbl, 0) + 1

print(f"Number of unique classes in sample: {len(label_counts)}")
print(f"Min class count: {min(label_counts.values())}")
print(f"Max class count: {max(label_counts.values())}")
avg_count = sum(label_counts.values()) / len(label_counts)
print(f"Average count per class: {avg_count:.1f}")

if max(label_counts.values()) / min(label_counts.values()) > 3:
    print("⚠️  WARNING: Significant class imbalance detected!")
else:
    print("✓ Classes appear reasonably balanced")

print("\n" + "="*60)
print("DATA DIAGNOSTIC COMPLETE")
print("="*60)
print("\nNext steps:")
print("1. Review 'data_diagnostic_samples.png' - do images look correct?")
print("2. Check if labels match what you see in the images")
print("3. If everything looks good, move to Step 2: Model Architecture Check")