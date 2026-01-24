import torch
import torch.nn as nn
from models import select_image_model, ViTLayerReduction
import numpy as np

print("="*60)
print("MODEL ARCHITECTURE DIAGNOSTIC")
print("="*60)

# Test with your actual model configuration
model_name = "ViTLayerReduction"
n_classes = 101

print(f"\nCreating model: {model_name}")
print(f"Number of classes: {n_classes}")

try:
    model = select_image_model(
        model_name=model_name, 
        n_classes=n_classes, 
        freeze_backbone=False,
        pretrained=True
    )
    print("✓ Model created successfully")
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    exit(1)

model.eval()

# Check model structure
print("\n" + "="*60)
print("MODEL STRUCTURE")
print("="*60)
print(f"Model type: {type(model).__name__}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

if trainable_params == 0:
    print("✗ ERROR: No trainable parameters! All layers frozen.")
else:
    print(f"✓ {trainable_params:,} trainable parameters")

# Check number of blocks
if hasattr(model, 'blocks'):
    print(f"\nNumber of transformer blocks: {len(model.blocks)}")
    print("Block types:")
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'attn'):
            attn_type = type(block.attn).__name__
            print(f"  Block {i}: {attn_type}")

# Test forward pass
print("\n" + "="*60)
print("FORWARD PASS TEST")
print("="*60)

batch_size = 4
dummy_input = torch.randn(batch_size, 3, 224, 224)
print(f"Input shape: {dummy_input.shape}")

try:
    with torch.no_grad():
        output = model(dummy_input)
    print(f"✓ Forward pass successful")
    print(f"Output shape: {output.shape}")
    
    # CRITICAL CHECK: Output shape must match n_classes
    if output.shape[1] != n_classes:
        print(f"✗ CRITICAL ERROR: Output has {output.shape[1]} classes but expected {n_classes}!")
        print("  This will cause training to fail completely.")
    else:
        print(f"✓ Output shape matches number of classes ({n_classes})")
    
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Check output statistics
print("\n" + "="*60)
print("OUTPUT STATISTICS")
print("="*60)
print(f"Output min: {output.min().item():.3f}")
print(f"Output max: {output.max().item():.3f}")
print(f"Output mean: {output.mean().item():.3f}")
print(f"Output std: {output.std().item():.3f}")

# Check if outputs are reasonable (not all zeros, not extreme values)
if output.std().item() < 0.01:
    print("⚠️  WARNING: Output has very low variance - model may not be learning")
if abs(output.mean().item()) > 100:
    print("⚠️  WARNING: Output values are very large - possible numerical instability")

# Apply softmax and check probabilities
probs = torch.softmax(output, dim=1)
print(f"\nAfter softmax:")
print(f"Probability min: {probs.min().item():.6f}")
print(f"Probability max: {probs.max().item():.6f}")
print(f"Sum of probabilities (should be ~1.0): {probs[0].sum().item():.6f}")

# Check predictions
predictions = output.argmax(dim=1)
print(f"\nPredictions for batch: {predictions.tolist()}")
print(f"Unique predictions: {len(torch.unique(predictions))} out of {batch_size} samples")

if len(torch.unique(predictions)) == 1:
    print("⚠️  WARNING: All predictions are the same class - model is stuck")

# Test with actual data if available
print("\n" + "="*60)
print("TEST WITH REAL DATA")
print("="*60)

try:
    import torchvision
    from torch.utils.data import DataLoader
    
    dataset_dir = "data/"
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = torchvision.datasets.Food101(root=dataset_dir, split='train', transform=transforms, download=False)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    
    batch = next(iter(train_loader))
    images, labels = batch
    
    print(f"Real batch - Images shape: {images.shape}")
    print(f"Real batch - Labels: {labels.tolist()}")
    print(f"Real batch - Label range: [{labels.min().item()}, {labels.max().item()}]")
    
    with torch.no_grad():
        outputs = model(images)
    
    print(f"\nModel outputs shape: {outputs.shape}")
    print(f"Output range: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
    
    # Compute loss
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs, labels)
    print(f"\nCrossEntropyLoss: {loss.item():.4f}")
    
    # Expected loss for random guessing
    random_loss = -np.log(1.0 / n_classes)
    print(f"Expected loss for random guessing: {random_loss:.4f}")
    
    if loss.item() > random_loss * 1.5:
        print("⚠️  WARNING: Loss is much higher than random - model may be predicting very wrong")
    elif loss.item() < random_loss * 0.5:
        print("✓ Loss is better than random - model has some signal")
    else:
        print("→ Loss is close to random guessing")
    
    # Compute accuracy
    preds = outputs.argmax(dim=1)
    accuracy = (preds == labels).float().mean()
    print(f"\nBatch accuracy: {accuracy.item()*100:.2f}%")
    print(f"Random guessing would be: {100.0/n_classes:.2f}%")
    
    if accuracy.item() < 0.001:
        print("✗ CRITICAL: Accuracy is essentially 0% - something is very wrong")
    
    print(f"\nPredictions: {preds.tolist()}")
    print(f"Unique predictions in batch: {len(torch.unique(preds))}")
    
except Exception as e:
    print(f"Could not test with real data: {e}")

# Check gradient flow
print("\n" + "="*60)
print("GRADIENT FLOW TEST")
print("="*60)

model.train()
dummy_input = torch.randn(2, 3, 224, 224)
dummy_labels = torch.tensor([0, 1])

try:
    outputs = model(dummy_input)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs, dummy_labels)
    
    print(f"Loss value: {loss.item():.4f}")
    
    loss.backward()
    
    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if grad_norm == 0:
                print(f"⚠️  {name}: gradient is zero!")
    
    if len(grad_norms) > 0:
        print(f"✓ Gradients computed for {len(grad_norms)} parameters")
        print(f"Gradient norm range: [{min(grad_norms):.6f}, {max(grad_norms):.6f}]")
        print(f"Mean gradient norm: {np.mean(grad_norms):.6f}")
        
        if max(grad_norms) > 100:
            print("⚠️  WARNING: Very large gradients detected - possible exploding gradients")
        if max(grad_norms) < 1e-7:
            print("⚠️  WARNING: Very small gradients - possible vanishing gradients")
    else:
        print("✗ ERROR: No gradients computed!")
        
except Exception as e:
    print(f"✗ Gradient test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
print("\nKey findings to report:")
print("1. Does output shape match n_classes?")
print("2. Is loss close to random guessing (~4.6)?")
print("3. Are gradients flowing (not zero, not exploding)?")
print("4. Are all predictions the same class?")