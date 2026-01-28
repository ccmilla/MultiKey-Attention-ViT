import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Food101
import pytorch_lightning as pl
import timm
import math
import numpy as np
from timm import create_model


# ==============
# DataModule
# ==============

class Food101DataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def setup(self,stage=None):
        self.train_dataset = Food101(
                                root=self.data_dir,
                                split='train',
                                transform=self.train_transform,
                                download=True)
        self.val_dataset = Food101(
                                root=self.data_dir,
                                split='test',
                                transform=self.val_transform,
                                download=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          persistent_workers=True)

#===============
# getRowsAndCols - Maps every image patch to row/column coordinates
#===============
def getRowsAndCols(height, width, patch_size):
    '''Given an image tensor of shape (C, H, W) and a patch size,
       compute the center row and column of each patch.
       '''
    num_patches_y = height // patch_size
    num_patches_x = width // patch_size
    token_rows = []
    token_cols = []
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            center_row = i * patch_size + patch_size / 2
            center_col = j * patch_size + patch_size / 2
            token_rows.append(center_row)
            token_cols.append(center_col)
    return np.array(token_rows), np.array(token_cols)


#===============
# makeMasks - Creates 5 directional spatial masks
#===============
def makeMasks(token_rows, token_cols):
    """
    Generate 5 spatial masks:
      left_mask, right_mask, up_mask, down_mask, identity_mask
    Each is (num_patches, num_patches).
    """
    num_patches = len(token_rows)
    left_mask = np.zeros((num_patches, num_patches))
    right_mask = np.zeros((num_patches, num_patches))
    up_mask = np.zeros((num_patches, num_patches))
    down_mask = np.zeros((num_patches, num_patches))
    
    for i in range(num_patches):
        for j in range(num_patches):
            if token_cols[i] < token_cols[j]:
                left_mask[i, j] = 1
            if token_cols[i] > token_cols[j]:
                right_mask[i, j] = 1
            if token_rows[i] < token_rows[j]:
                up_mask[i, j] = 1
            if token_rows[i] > token_rows[j]:
                down_mask[i, j] = 1

    identity_mask = np.eye(num_patches)
    return left_mask, right_mask, up_mask, down_mask, identity_mask

#===============
# Custom 5-way Spatial Attention Module
#===============
class CustomAttentionMultipleFiveSpatial(nn.Module):
    def __init__(self, orig_attn: nn.Module, patch_size=16, img_size=224):
        super().__init__()
        self.num_heads = orig_attn.num_heads
        self.embed_dim = orig_attn.qkv.in_features
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.patch_size = patch_size
        self.img_size = img_size
        
        # Create spatial masks
        token_rows, token_cols = getRowsAndCols(img_size, img_size, patch_size)
        left_mask, right_mask, up_mask, down_mask, identity_mask = makeMasks(token_rows, token_cols)
        
        # Pad for cls token
        left_mask = np.pad(left_mask, ((1, 0), (1, 0)), mode='constant', constant_values=0)
        right_mask = np.pad(right_mask, ((1, 0), (1, 0)), mode='constant', constant_values=0)
        up_mask = np.pad(up_mask, ((1, 0), (1, 0)), mode='constant', constant_values=0)
        down_mask = np.pad(down_mask, ((1, 0), (1, 0)), mode='constant', constant_values=0)
        identity_mask = np.pad(identity_mask, ((1, 0), (1, 0)), mode='constant', constant_values=0)

        # Register as buffers
        self.register_buffer("left_mask", torch.tensor(left_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.register_buffer("right_mask", torch.tensor(right_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.register_buffer("up_mask", torch.tensor(up_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.register_buffer("down_mask", torch.tensor(down_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.register_buffer("identity_mask", torch.tensor(identity_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
    
        # Q, V, and 5 K projections
        self.q_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.kA_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.kB_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.kC_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.kD_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.kE_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj = orig_attn.proj

        # Initialize from original QKV weights
        qkv_weight = orig_attn.qkv.weight.clone()
        qkv_bias = orig_attn.qkv.bias.clone() if orig_attn.qkv.bias is not None else None

        self.q_linear.weight.data.copy_(qkv_weight[:self.embed_dim, :].clone())
        self.kA_linear.weight.data.copy_(qkv_weight[self.embed_dim:2*self.embed_dim, :].clone())
        self.kB_linear.weight.data.copy_(qkv_weight[self.embed_dim:2*self.embed_dim, :].clone())
        self.kC_linear.weight.data.copy_(qkv_weight[self.embed_dim:2*self.embed_dim, :].clone())
        self.kD_linear.weight.data.copy_(qkv_weight[self.embed_dim:2*self.embed_dim, :].clone())
        self.kE_linear.weight.data.copy_(qkv_weight[self.embed_dim:2*self.embed_dim, :].clone())
        self.v_linear.weight.data.copy_(qkv_weight[2*self.embed_dim:, :].clone())

        if qkv_bias is not None:
            self.q_linear.bias.data.copy_(qkv_bias[:self.embed_dim].clone())
            self.kA_linear.bias.data.copy_(qkv_bias[self.embed_dim:2*self.embed_dim].clone())
            self.kB_linear.bias.data.copy_(qkv_bias[self.embed_dim:2*self.embed_dim].clone())
            self.kC_linear.bias.data.copy_(qkv_bias[self.embed_dim:2*self.embed_dim].clone())
            self.kD_linear.bias.data.copy_(qkv_bias[self.embed_dim:2*self.embed_dim].clone())
            self.kE_linear.bias.data.copy_(qkv_bias[self.embed_dim:2*self.embed_dim].clone())
            self.v_linear.bias.data.copy_(qkv_bias[2*self.embed_dim:].clone())

    def forward(self, x, **kwargs):
        query = x
        B, N, _ = query.shape
        
        # Project Q, K, V
        q = self.q_linear(query).reshape(B, N, self.num_heads, self.head_dim)
        kA = self.kA_linear(query).reshape(B, N, self.num_heads, self.head_dim)
        kB = self.kB_linear(query).reshape(B, N, self.num_heads, self.head_dim)
        kC = self.kC_linear(query).reshape(B, N, self.num_heads, self.head_dim)
        kD = self.kD_linear(query).reshape(B, N, self.num_heads, self.head_dim)
        kE = self.kE_linear(query).reshape(B, N, self.num_heads, self.head_dim)
        v = self.v_linear(query).reshape(B, N, self.num_heads, self.head_dim)

        # Transpose to [B, num_heads, N, head_dim]
        q = q.transpose(1, 2) * self.scale
        kA = kA.transpose(1, 2)
        kB = kB.transpose(1, 2)
        kC = kC.transpose(1, 2)
        kD = kD.transpose(1, 2)
        kE = kE.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply masks BEFORE softmax
        def apply_mask(attn_scores, mask):
            mask_additive = (1.0 - mask) * -1e9
            return attn_scores + mask_additive
        
        # Compute attention scores
        attn_scores_ka = q @ kA.transpose(-2, -1)
        attn_scores_kb = q @ kB.transpose(-2, -1)
        attn_scores_kc = q @ kC.transpose(-2, -1)
        attn_scores_kd = q @ kD.transpose(-2, -1)
        attn_scores_ke = q @ kE.transpose(-2, -1)
        
        # Apply directional masks
        attn_scores_ka = apply_mask(attn_scores_ka, self.left_mask)
        attn_scores_kb = apply_mask(attn_scores_kb, self.right_mask)
        attn_scores_kc = apply_mask(attn_scores_kc, self.up_mask)
        attn_scores_kd = apply_mask(attn_scores_kd, self.down_mask)
        attn_scores_ke = apply_mask(attn_scores_ke, self.identity_mask)

        # Softmax
        attn_ka = attn_scores_ka.softmax(dim=-1)
        attn_kb = attn_scores_kb.softmax(dim=-1)
        attn_kc = attn_scores_kc.softmax(dim=-1)
        attn_kd = attn_scores_kd.softmax(dim=-1)
        attn_ke = attn_scores_ke.softmax(dim=-1)

        # Apply attention to values
        out_a = attn_ka @ v
        out_b = attn_kb @ v
        out_c = attn_kc @ v
        out_d = attn_kd @ v
        out_e = attn_ke @ v
        
        # Sum all outputs
        out = out_a + out_b + out_c + out_d + out_e

        out = out.transpose(1, 2).reshape(B, N, self.embed_dim)
        out = self.proj(out)
        return out

# ===============
# ViT with Layer Reduction and Custom Attention
# ===============

class ViTLayerReduction(nn.Module):
    def __init__(self, num_blocks_to_keep, patch_size=16, num_classes=101, img_size=224, pretrained=True):
        super().__init__()
        full_model = create_model(
            "vit_small_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=0.3,
            drop_path_rate=0.1
        )
        
        # Extract components
        self.patch_embed = full_model.patch_embed
        self.cls_token = full_model.cls_token
        self.pos_embed = full_model.pos_embed
        self.pos_drop = full_model.pos_drop
        
        # Inject custom attention into blocks
        self.blocks = nn.Sequential()
        for i, block in enumerate(full_model.blocks[:num_blocks_to_keep]):
            if hasattr(block, 'attn'):
                block.attn = CustomAttentionMultipleFiveSpatial(block.attn, patch_size=patch_size, img_size=img_size)
                print(f"Block {i}: Custom 5-way spatial attention injected.")
            else:
                raise AttributeError(f"Block {i} has no attention module.")
            self.blocks.append(block)

        self.norm = full_model.norm
        self.head = full_model.head

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1) 
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:,0])

#=============
# Model Selection Function
# ============      
def select_image_model(
        model_name="ViTLayerReduction",
        n_classes=101,
        freeze_backbone=False,
        pretrained=False,
        num_blocks=12
        ):
    """
    Select and configure model architecture.
    
    Supported models:
    - resnet18tv: ResNet18 from torchvision
    - resnet18timm: ResNet18 from timm
    - resnet50tv: ResNet50 from torchvision
    - efficientnet_b0: EfficientNet-B0 from timm
    - ViTLayerReduction: Custom ViT with 5-way spatial attention
    - Any timm model name (e.g., vit_small_patch16_224)
    """
    
    if model_name == "resnet18tv":
        model = torchvision.models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, n_classes)

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
                
    elif model_name == "resnet18timm":
        model = timm.create_model("resnet18", pretrained=pretrained, num_classes=n_classes)

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
                
    elif model_name == "resnet50tv":
        model = torchvision.models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, n_classes)

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
                
    elif model_name == "efficientnet_b0":
        model = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=n_classes)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
                
    elif model_name == "ViTLayerReduction":
        model = ViTLayerReduction(
                    num_blocks_to_keep=num_blocks,
                    patch_size=16, 
                    img_size=224, 
                    pretrained=pretrained,
                    num_classes=n_classes
                    ) 
    else:
        # Default to timm model
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=n_classes)

    return model