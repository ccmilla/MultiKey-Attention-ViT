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

'''1) Load the Food101 datset
   2) Apply Transforms (data augmentations for training, resizing for validation)
   3) Return dataloaders with eht ecorrect batch size (images per patch), shuffling, and workers
   num_workers - background threads to load data
   persistent_workers - true means keeps worker threads alive which makes it faster'''

#==========
# This doesn't get called - it is in PyTorchLgtAttTemplate main
#===========
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
# getRowsAndCols also called by CustomAttentionMultipleFiveSpatial
# Maps every image patch to the row and column coordinates of its center, so later code can reason about spatial
# relationships between patches.
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
# makeMasks called by CustomAttentionMultipleFiveSpatial
# This function is crafting five spatial relationhip masks between image patches.
# It builds five square masks, each [i,j] pair describes a relationship patch (query) and patch j (key)
# Creates a kind of compass-guided attention field, letting the transformer focus along structured spatial directions.
# These masks will later modulate multiple key projections in a specialized spatial-attention block.
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
    # column(i) < column(j) inverted naming
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

    identity_mask = np.eye(num_patches)  # patches attend to themeselves
    return left_mask, right_mask, up_mask, down_mask, identity_mask

#===============
# Custom Attention Multiple Five Spatial Model called by ViTLayerReduction
#===============
class CustomAttentionMultipleFiveSpatial(nn.Module):
    def __init__(self, orig_attn: nn.Module, patch_size=16, img_size=224):
        super().__init__()
        self.num_heads = orig_attn.num_heads
        self.embed_dim = orig_attn.qkv.in_features
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        '''added by Dr. Hart - overfitting issue or attribute error
        This determines how many patch tokens the input image will have.
        Creates directional masks for each token: left, right, upward, downward, identity (token itself).
        Adds padding for the CLS token so all masks align with the final ViT sequence length.
        Prepares masks that can be applied to attention maps or token-selection mechanisms.'''
        self.patch_size = patch_size
        self.img_size = img_size
        token_rows, token_cols = getRowsAndCols(img_size, img_size, patch_size)
        left_mask, right_mask, up_mask, down_mask, identity_mask = makeMasks(token_rows, token_cols)
        # Pad for cls token
        # Adds a zero row and zero column at index 0 (1 row top and 1 column left).  No spatial neighbors.
        # Modifies self-attention
        # Adds structural inductive bias
        # Implements directional masked autoencoding
        # Restricts token mixing to neighbors
        # Adds spatial locality to ViTs
        left_mask = np.pad(left_mask, ((1, 0), (1, 0)), mode='constant', constant_values=0)
        right_mask = np.pad(right_mask, ((1, 0), (1, 0)), mode='constant', constant_values=0)
        up_mask = np.pad(up_mask, ((1, 0), (1, 0)), mode='constant', constant_values=0)
        down_mask = np.pad(down_mask, ((1, 0), (1, 0)), mode='constant', constant_values=0)
        identity_mask = np.pad(identity_mask, ((1, 0), (1, 0)), mode='constant', constant_values=0)

        self.register_buffer("left_mask", torch.tensor(left_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.register_buffer("right_mask", torch.tensor(right_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.register_buffer("up_mask", torch.tensor(up_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.register_buffer("down_mask", torch.tensor(down_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.register_buffer("identity_mask", torch.tensor(identity_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
    
        #original code
        self.q_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.kA_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.kB_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.kC_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.kD_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.kE_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj = orig_attn.proj

        qkv_weight = orig_attn.qkv.weight.clone()
        qkv_bias = orig_attn.qkv.bias.clone() if orig_attn.qkv.bias is not None else None

        '''Dr. Hart's addition -
        Takes a single fused QKV weight matrix from a Vision Transformer (shape= (3*embed_dim, embed_dim))
        Splits it into 1 query, 5 key projections (kA, kB, kC, kD, kE), 1 value projection(V)
        Copies slices into custom linear layers. 
        Creating multiple K projections that uses direction-specific key matrices matching
        the directional masks
        ┌─────────────────────────────┐
        │   Q weights   (embed_dim)   │   ← slice [0 : embed_dim]
        ├─────────────────────────────┤
        │   K weights   (embed_dim)   │   ← slice [embed_dim : 2*embed_dim]
        ├─────────────────────────────┤
        │   V weights   (embed_dim)   │   ← slice [2*embed_dim : 3*embed_dim]
        └─────────────────────────────┘
        '''
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

        '''Original Code'''
        #self.q_linear.weight.data.copy_(qkv_weight[:self.embed_dim, :])
        # for k_lin in [self.kA_linear, self.kB_linear, self.kC_linear, self.kD_linear, self.kE_linear]:
        #     k_lin.weight.data.copy_(qkv_weight[self.embed_dim:2*self.embed_dim, :])
        # self.v_linear.weight.data.copy_(qkv_weight[2*self.embed_dim:, :])
        #if qkv_bias is not None:
            # self.q_linear.bias.data.copy_(qkv_bias[:self.embed_dim])
            # for k_lin in [self.kA_linear, self.kB_linear, self.kC_linear, self.kD_linear, self.kE_linear]:
            #     k_lin.bias.data.copy_(qkv_bias[self.embed_dim:2*self.embed_dim])
            # self.v_linear.bias.data.copy_(qkv_bias[2*self.embed_dim:])

    # def forward(self, query, attn_mask=None):
    #     if not hasattr(self, "current_mask_list"):
    #         raise ValueError("current_mask_list attribute not set in CustomAttentionMultipleFiveSpatial!")
    #     left_mask, right_mask, up_mask, down_mask, identity_mask = self.current_mask_list
    #     B, N, _ = query.shape
    #     q = self.q_linear(query).reshape(B, N, self.num_heads, -1).transpose(1, 2) * self.scale
    #     v = self.v_linear(query).reshape(B, N, self.num_heads, -1).transpose(1, 2)
    #     ks = [self.kA_linear, self.kB_linear, self.kC_linear, self.kD_linear, self.kE_linear]
    #     masks = [left_mask, right_mask, up_mask, down_mask, identity_mask]
    #     out = sum((q @ k(query).reshape(B, N, self.num_heads, -1).transpose(1, 2).transpose(-2, -1)).softmax(-1) * m @ v for k, m in zip(ks, masks))
    #     if left_mask.shape[1] != self.num_heads:
    #         left_mask = left_mask[:, :self.num_heads, :, :]
    #         right_mask = right_mask[:, :self.num_heads, :, :]
    #         up_mask = up_mask[:, :self.num_heads, :, :]
    #         down_mask = down_mask[:, :self.num_heads, :, :]
    #         identity_mask = identity_mask[:, :self.num_heads, :, :]

    #     return self.proj(out.transpose(1, 2).reshape(B, N, -1))

    def forward(self, x, **kwargs):
        # Ignore any extra kwargs like attn_mask that timm might pass
        query = x

        B, N, _ = query.shape
        q = self.q_linear(query).reshape(B, N, self.num_heads, self.head_dim)
        kA = self.kA_linear(query).reshape(B, N, self.num_heads, self.head_dim)
        kB = self.kB_linear(query).reshape(B, N, self.num_heads, self.head_dim)
        kC = self.kC_linear(query).reshape(B, N, self.num_heads, self.head_dim)
        kD = self.kD_linear(query).reshape(B, N, self.num_heads, self.head_dim)
        kE = self.kE_linear(query).reshape(B, N, self.num_heads, self.head_dim)
        v = self.v_linear(query).reshape(B, N, self.num_heads, self.head_dim)

        q = q.transpose(1, 2) * self.scale
        kA = kA.transpose(1, 2)
        kB = kB.transpose(1, 2)
        kC = kC.transpose(1, 2)
        kD = kD.transpose(1, 2)
        kE = kE.transpose(1, 2)
        v = v.transpose(1, 2)

        # print(left_mask.shape)
        # print(q.shape)
        # print(kA.shape)
        # print((q @ kA.transpose(-2, -1)).softmax(dim=-1).shape)
        
        # ============================================================
        # CRITICAL FIX: Apply masks BEFORE softmax using additive mask
        # ============================================================
        # Convert binary mask to additive mask: where mask == 0, add 149 (effectively -inf)
        # This way softmax will make those positions ~0 probability

        def apply_mask(attn_scores, mask):
            # mask shape: [1, 1, N, N] or [N, N]
            # attn_scores shape: [B, num_heads, N, N]
            # Where mask is 0, we want to prevent attention (set to -inf before softmax)
            # Where mask is 1, we want to allow attention (leave as is)
            mask_additive = (1.0 - mask) * -1e9 # 0->0, 1->-1e9, then invert
            return attn_scores + mask_additive
        
        # Compute attention scores (before softmax)
        attn_scores_ka = q @ kA.transpose(-2, -1)
        attn_scores_kb = q @ kB.transpose(-2, -1)
        attn_scores_kc = q @ kC.transpose(-2, -1)
        attn_scores_kd = q @ kD.transpose(-2, -1)
        attn_scores_ke = q @ kE.transpose(-2, -1)
        
        # Apply masks BEFORE softmax
        attn_scores_ka = apply_mask(attn_scores_ka, self.left_mask)
        attn_scores_kb = apply_mask(attn_scores_kb, self.right_mask)
        attn_scores_kc = apply_mask(attn_scores_kc, self.up_mask)
        attn_scores_kd = apply_mask(attn_scores_kd, self.down_mask)
        attn_scores_ke = apply_mask(attn_scores_ke, self.identity_mask)

        # Now apply softmax (gradients will flow through!)
        attn_ka = attn_scores_ka.softmax(dim=-1)
        attn_kb = attn_scores_kb.softmax(dim=-1)
        attn_kc = attn_scores_kc.softmax(dim=-1)
        attn_kd = attn_scores_kd.softmax(dim=-1)
        attn_ke = attn_scores_ke.softmax(dim=-1)

        '''The masks (left, right, up, down, and identity) are still being applied
        to create directional attention patterns.  The only thing that changed is
        how we apply them.  We are applying masks BEFORE softmax by adding -inf (gradients flow)
        '''
        # below is the old code
        # attn_ka = (q @ kA.transpose(-2, -1)).softmax(dim=-1) * self.left_mask
        # attn_kb = (q @ kB.transpose(-2, -1)).softmax(dim=-1) * self.right_mask
        # attn_kc = (q @ kC.transpose(-2, -1)).softmax(dim=-1) * self.up_mask
        # attn_kd = (q @ kD.transpose(-2, -1)).softmax(dim=-1) * self.down_mask
        # attn_ke = (q @ kE.transpose(-2, -1)).softmax(dim=-1) * self.identity_mask

        #apply attention values
        out_a = attn_ka @ v
        out_b = attn_kb @ v
        out_c = attn_kc @ v
        out_d = attn_kd @ v
        out_e = attn_ke @ v
        #sum all outputs
        out = out_a + out_b + out_c + out_d + out_e

        out = out.transpose(1, 2).reshape(B, N, self.embed_dim)
        out = self.proj(out)
        return out

# ===============
# ViT model
# ===============

'''Loading a full ViT model (vit_small_patch16_224)
   Keeping on a subset of the transformer blocks
   Keeping all patch embedding and classification heads
   Running the forward pass just like ViT, but with fewer layers
   It is "reduced" ViT that uses fewer transformer blocks
   Choose how many blocks with num_blocks
   **smaller, faster ViT**
   uses only the first N layers of a standard ViT model
   while keeping the same classification head.
'''
class ViTLayerReduction(nn.Module):
    #added patch size and img_size
    #num_blocks_to_keep for consistency
    def __init__(self, num_blocks_to_keep, patch_size=16, num_classes=101, img_size=224, pretrained=True): #added pretrained
        super().__init__()
        full_model = create_model(
            "vit_small_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=0.3,  #changing back to original
            drop_path_rate=0.1 #changing back to orginal
        )
        #extracting important internal modules
        self.patch_embed = full_model.patch_embed #converts image patches into vectors
        self.cls_token = full_model.cls_token #special 'classification token'
        self.pos_embed = full_model.pos_embed #position embeddings
        self.pos_drop = full_model.pos_drop #drop out after adding positions
        '''ViT normally has 12 blocks (ViT-small)  We keep 10 out of 12
        #self.blocks = nn.Sequential(*list(full_model.blocks[:num_blocks])) 
        #nn.Sequential wraps modules and sometimes hides attributes, including num_heads
        #nn.ModuleList preserves every block exactly as built.
        #self.blocks = nn.ModuleList(list(full_model.blocks[:num_blocks]))
        #convert CLS token into class logits'''
        #Changes from Dr. Hart for the overfitting issue
        self.blocks = nn.Sequential()
        for i, block in enumerate(full_model.blocks[:num_blocks_to_keep]):
            if hasattr(block, 'attn'):
                block.attn = CustomAttentionMultipleFiveSpatial(block.attn, patch_size=patch_size, img_size=img_size)
                print(f"Block {i}: Custom attention injected.")
            else:
                raise AttributeError(f"Block {i} has no attention module.")
            self.blocks.append(block)

        self.norm = full_model.norm
        self.head = full_model.head

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) # ViT converts each 16x16 patch 384 dim vector
        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1) 
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)  #pass through reduced transformer blocks
        x = self.norm(x)
        return self.head(x[:,0]) # the CLS token output that becomes logits over 101 classes.

#=============
# Image-model dispatcher written by Dr. Hart  called by PyTorchLgtAttTemplate
# Given the model name, class count, and free/fine-tune preference, and it hands back
# the right neural network configured for training
# ============      
def select_image_model(model_name="ViTLayerReduction", n_classes=5, freeze_backbone=False, pretrained=False):
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
    elif model_name == "ViTLayerReduction":
        model = ViTLayerReduction(num_blocks_to_keep=10, patch_size=16, img_size=224, pretrained=pretrained) #keeping 10 blocks and added pretrained
    else:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=n_classes)

    return model