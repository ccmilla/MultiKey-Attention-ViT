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
from timm.layers import DropPath  #adding stochastic depth to custom ViT

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
        #sum all outputs   Claude fix - also need to divide by 5 for average and not just the sum
        out = (out_a + out_b + out_c + out_d + out_e) / 5.0  #fix by Claude for nonpretrained datasets. Not dividing can destablize training.
        out = out.transpose(1, 2).reshape(B, N, self.embed_dim)
        out = self.proj(out)
        return out
    
#================
# DWConv bypass
#================
class BlockWithDWConv(nn.Module):
    #wraps a transformerblock and adds a depthwise convolution bypass path.
    def __init__(self, block, embed_dim, img_size=224, patch_size=16):
        super().__init__()
        self.block = block # original transformer block
        self.embed_dim = embed_dim

        #Depthwise convolution for local features
        self.dwconv = nn.Conv2d(
            embed_dim, embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim, #depthwise = each channel processed separately
            bias=True
        )
        #Projection to combine conv features back
        self.norm = nn.LayerNorm(embed_dim)
        #calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
        self.H = self.W = img_size // patch_size # 224//16 = 14

    def forward(self, x):
        B,N,C = x.shape

        # Path 1: original transformer block (attention + FFN)
        attn_out = self.block(x)

        # Path 2: depthwise convolution (local features)
        cls_token = x[:, :1, :] #(B,1,C)
        patch_tokens = x[:,1:,:] # (B, num_patches, C)
        #reshape to 2D spatial format for convultion
        #(B, num_patches, C) -> (B,H,W,C) -> (B,C,H,W)
        patch_tokens_2d = patch_tokens.reshape(B, self.H, self.W, C)
        patch_tokens_2d = patch_tokens_2d.permute(0,3,1,2) #(B,C, H, W)
        #apply depthwise convolution
        conv_out = self.dwconv(patch_tokens_2d) #(B,C,H,W)
        #reshape back to sequence format
        #(B,C,H,W) - > (B,C, num_patches) - > (B, num_patches, C)
        conv_out = conv_out.flatten(2).transpose(1,2) #(B, num_patches, C)
        # add cls token back
        conv_out = torch.cat([cls_token, conv_out], dim=1) #(B,N,C)
        #normalize conv output
        conv_out = self.norm(conv_out)
        #combine both paths
        #attn_out already includes residual (from origin block)
        #add conv features on top
        output = attn_out + conv_out
        return output
        
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
    def __init__(self, 
                 num_blocks_to_keep, 
                 patch_size, 
                 num_classes, 
                 img_size, 
                 pretrained,
                 drop_path_rate, # adding this parameter for stochastic depth
                 use_dwconv_bypass # passing through depth wise convolution
                 ): 
        super().__init__()
        full_model = create_model(
            "vit_small_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
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
        # Create linearly increasing drop path rates for stochastic depth
        dpr = [x.item() for x in torch.linspace (0, drop_path_rate, num_blocks_to_keep)]

        #Changes from Dr. Hart for the overfitting issue
        self.blocks = nn.Sequential()
        for i, block in enumerate(full_model.blocks[:num_blocks_to_keep]):
            if hasattr(block, 'attn'):
              #Add another conditional statement where we keep the last two blocks only
              if i >= num_blocks_to_keep - 2: # last blocks only then we call CustomAttention
                block.attn = CustomAttentionMultipleFiveSpatial(block.attn, patch_size=patch_size, img_size=img_size)
                print(f"Block {i}: Custom attention injected.")
            else:
                raise AttributeError(f"Block {i} has no attention module.")
            # add stochastic depth to all blocks
            if hasattr(block, 'drop_path'):
                block.drop_path = DropPath(dpr[i]) if dpr[i] > 0. else nn.Identity()
                print(f"Block {i}: Drop path rate = {dpr[i]:.4f}")
            #wrap block with DWConv bypass if requested
            if use_dwconv_bypass:
                block = BlockWithDWConv(
                    block,
                    embed_dim=384, #vit_small has 384 dim
                    img_size=img_size,
                    patch_size=patch_size
                )
                print(f"Block, {i}: DWConv bypass added.")
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
# DWConv model only - to test if Custom ViT is not helping
#=============
class StandardViTWithDWConv(nn.Module):
    #Standard vit-smallk with DWConv bypass added (no custom attention)
    def __init__(self, num_blocks_to_keep):
        super().__init__()

        #Load standard vit_small
        full_model = timm.create_model("vit_small_patch16_224")

        self.patch_embed = full_model.patch_embed
        self.cls_token = full_model.cls_token
        self.pos_embed = full_model.pos_embed
        self.pos_drop = full_model.pos_drop

        #Wrap blocks with DWConv (again no custom attention)
        self.blocks = nn.Sequential()
        for i, block in enumerate(full_model.blocks[:num_blocks_to_keep]):     
            block = BlockWithDWConv(
                block,
                embed_dim=384,
                img_size=224,
                patch_size=16
            )
            print(f"Block {i}: DWConv bypass added (standard attention)")
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
    
class LocalDirectionalAttention(nn.Module):
    ''' Local directional attention within a window.  Splits attention into 5 directional components,
        but only within local windows'''
    def __init__(self, orig_attn: nn.Module, window_size, patch_size=16, img_size=224):
        super().__init__()
        self.num_heads = orig_attn.num_heads
        self.embed_dim = orig_attn.qkv.in_features
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = 1.0 /math.sqrt(self.head_dim)

        self.window_size = window_size # e.g., 7x7 window
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches_per_side = img_size // patch_size

        #Create directional masks for LOCAL window
        self.create_local_directional_masks()

        #Q, K, V projections (5 keys for 5 directions)
        self.q_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.kA_linear = nn.Linear(self.embed_dim, self.embed_dim)  # Left
        self.kB_linear = nn.Linear(self.embed_dim, self.embed_dim)  # Right
        self.kC_linear = nn.Linear(self.embed_dim, self.embed_dim)  # Up
        self.kD_linear = nn.Linear(self.embed_dim, self.embed_dim)  # Down
        self.kE_linear = nn.Linear(self.embed_dim, self.embed_dim)  # Identity
        self.proj = orig_attn.proj

    def create_local_directional_masks(self):
        '''Create directional masks within a local window.  For a window_size
            x window_size window, create masks for: left, right, up, down, identity'''
        ws = self.window_size
        num_tokens = ws * ws
        
        # Create position grid for window
        positions = []
        for i in range(ws):
            for j in range(ws):
                positions.append((i, j))
        
        # Initialize masks
        left_mask = np.zeros((num_tokens, num_tokens))
        right_mask = np.zeros((num_tokens, num_tokens))
        up_mask = np.zeros((num_tokens, num_tokens))
        down_mask = np.zeros((num_tokens, num_tokens))
        identity_mask = np.eye(num_tokens)
        
        # Fill directional masks
        for i, (row_i, col_i) in enumerate(positions):
            for j, (row_j, col_j) in enumerate(positions):
                if col_j < col_i:  # j is to the left of i
                    left_mask[i, j] = 1
                if col_j > col_i:  # j is to the right of i
                    right_mask[i, j] = 1
                if row_j < row_i:  # j is above i
                    up_mask[i, j] = 1
                if row_j > row_i:  # j is below i
                    down_mask[i, j] = 1
        
        # Register as buffers (shape: [1, 1, ws*ws, ws*ws])
        self.register_buffer("left_mask", torch.tensor(left_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.register_buffer("right_mask", torch.tensor(right_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.register_buffer("up_mask", torch.tensor(up_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.register_buffer("down_mask", torch.tensor(down_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.register_buffer("identity_mask", torch.tensor(identity_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
    def forward(self, x, **kwargs):
        B, N, C = x.shape
        
        # Extract CLS token
        cls_token = x[:, :1, :]  # (B, 1, C)
        patch_tokens = x[:, 1:, :]  # (B, num_patches, C)
        
        # Reshape patches to 2D grid
        H = W = self.num_patches_per_side  # 14
        patch_tokens = patch_tokens.reshape(B, H, W, C)
        
        # Apply local windowed attention
        output_tokens = self._apply_windowed_attention(patch_tokens)
        
        # Flatten back to sequence
        output_tokens = output_tokens.reshape(B, H * W, C)
        
        # Re-attach CLS token (apply standard attention to CLS)
        cls_out = self._attend_cls(cls_token, x)
        
        # Combine
        output = torch.cat([cls_out, output_tokens], dim=1)
        
        return output
    
    def _apply_windowed_attention(self, patch_tokens):
        """
        Apply directional attention within local windows.
        patch_tokens: (B, H, W, C)
        """
        B, H, W, C = patch_tokens.shape
        ws = self.window_size
        
        # Pad if necessary to make divisible by window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            patch_tokens = torch.nn.functional.pad(patch_tokens, (0, 0, 0, pad_w, 0, pad_h))
            H_pad, W_pad = H + pad_h, W + pad_w
        else:
            H_pad, W_pad = H, W
        
        # Partition into windows
        # (B, H_pad, W_pad, C) -> (B, num_windows_h, num_windows_w, ws, ws, C)
        num_windows_h = H_pad // ws
        num_windows_w = W_pad // ws
        
        windows = patch_tokens.reshape(B, num_windows_h, ws, num_windows_w, ws, C)
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, nH, nW, ws, ws, C)
        windows = windows.reshape(B * num_windows_h * num_windows_w, ws * ws, C)  # (B*nWindows, ws*ws, C)
        
        # Apply directional attention within each window
        attended_windows = self._directional_attention(windows)  # (B*nWindows, ws*ws, C)
        
        # Reverse windowing
        attended_windows = attended_windows.reshape(B, num_windows_h, num_windows_w, ws, ws, C)
        attended_windows = attended_windows.permute(0, 1, 3, 2, 4, 5).contiguous()
        output = attended_windows.reshape(B, H_pad, W_pad, C)
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            output = output[:, :H, :W, :]
        
        return output
    
    def _directional_attention(self, windows):
        """
        Apply 5-way directional attention within windows.
        windows: (B*nWindows, ws*ws, C)
        """
        BW, N, C = windows.shape  # BW = B * num_windows
        
        # Compute Q, K, V
        q = self.q_linear(windows).reshape(BW, N, self.num_heads, self.head_dim)
        kA = self.kA_linear(windows).reshape(BW, N, self.num_heads, self.head_dim)
        kB = self.kB_linear(windows).reshape(BW, N, self.num_heads, self.head_dim)
        kC = self.kC_linear(windows).reshape(BW, N, self.num_heads, self.head_dim)
        kD = self.kD_linear(windows).reshape(BW, N, self.num_heads, self.head_dim)
        kE = self.kE_linear(windows).reshape(BW, N, self.num_heads, self.head_dim)
        v = self.v_linear(windows).reshape(BW, N, self.num_heads, self.head_dim)
        
        # Transpose for attention
        q = q.transpose(1, 2) * self.scale  # (BW, num_heads, N, head_dim)
        kA = kA.transpose(1, 2)
        kB = kB.transpose(1, 2)
        kC = kC.transpose(1, 2)
        kD = kD.transpose(1, 2)
        kE = kE.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attn_scores_ka = q @ kA.transpose(-2, -1)  # (BW, num_heads, N, N)
        attn_scores_kb = q @ kB.transpose(-2, -1)
        attn_scores_kc = q @ kC.transpose(-2, -1)
        attn_scores_kd = q @ kD.transpose(-2, -1)
        attn_scores_ke = q @ kE.transpose(-2, -1)
        
        # Apply directional masks BEFORE softmax
        def apply_mask(attn_scores, mask):
            mask_additive = (1.0 - mask) * -1e9
            return attn_scores + mask_additive
        
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
        
        # Average (not sum!)
        out = (out_a + out_b + out_c + out_d + out_e) / 5.0
        
        # Reshape
        out = out.transpose(1, 2).reshape(BW, N, self.embed_dim)
        out = self.proj(out)
        
        return out
    
    def _attend_cls(self, cls_token, full_x):
        """
        Apply standard global attention for CLS token.
        cls_token: (B, 1, C)
        full_x: (B, N+1, C) - includes CLS and all patches
        """
        B = cls_token.shape[0]
        
        # Simple approach: use identity projection for CLS
        # (In full implementation, could do proper attention)
        return cls_token
    
class LocalDirectionalViT(nn.Module):
    """
    ViT with local directional attention (windowed 5-way directional attention).
    Separate from ViTLayerReduction for easy comparison.
    """
    def __init__(self, 
                 num_blocks_to_keep,
                 num_classes,
                 pretrained,
                 drop_path_rate,
                 num_local_directional_blocks,
                 use_dwconv_bypass,
                 window_size,
                 altGlobal):
        super().__init__()
        
        # Load base model
        full_model = timm.create_model(
            "vit_small_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        self.patch_embed = full_model.patch_embed
        self.cls_token = full_model.cls_token
        self.pos_embed = full_model.pos_embed
        self.pos_drop = full_model.pos_drop
        
        # Drop path rates
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks_to_keep)]
        
        # Calculate which blocks get local directional attention
        local_dir_start = num_blocks_to_keep - num_local_directional_blocks
        
        self.blocks = nn.Sequential()
        print(f"altGlobal: {altGlobal}")
        for i, block in enumerate(full_model.blocks[:num_blocks_to_keep]):
            # Do not use alternate but just the last N blocks
            if (not altGlobal):
                # Apply local directional attention to last N blocks
                if i >= local_dir_start:
                    block.attn = LocalDirectionalAttention(
                        block.attn,
                        window_size=window_size,
                        patch_size=16,
                        img_size=224
                    )
                    print(f"Block {i}: Local directional attention (window={window_size}).")
                else:
                    print(f"Block {i}: Standard attention.")
            #apply local every other block
            else:
                if i % 2 == 0:
                    block.attn = LocalDirectionalAttention(
                        block.attn,
                        window_size=window_size,
                        patch_size=16,
                        img_size=224
                    )
                    print(f"Block {i}: Local directional attention (window={window_size}).")
                else:
                    print(f"Block {i}: Standard attention.")    
            # Drop path
            if hasattr(block, 'drop_path'):
                block.drop_path = DropPath(dpr[i]) if dpr[i] > 0. else nn.Identity()
            
            # Optional DWConv
            if use_dwconv_bypass:
                block = BlockWithDWConv(block, embed_dim=384, img_size=224, patch_size=16)
                print(f"Block {i}: DWConv bypass added.")
            
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
        return self.head(x[:, 0])
#=============
# Image-model dispatcher written by Dr. Hart  called by PyTorchLgtAttTemplate
# Given the model name, class count, and free/fine-tune preference, and it hands back
# the right neural network configured for training
# ============      
def select_image_model( 
                       model_name, 
                       n_classes, 
                       freeze_backbone, 
                       pretrained,
                       num_blocks_to_keep,
                       drop_path_rate,  #add this to default 
                       use_dwconv_bypass, # pass through for depth wise convo
                       num_local_directional_blocks, # added for local direction
                       window_size, #added for local direction
                       altGlobal # alternating global vs local
):
    if model_name == "vit_small_patch16_224":
        model = timm.create_model("vit_small_patch16_224", 
                                    pretrained=pretrained,
                                    num_blocks_to_keep=num_blocks_to_keep, 
                                    num_classes=n_classes)
        #reducing the number of blocks that will hopefully help with overfitting
        model.blocks = nn.Sequential(*model.blocks[:num_blocks_to_keep])
    elif model_name == "resnet18tv":
        model = torchvision.models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, n_classes)

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
    elif model_name == "swin_tiny_patch4_window7_224":
        model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained, num_classes=n_classes)
    elif model_name == "efficientvit_b1.r224_in1k":
        model = timm.create_model('efficientvit_b1.r224_in1k', pretrained=pretrained, num_classes=n_classes)
    elif model_name == "ViTLayerReduction":
        model = ViTLayerReduction(num_blocks_to_keep=num_blocks_to_keep, 
                                  patch_size=16,
                                  num_classes=n_classes, 
                                  img_size=224, 
                                  pretrained=pretrained,
                                  drop_path_rate=drop_path_rate, # coming through for stochastic depth
                                  use_dwconv_bypass=use_dwconv_bypass #pass through
                                  )
    elif model_name == "DWConv_vit_small":
        # using depthwise convolution without custom vit
        model = StandardViTWithDWConv(
                num_blocks_to_keep=num_blocks_to_keep #adding this for overfitting
            )
    elif model_name == "LocalDirectionalViT":
        model = LocalDirectionalViT(
            num_blocks_to_keep=num_blocks_to_keep,
            num_classes=n_classes,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            num_local_directional_blocks=num_local_directional_blocks,
            use_dwconv_bypass=use_dwconv_bypass,
            window_size=window_size,
            altGlobal=altGlobal
        )
    else:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=n_classes)

    return model