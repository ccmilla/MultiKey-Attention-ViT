import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Food101
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from timm import create_model

#================
# Defines image transformations for training and validation.  Training data is augmented with cropping, 
# flipping, and color jittering, while validation data is resized and center cropped.  Both are converted
# to tensors and normalized using ImageNet statistics.
# train_transform, val_transform, and transforms.Normalize putting it into ImageDataModule.setup()
# creates datasets using transforms train_dataloader().  Auto generated in init so it's cleaner.
# Organize Datasets into a lightningDataModule in PyTorch Ligntning

from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from torchvision import transforms

#==========
# DataModule     I know Dr. Hart already created the PyTorchLgtAttTemplate, but I want to go back to my previous Tensorboard
#===========

class Food101DataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size= 32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        self.train_dataset = Food101(root=self.data_dir, split='train', transform=self.train_transform, download=True)
        self.val_dataset = Food101(root=self.data_dir, split='test', transform=self.val_transform, download=True)
    
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

''' create_model instantiates a Vision Transformer (small variant) from timm.num_classes = 101 sets the final
 linear head to output 101 logits(Food101).  Now I'm still usng this VitLayerReduction model but putting it
  into lightning.  This time, weight_decay is 0.5 because of overfitting from the previous code.  Also, changing
   the drop rate to 0.5 and drop_path_rate to 0.2 (regularization hyperparameters) - dropout and stochastic depth '''

from timm import create_model

'''By default PyTorch uses full float32 percision for matrix multiplications.  Tensor Cores on modern
GPUs can accelerate mixed-precision math (float16) or reduced precision float32 computations.  
Trade a tiny bit of numerical percision for faster training.'''

torch.set_float32_matmul_precision('medium')

#=========
#ViT model
#=========

class VitLayerReduction(nn.Module):
    def __init__(self, num_blocks=10, num_classes=101):
        super().__init__()
        full_model = create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=num_classes,
            drop_rate=0.5,
            drop_path_rate=0.2
        )
        self.patch_embed = full_model.patch_embed
        self.cls_token = full_model.cls_token
        self.pos_embed = full_model.pos_embed
        self.pos_drop = full_model.pos_drop
        self.blocks = nn.Sequential(*list(full_model.blocks[:num_blocks]))
        self.norm = full_model.norm
        self.head = full_model.head

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens =self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:,0])
    
'''
Shape example (for 224x2224, patch_size=16): Image x: (B,3,224,224) After patch embed: (B, 196, D) - 14x14 = 196 patches
full_model.blocks is a list of transformer encoder blocks(attention + MLP).  This keeps only the first 10 blocks (layer reduction)
Wrapping them in nn.Sequential allows you to call them as one module.
Why?  Fewer blocks mean fewer parameters & FLOPS, and possibly less overfitting.
norm is the final layer norm applied to token embeddings.  head is the classification head (a linear mapping D -> num_classes)
The class token is at index 0 after concatenation; x[:0] extracts it per batch entry head returns raw logits(not softmax).
Loss functions like CrossEntropy expects logits
'''
class LitViT(pl.LightningModule):
    def __init__(self, num_classes=101, base_lr=1e-5, peak_lr=1e-4, final_lr_fraction=0.1,
                 num_epochs=60, warmup_epochs=15, rampup_epochs=15, weight_decay=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.model = VitLayerReduction(num_blocks=10, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.epoch_start_time = None

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()
    
    def on_train_epoch_end(self):
        duration = time.time() - self.epoch_start_time
        train_loss = self.trainer.callback_metrics.get("train_loss")
        train_acc = self.trainer.callback_metrics.get("train_acc")
        val_loss = self.trainer.callback_metrics.get("val_loss")
        val_acc = self.trainer.callback_metrics.get("val_acc")
        #Epoch number start at 1
        epoch_num = self.current_epoch + 1
        print(f"Epoch {epoch_num} completed in {duration: .2f} seconds\n")
        print(f"Train Loss: {train_loss: .4f}  Train acc: {train_acc: .4f}\n")
        print(f"Val Loss: {val_loss: .4f}  Val acc: {val_acc: .4f} \n")

    def forward(self,x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) ==y).float().mean()

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)

        #Log LR
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]['lr']
        self.log('lr', current_lr, prog_bar=True, on_epoch=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x,y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) ==y).float().mean()

        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
                                      self.parameters(), 
                                      lr = self.hparams.peak_lr, 
                                      weight_decay=self.hparams.weight_decay
                                      )

        def lr_lambda(step):
            steps_per_epoch = self.trainer.estimated_stepping_batches / self.hparams.num_epochs
            epoch_step = step / steps_per_epoch

            # the following code keeps the LR constant during warmup instead of increasing it.
            # if epoch_step < self.hparams.warmup_epochs:
            #     return self.hparams.base_lr / self.hparams.peak_lr
            # elif epoch_step < self.hparams.warmup_epochs + self.hparams.rampup_epochs:
            #     progress = (epoch_step - self.hparams.warmup_epochs)/self.hparams.rampup_epochs
            #     lr = self.hparams.base_lr + progress * (self.hparams.peak_lr - self.hparams.base_lr)
            #     return lr / self.hparams.peak_lr

            # === Warmup ====
            if epoch_step < self.hparams.warmup_epochs:
                progress = epoch_step / self.hparams.warmup_epochs
                lr = self.hparams.base_lr + progress * (self.hparams.peak_lr - self.hparams.base_lr)
            # == Ramp == (hold peak)
            elif epoch_step < self.hparams.warmup_epochs + self.hparams.rampup_epochs:
                lr = self.hparams.peak_lr
            # == Cosine decay ==
            else:
                decay_epochs = (
                    self.hparams.num_epochs
                    -self.hparams.warmup_epochs
                    -self.hparams.rampup_epochs
                )
                decay_progress = (epoch_step 
                                  - self.hparams.warmup_epochs 
                                  - self.hparams.rampup_epochs
                                  ) / max(1, decay_epochs)
                # if training runs longer than planned, decay_progress can exceed 1.
                decay_progress = min(decay_progress, 1.0)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
                lr = self.hparams.final_lr_fraction * self.hparams.peak_lr + (1 - self.hparams.final_lr_fraction) * self.hparams.peak_lr * cosine_decay

            return lr / self.hparams.peak_lr
            
        scheduler = torch.optim.lr_scheduler.LambdaLR(
                                                        optimizer, 
                                                        lr_lambda=lr_lambda
                                                    )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step" 
                }
            }
    
''' ^^ Lightning Boilerplate ^^ 
     Lightning wraps training boilerplate so you write only the core logic.  LitViT wraps the model and stores configuration
     (leaning rate).  It still calls forward() for inference.  Keep it simple -- delegate to the underlying module.

     training_step is called for each training batch.  Batch is typically (inputs, labels) from DataLoader.
     Compute logits -> compute cross_entropy loss (combines log_softmax + nll_loss).self.log records metrics;
     prog_bar=True shows it in the progress bar.  Returning loss tells lightning to run backprop with it.

     The validation step is similar ot the training step, but for validation batches.  NO optimizer step.

     Training the model now... Training orchestrates the whole run: device placement, loop, checkingpoint( if configured),
     logging, etc.
     accelerlator="gpu", devices=1 runs on 1 GPU (if available).  Use "cpu" or remove parameters if you don't have a GPU
     trainer.fit(model, data_module) starts training.data_module supplies train_dataloader() and val_dataloader()
    
'''

#================
# Training Script
#================

if __name__ == "__main__":
    batch_size = 32
    max_epochs = 60

    data_module = Food101DataModule(batch_size=batch_size)
    model = LitViT(num_classes=101, num_epochs=max_epochs)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="best-vit-{epoch:02d}-{val_acc:.4f}"
    )

    logger = pl_loggers.TensorBoardLogger(save_dir="mine",name="vit_small_patch16_224")
    #device = "gpu" # Use 'mps' for Mac M1 or M2 Core, 'gpu' for Windows with Nvidia GPU, or 'cpu' for Windows without Nvidia GPU
    device = "gpu"
    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        precision=16,
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=50
    )
    # changing the parameters
    trainer.fit(model, datamodule=data_module)

    print(f"Best model saved at: {checkpoint_callback.best_model_path}")