import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import Dataset, DataLoader
import torchvision
import timm
import pandas as pd
import math

from models import select_image_model

#==========
# Lightning Module - Wraps model for training
#===========
class LitNetwork(pl.LightningModule):
    def __init__(self, 
                 model_name="ViTLayerReduction", 
                 freeze_backbone=False, 
                 pretrained=True,
                 lr=1e-4,
                 base_lr=1e-6,
                 peak_lr=1e-4,
                 weight_decay=0.01,
                 num_epochs=120,
                 warmup_epochs=3,
                 rampup_epochs=5,
                 final_lr_fraction=0.1,
                 num_blocks=12,
                 scheduler_type="warmup_cosine"  #New: track scheduler type
                 ):
        super().__init__()
        self.save_hyperparameters()

        h = self.hparams
        n_classes = 101

        self.model = select_image_model(
            model_name=model_name, 
            n_classes=n_classes, 
            freeze_backbone=freeze_backbone,
            pretrained=pretrained,
            num_blocks=h.num_blocks
        )

        self.loss_func = torch.nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy("multiclass", num_classes=n_classes, average='micro')
        self.val_acc = torchmetrics.Accuracy("multiclass", num_classes=n_classes, average='micro')
        self.test_acc = torchmetrics.Accuracy("multiclass", num_classes=n_classes, average='micro')

    def on_fit_start(self):
        # Log hyperparameters with metrics placeholders for HPARAMS tab
        self.logger.log_hyperparams(
                            self.hparams,
                            {
                                "hp/train_acc": 0,
                                "hp/val_acc": 0,
                                "hp/test_acc": 0,
                            })
        writer = self.logger.experiment
        dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
        writer.add_graph(self, dummy_input)

        # Add detailed configuration as text
        config_text = f"""
        # Model Configuration
        **Model:** {self.hparams.model_name}  
        **Pretrained:** {self.hparams.pretrained}  
        **Num Blocks:** {self.hparams.num_blocks}  
        **Freeze Backbone:** {self.hparams.freeze_backbone}

        # Training Configuration
        **Peak LR:** {self.hparams.peak_lr}  
        **Base LR:** {self.hparams.base_lr}  
        **Weight Decay:** {self.hparams.weight_decay}  
        **Scheduler:** {self.hparams.scheduler_type}  
        **Warmup Epochs:** {self.hparams.warmup_epochs}  
        **Rampup Epochs:** {self.hparams.rampup_epochs}  
        **Total Epochs:** {self.hparams.num_epochs}

        # Regularization
        **Dropout Rate:** 0.3  
        **Drop Path Rate:** 0.1  
        **Mixed Precision:** True
        """
        writer.add_text("Configuration", config_text,0)
        
    def forward(self, x):
        x = self.model(x)
        return x
        
    def training_step(self, batch, batch_idx):
        images, labels = batch
        
        if batch_idx == 0:
            print("ACTUAL_LR: ", self.optimizers().param_groups[0]["lr"])

        # Forward pass
        logits = self(images)
        loss = self.loss_func(logits, labels)

        # Batch accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc_epoch", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        # Log learning rate
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log("lr", lr, prog_bar=True, on_step=True, on_epoch=False)
        
        return loss
    
    def validation_step(self, val_data, batch_idx):
        im, label = val_data[0], val_data[1]
        outs = self.forward(im)

        val_loss = self.loss_func(outs, label)
        
        self.val_acc(outs, label)
        self.log("val_acc_epoch", self.val_acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        return val_loss

    def test_step(self, test_data, batch_idx):
        im, label = test_data[0], test_data[1]
        outs = self.forward(im)
        self.test_acc(outs, label)
        self.log("test_acc", self.test_acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return None
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.peak_lr, 
            weight_decay=self.hparams.weight_decay
        )

        assert optimizer.param_groups[0]["lr"] == self.hparams.peak_lr

        def lr_lambda(step):
            h = self.hparams
            steps_per_epoch = self.trainer.estimated_stepping_batches / h.num_epochs
            epoch_step = step / steps_per_epoch

            if epoch_step < h.warmup_epochs:
                return h.base_lr / h.peak_lr
            elif epoch_step < h.warmup_epochs + h.rampup_epochs:
                progress = (epoch_step - h.warmup_epochs) / h.rampup_epochs
                lr = h.base_lr + progress * (h.peak_lr - h.base_lr)
                return lr / h.peak_lr
            else:
                decay_progress = (epoch_step - h.warmup_epochs - h.rampup_epochs) / max(1, h.num_epochs - h.warmup_epochs - h.rampup_epochs)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
                lr = h.final_lr_fraction * h.peak_lr + (1 - h.final_lr_fraction) * h.peak_lr * cosine_decay
                return lr / h.peak_lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

#==========
# Main training script (for standalone testing)
#==========
if __name__ == "__main__":

    dataset_dir = "data/"
    model_names = [
        "ViTLayerReduction", 
        "vit_small_patch16_224", 
        "resnet18tv", 
        "resnet18timm",
        "resnet50tv",
        "efficientnet_b0"
    ]
    model_name = model_names[0]  # Change index to test different models
    pretrained = True
    b = 64
    width = 224
    height = 224
    
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop((height, width)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = torchvision.datasets.Food101(
        root=dataset_dir, 
        split='train',
        transform=transforms,
        download=True
    )
    
    test_dataset = torchvision.datasets.Food101(
        root=dataset_dir,
        split="test",
        transform=transforms,
        download=True 
    )   
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, 
        [int(len(train_dataset)*0.8), len(train_dataset) - int(len(train_dataset)*0.8)]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=b, shuffle=True, num_workers=12, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=b, shuffle=False, num_workers=12, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=b)

    model = LitNetwork(
        model_name=model_name,
        pretrained=True,
        freeze_backbone=False,
        lr=1e-4,
        peak_lr=1e-4,
        weight_decay=0.01,
        warmup_epochs=3,
        rampup_epochs=5,
        num_epochs=120
    )
    
    checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_acc_epoch', save_top_k=1, mode='max')
    logger = pl_loggers.TensorBoardLogger(save_dir="compare_more_models", name=model_name)

    device = "gpu"
    torch.set_float32_matmul_precision('medium')
    
    trainer = pl.Trainer(
        max_epochs=120, 
        accelerator=device, 
        callbacks=[checkpoint], 
        logger=logger,
        precision="16-mixed"  # Mixed precision for efficiency
    )
    
    trainer.fit(model, train_loader, val_loader)
    trainer.test(ckpt_path="best", dataloaders=test_loader)