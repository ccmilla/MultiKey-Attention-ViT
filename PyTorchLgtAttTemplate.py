import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader
import torchvision
#import timm
import pandas as pd
import math
from pytorch_lightning.callbacks import LearningRateMonitor
from models import select_image_model

#==========
# Lightning powered training wrapper
# Takes a plain PyTorch model and wraps it in all the orchestration a training loop
# normally needs: forward, pass, loss, optimizer, metrics, and logging.
# Converts model into a fully trainable PyTorch Lightning system.
#===========
class LitNetwork(pl.LightningModule):
    #Residual Network Deep Residual Learning for Image Recognition with 18 layers.
    #Convultional, pooling, fully connected, and skip connection layers.
    def __init__(self, model_name, 
                       freeze_backbone=False, 
                       pretrained=True, #hopefully this will increase my accuracy
                       lr=1e-4,
                       base_lr=1e-6,
                       peak_lr =1e-4,
                       weight_decay=1e-2, # from 0.5
                       num_epochs=60,
                       warmup_epochs=15,
                       rampup_epochs=15,
                       final_lr_fraction=0.1
                ):
        super(LitNetwork, self).__init__()
        #logging the hyperparameters on each run.
        self.save_hyperparameters()

        # scale warmup/rampup with num_epochs
        h = self.hparams
        h.warmup_epochs = max(1, int(0.1 * h.num_epochs))   # 10% of total epochs
        h.rampup_epochs = max(1, int(0.1 * h.num_epochs))   # 10% of total epochs

        n_classes = 101 #Change num_classes to the number of classification categories in your dataset

        self.model = select_image_model(model_name=model_name, n_classes=n_classes, freeze_backbone=freeze_backbone, pretrained=pretrained)

        self.loss_func = torch.nn.CrossEntropyLoss()
        self.val_acc = torchmetrics.Accuracy("multiclass",num_classes=n_classes,average='micro')
        self.test_acc = torchmetrics.Accuracy("multiclass",num_classes=n_classes,average='micro')

    def on_fit_start(self):
        writer = self.logger.experiment
        dummy_input = torch.randn(1,3,224,224, device=self.device)
        writer.add_graph(self, dummy_input)
    def forward(self, x):
        x = self.model(x)
        return x
    def training_step(self, data, batch_idx):
        # im, label = data[0], data[1] doing forward pass twice
        im, label = data
        # x,y = data
        #logits = self(x)
        logits = self(im)
        #outs = self.forward(im) this was the second time.
        # loss = self.loss_func(outs, label)
        loss = self.loss_func(logits, label)
        # acc = (logits.argmax(dim=1) == y).float().mean()
        acc = (logits.argmax(dim=1) == label).float().mean()

        self.log("train_loss",loss,sync_dist=True,on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        
        # Log LR
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]['lr']
        self.log("lr", current_lr, prog_bar=True, on_step=True, on_epoch=False)
        return loss
    
    def validation_step(self, val_data, batch_idx):
        im, label = val_data[0], val_data[1]
        outs = self.forward(im)

        #validation loss
        val_loss = self.loss_func(outs, label)
        self.log(
            "val_loss",
            val_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )
        #validation accuracy
        self.val_acc(outs,label)
        self.log(
            "val_acc",
            self.val_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True)
        return val_loss

    def test_step(self, test_data, batch_idx):
        im, label = test_data[0], test_data[1]
        outs = self.forward(im)
        self.test_acc(outs,label)
        self.log("test_acc",self.test_acc,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        return None
    
    def configure_optimizers(self):
        # just making this simpler
        h = self.hparams
        #optimizer = torch.optim.AdamW(self.parameters(), lr=h.peak_lr, weight_decay=h.weight_decay)
        #let's remove lr_lambda since that's giving me so much trouble
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-2)
        # going back to steps
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.1 * total_steps)
        rampup_steps = int(0.1 * total_steps)
        decay_steps = total_steps - warmup_steps - rampup_steps
        def lr_lambda(step):
            #steps_per_epoch = self.trainer.estimated_stepping_batches / self.hparams.num_epochs
            # steps_per_epoch = self.trainer.estimated_stepping_batches / h.num_epochs
            # epoch_step = step / steps_per_epoch

            if step < warmup_steps:
                # return h.base_lr / h.peak_lr this is flat
                # lr = h.base_lr + (epoch_step / h.warmup_epochs) * (h.peak_lr - h.base_lr)
                # return lr / h.peak_lr
                #return (h.base_lr + (step/warmup_steps) * h.peak_lr - h.base_lr) /h.peak_lr
                #latest AI suggestion
                # progress = step / max(1, warmup_steps)
                # lr = h.base_lr + progress * (h.peak_lr - h.base_lr)
                # return lr / h.peak_lr
                # now AI is telling me that is where my random guessing is happening.  So here goes:
                return h.base_lr + (step / warmup_steps)*(h.peak_lr - h.base_lr)/h.peak_lr

            elif step < warmup_steps +rampup_steps:
                # progress = (epoch_step - h.warmup_epochs) / h.rampup_epochs
                # lr = h.base_lr + progress * (h.peak_lr - h.base_lr)
                # return lr / h.peak_lr
                return 1.0 # stay at peak
            else:
                #decay_progress = (epoch_step - h.warmup_epochs - h.rampup_epochs) / max(1, h.num_epochs - h.warmup_epochs - h.rampup_epochs)
                decay_step = step - warmup_steps - rampup_steps
                progress = decay_step/max(1,decay_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                lr = h.final_lr_fraction * h.peak_lr + (1 - h.final_lr_fraction) * h.peak_lr * cosine_decay
                return lr / h.peak_lr
        # not even using this lr_scheduler
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        # now I will just return the optimizer
        return optimizer

#==========
# main training script
# Launches an entire training pipeline for an image classifer on Food-101
#==========
if __name__ == "__main__":

    # Uncomment to see list of timm models
    import timm
    # model_names = timm.list_models(pretrained=True)
    # model_names = timm.list_models("*vit*")  
    # print(model_names)

    dataset_dir = "data/"
    model_names = ["ViTLayerReduction", "vit_small_patch16_224", "resnet18tv", "resnet18timm"]  # Add more model names as needed
    model_name = model_names[0]
    #pretrained = False
    pretrained = True # Hopefully this will increase my accuracy
    b = 64
    width = 224
    height = 224
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop((height, width)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = torchvision.datasets.Food101(root=dataset_dir, split='train', transform=transforms, download=True)

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(len(train_dataset)*0.8), len(train_dataset) - int(len(train_dataset)*0.8)])

    test_dataset = torchvision.datasets.Food101(root=dataset_dir, split='test', transform=transforms, download=True)

    train_loader = DataLoader(train_dataset,batch_size=b,shuffle=True, num_workers=12, persistent_workers=True)
    val_loader = DataLoader(val_dataset,batch_size=b,shuffle=False, num_workers=12, persistent_workers=True)
    test_loader = DataLoader(test_dataset,batch_size=b)

    model = LitNetwork(model_name=model_name, pretrained=pretrained)
    checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_acc', save_top_k=1, mode='max')
    logger = pl_loggers.TensorBoardLogger(save_dir="jan_logs",name=model_name)
    #logger = pl_loggers.CSVLogger(save_dir="my_logs",name="my_csv_logs")

    #device = "gpu" # Use 'mps' for Mac M1 or M2 Core, 'gpu' for Windows with Nvidia GPU, or 'cpu' for Windows without Nvidia GPU
    device = "gpu"
    torch.set_float32_matmul_precision('medium')
    #adding Learning Rate Monitor to trainer
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(max_epochs=60, accelerator=device, callbacks=[checkpoint, lr_monitor], logger=logger)#need to match max to num_epochs
    trainer.fit(model,train_loader,val_loader)
    
    trainer.test(ckpt_path="best", dataloaders=test_loader)