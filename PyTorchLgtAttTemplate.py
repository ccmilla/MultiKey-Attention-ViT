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
# Lightning powered training wrapper
# Takes a plain PyTorch model and wraps it in all the orchestration a training loop
# normally needs: forward, pass, loss, optimizer, metrics, and logging.
# Converts model into a fully trainable PyTorch Lightning system.
#===========
class LitNetwork(pl.LightningModule):
    #Residual Network Deep Residual Learning for Image Recognition with 18 layers.
    #Convultional, pooling, fully connected, and skip connection layers.
    def __init__(self, 
                 model_name="ViTLayerReduction", 
                 freeze_backbone=False, 
                 pretrained=True,
                 lr=1e-4,
                 base_lr=1e-6,
                 peak_lr =1e-4,
                 weight_decay=0.01, #changing from .5 to 0.01
                 num_epochs=120,
                 warmup_epochs=3,  # this depends on whether or not it is pretrained shall we do 3 for pretrained?
                 rampup_epochs=5,   # recommended 5?
                 final_lr_fraction=0.1):
        super().__init__()
        #logging the hyperparameters on each run.
        self.save_hyperparameters()

        # scale warmup/rampup with num_epochs
        h = self.hparams
        #passing in warmup and rampup epochs
        # h.warmup_epochs = max(1, int(0.1 * h.num_epochs))   # 10% of total epochs
        # h.rampup_epochs = max(1, int(0.1 * h.num_epochs))   # 10% of total epochs
        
        n_classes = 101 #Change num_classes to the number of classification categories in dataset

        self.model = select_image_model(model_name=model_name, 
                                        n_classes=n_classes, 
                                        freeze_backbone=freeze_backbone,
                                        pretrained=pretrained)

        self.loss_func = torch.nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy("multiclass",num_classes=n_classes,average='micro')
        self.val_acc = torchmetrics.Accuracy("multiclass",num_classes=n_classes,average='micro')
        self.test_acc = torchmetrics.Accuracy("multiclass",num_classes=n_classes,average='micro')

    def on_fit_start(self):
        self.logger.log_hyperparams(self.hparams)
        writer = self.logger.experiment
        dummy_input = torch.randn(1,3,224,224, device=self.device)
        writer.add_graph(self, dummy_input)
    def forward(self, x):
        x = self.model(x)
        return x
    def training_step(self, batch, batch_idx):
        images, labels = batch
        
        if batch_idx == 0:
            print("ACTUAL_LR: ", self.optimizers().param_groups[0]["lr"])

        #forward pass
        logits = self(images)
        loss = self.loss_func(logits, labels)

        #batch accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds== labels).float().mean()
        
        #apparently, train_loss needs to be logged - lightning expects a training loss to anchor the training loop.
        self.log("train_loss",loss,on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.log(
                 "train_acc_epoch",
                 acc,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True
                )
        #self.log("train_acc_step", acc, prog_bar=True, on_step=True, on_epoch=False)
        
        # Log LR
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log("lr", lr, prog_bar=True, on_step=True, on_epoch=False)
        return loss
    
    def validation_step(self, val_data, batch_idx):
        im, label = val_data[0], val_data[1]
        outs = self.forward(im)

        #validation loss
        val_loss = self.loss_func(outs, label)
        '''
        self.log(
            "val_loss",
            val_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )
        '''
        #validation accuracy
        self.val_acc(outs,label)
        self.log(
            "val_acc_epoch",
            self.val_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True)
        # self.log(
        #     "val_acc_step",
        #     self.val_acc,
        #     prog_bar=True,
        #     on_step=True,
        #     on_epoch=False,
        #     sync_dist=True)
        return val_loss

    def test_step(self, test_data, batch_idx):
        im, label = test_data[0], test_data[1]
        outs = self.forward(im)
        self.test_acc(outs,label)
        self.log("test_acc",self.test_acc,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        return None
    
    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), 
    #                                   lr=self.hparams.peak_lr, 
    #                                   weight_decay=self.hparams.weight_decay
    #     )

    #     #adding this to make sure that lr is the same as peak_lr to increase training and validation accuracy
    #     assert optimizer.param_groups[0]["lr"] == self.hparams.peak_lr

    #     def lr_lambda(step):
    #         h = self.hparams # moved all the variables into init
    #         #steps_per_epoch = self.trainer.estimated_stepping_batches / self.hparams.num_epochs
    #         steps_per_epoch = self.trainer.estimated_stepping_batches / h.num_epochs
    #         epoch_step = step / steps_per_epoch

    #         if epoch_step < h.warmup_epochs:
    #             return h.base_lr / h.peak_lr
    #         elif epoch_step < h.warmup_epochs + h.rampup_epochs:
    #             progress = (epoch_step - h.warmup_epochs) / h.rampup_epochs
    #             lr = h.base_lr + progress * (h.peak_lr - h.base_lr)
    #             return lr / h.peak_lr
    #         else:
    #             decay_progress = (epoch_step - h.warmup_epochs - h.rampup_epochs) / max(1, h.num_epochs - h.warmup_epochs - h.rampup_epochs)
    #             cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
    #             lr = h.final_lr_fraction * h.peak_lr + (1 - h.final_lr_fraction) * h.peak_lr * cosine_decay
    #             return lr / h.peak_lr

    #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    #     return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
    ''' New configure_optimizers function.  Pretrained is true so we won't be using lr_lambda'''
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.peak_lr,
            weight_decay=self.hparams.weight_decay
        )
        #Simple cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.num_epochs,
            eta_min=1e-6 # min learning rate at end
        )
        return {
            "optimizer": optimizer,
            "lr_schedule": {
                "scheduler": scheduler,
                "interval": "epoch"    # update once per epoch, not per step
            }
        }

    # def setup(self, stage=None):
    #     #Guardrail: optimizer LR must equal peak LR for LambdaLR logic
    #     assert self.hparams.lr == self.hparams.peak_lr, (
    #         f"Expect lr ({self.hparams.lr}) to equal peak_lr"
    #         f"({self.hparams.peak_lr}) for LambdaLR scaling"
    #     )

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
    #changing this for an increase in accuracy
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
                          download=True)
    test_dataset = torchvision.datasets.Food101(
                          root=dataset_dir,
                          split="test",
                          transform=transforms,
                          download=True 
    )

    #train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(len(train_dataset)*0.8), len(train_dataset) - int(len(train_dataset)*0.8)])
    #train_dataset= torch.utils.data.random_split(train_dataset, [int(len(train_dataset)*0.8), len(train_dataset) - int(len(train_dataset)*0.8)])

    #test_dataset = torchvision.datasets.Food101(root=dataset_dir, split='test', transform=transforms, download=True)
    # Split test into val + test (80/20 split of the test set)

    val_dataset, test_dataset = torch.utils.data.random_split(
                                                    test_dataset, 
                                                    [int(len(test_dataset)*0.8), len(test_dataset) - int(len(test_dataset)*0.8)]
                                                    )

    train_loader = DataLoader(train_dataset,batch_size=b,shuffle=True, num_workers=12, persistent_workers=True)
    val_loader = DataLoader(val_dataset,batch_size=b,shuffle=False, num_workers=12, persistent_workers=True)
    test_loader = DataLoader(test_dataset,batch_size=b)

    #added lr, increased num_epochs to 120 and made sure freeze_backbone is false and pretrain is true.
    model = LitNetwork(
                model_name=model_name,
                pretrained=True,
                freeze_backbone=False,
                lr=1e-4, #toggling between 2e-4 and 1e-4
                peak_lr=1e-4, # this a good number?
                weight_decay=0.01, #added (was using 0.5 which is too high which was killing learning)
                warmup_epochs=3, # passing this in shorter for pretrained
                rampup_epochs=5, #also passing in smaller rampup
                num_epochs=120
                )
    checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_acc_epoch', save_top_k=1, mode='max')
    logger = pl_loggers.TensorBoardLogger(save_dir="logs_VitLayer",name=model_name)
    #logger = pl_loggers.CSVLogger(save_dir="my_logs",name="my_csv_logs")

    #device = "gpu" # Use 'mps' for Mac M1 or M2 Core, 'gpu' for Windows with Nvidia GPU, or 'cpu' for Windows without Nvidia GPU
    device = "gpu"
    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(max_epochs=120, accelerator=device, callbacks=[checkpoint], logger=logger)
    trainer.fit(model,train_loader,val_loader)
    
    trainer.test(ckpt_path="best", dataloaders=test_loader)