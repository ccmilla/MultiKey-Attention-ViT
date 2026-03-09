import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader
import torchvision
import math
import time

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
                 model_name, 
                 freeze_backbone, 
                 pretrained,
                 lr,
                 peak_lr,
                 weight_decay,
                 warmup_epochs,  
                 rampup_epochs,
                 num_epochs,
                 final_lr_fraction,   
                 num_blocks_to_keep,
                 drop_path_rate,
                 use_dwconv_bypass, #adding dephthwise convolution
                 num_local_directional_blocks, # adding this for local directional attention
                 window_size, #adding this for local directional attention
                 altGlobal  #adding this for local directional attention 
                 ):
        super().__init__()
        self.lr = lr
        self.peak_lr = peak_lr
        self.warmup_epochs = warmup_epochs
        self.rampup_epochs = rampup_epochs
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.final_lr_fraction = final_lr_fraction
        self.num_blocks_to_keep = num_blocks_to_keep
        self.drop_path_rate = drop_path_rate
        self.use_dwconv_bypass = use_dwconv_bypass
        self.num_local_directional_blocks = num_local_directional_blocks
        self.window_size = window_size
        self.altGlobal = altGlobal
        
        #logging the hyperparameters on each run.
        self.save_hyperparameters()
        n_classes = 101 #Change num_classes to the number of classification categories in dataset
        self.model = select_image_model(model_name=model_name, 
                                        n_classes=n_classes, 
                                        freeze_backbone=freeze_backbone,
                                        pretrained=pretrained,
                                        num_blocks_to_keep=num_blocks_to_keep,
                                        drop_path_rate=drop_path_rate,
                                        use_dwconv_bypass=use_dwconv_bypass, #pass through wrapper dwconv
                                        window_size=window_size, # pass through for localDirectionalViT
                                        num_local_directional_blocks=num_local_directional_blocks, # pass through for localDirectionalViT
                                        altGlobal=altGlobal # pass through for localDirectionalViT
                                        )

        self.loss_func = torch.nn.CrossEntropyLoss() 
        self.train_acc = torchmetrics.Accuracy("multiclass",num_classes=n_classes,average='micro')
        self.val_acc = torchmetrics.Accuracy("multiclass",num_classes=n_classes,average='micro')
        self.test_acc = torchmetrics.Accuracy("multiclass",num_classes=n_classes,average='micro')

    def on_fit_start(self):
        self.logger.log_hyperparams(self.hparams)
        # writer = self.logger.experiment
        # dummy_input = torch.randn(1,3,224,224, device=self.device)
        # writer.add_graph(self, dummy_input)
    def forward(self, x):
        x = self.model(x)
        return x
    def training_step(self, batch, batch_idx):
        images, labels = batch
        
        # if batch_idx == 0:
        #     print("ACTUAL_LR: ", self.optimizers().param_groups[0]["lr"])

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
        
        # Log LR
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log("lr", lr, prog_bar=True, on_step=True, on_epoch=False)
        return loss
    
    def validation_step(self, val_data):
        im, label = val_data[0], val_data[1]
        outs = self.forward(im)

        #validation loss
        val_loss = self.loss_func(outs, label)
        #validation accuracy
        self.val_acc(outs,label)
        self.log(
            "val_acc_epoch",
            self.val_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True)
        #adding validation loss
        self.log("train_loss",val_loss,on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return val_loss

    def test_step(self, test_data):
        im, label = test_data[0], test_data[1]
        outs = self.forward(im)
        self.test_acc(outs,label)
        self.log("test_acc",self.test_acc,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        return None
    
    def on_train_epoch_start(self):
        #return super().on_train_epoch_start() #keeping track of time per epoch to see if each epoch is computationaly expensive.
        self.epoch_start_time = time.time()
    
    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        self.log("epoch_time_minutes", epoch_time/60.0, on_epoch=True, prog_bar=True)
        #return super().on_train_epoch_end()
    
    def configure_optimizers(self):
         # shortcut
        h = self.hparams
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=h.lr, #claude said to use peak_lr
                                      weight_decay=h.weight_decay
        )

        # Choose scheduler based on model architecture
        model_name = h.model_name

        #adding this to make sure lr has correctly set the initial lr
        #assert optimizer.param_groups[0]["lr"] == h.peak_lr # claude suggested peak_lr

        #custom learning rate for custom ViT as well as vit_small_patch16_224
        def lr_lambda(step):
            h = self.hparams # moved all the variables into init
            #steps_per_epoch = self.trainer.estimated_stepping_batches / self.hparams.num_epochs
            steps_per_epoch = self.trainer.estimated_stepping_batches / h.num_epochs
            epoch_step = step / steps_per_epoch

            if epoch_step < h.warmup_epochs:
                return h.lr / h.peak_lr
            elif epoch_step < h.warmup_epochs + h.rampup_epochs:
                progress = (epoch_step - h.warmup_epochs) / h.rampup_epochs
                lr = h.lr + progress * (h.peak_lr - h.lr)
                return lr / h.peak_lr
            else:
                decay_progress = (epoch_step - h.warmup_epochs - h.rampup_epochs) / max(1, h.num_epochs - h.warmup_epochs - h.rampup_epochs)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
                lr = h.final_lr_fraction * h.peak_lr + (1 - h.final_lr_fraction) * h.peak_lr * cosine_decay
                return lr / h.peak_lr
        if "vit_small" in model_name.lower() \
            or "ViT" in model_name \
            or "DWConv" in model_name \
            or "LocalDirectional" in model_name:
            # Vision Transformers: Use custom warmup + cosine
            print(f"Using LambdaLR with warmup for model {model_name}")
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            
            #Calculate steps per epoch properly
            # steps_per_epoch = self.trainer.estimated_stepping_batches // h.num_epochs

            # scheduler = torch.optim.lr_scheduler.OneCycleLR(
            #                             optimizer,
            #                             max_lr=h.peak_lr,   #peak learning rate
            #                             total_steps=self.trainer.estimated_stepping_batches,
            #                             epochs=h.num_epochs,    #training epochs
            #                             steps_per_epoch=steps_per_epoch,
            #                             pct_start= h.warmup_epochs/h.num_epochs,    #percentage of training warmup
            #                             anneal_strategy='cos', #cosine annealing
            #                             div_factor=h.peak_lr/h.lr,   #initial_lr = max_lr/div_factor (so ie-3/25 = 4e-5)
            #                             final_div_factor=h.peak_lr/ (h.final_lr_fraction * h.peak_lr)
            #                             )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                }
            }
        elif "resnet" in model_name.lower() or "swin" in model_name.lower():
            #CNNs: Simple consine annealing (no warmup needed)
            print(f"Using CosineAnnealingLr for {model_name}")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.num_epochs, eta_min=1e-6)
            return{
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        elif "efficientvit" in model_name.lower():
            print(f"Using CosineAnnealingLR for {model_name}")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=h.num_epochs, 
                eta_min=1e-6
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
                }
            }
        else:
            #Default: Reduce on plateau (safe choice)
            print(f"Using ReduceLROnPlateau for {model_name}")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
            return {"optimizer": optimizer, 
                    "lr_scheduler": {
                        "scheduler": scheduler, 
                        "interval": "epoch"}}

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
    model_names = ["ViTLayerReduction", "vit_small_patch16_224", "resnet18tv", 
                   "swin_tiny_patch4_window7_224", "efficientvit_b1.r224_in1k",
                   "DWConv_vit_small", "LocalDirectionalViT"]  # Add more model names as needed
    model_name = model_names[5]
    b = 64
    width = 224
    height = 224
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop((height, width)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip(),
        #torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), #added back Thomas's colorjitter
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

    val_dataset, test_dataset = torch.utils.data.random_split(
                                                    test_dataset, 
                                                    [int(len(test_dataset)*0.8), len(test_dataset) - int(len(test_dataset)*0.8)]
                                                    )

    train_loader = DataLoader(train_dataset,batch_size=b,shuffle=True, num_workers=12, persistent_workers=True)
    val_loader = DataLoader(val_dataset,batch_size=b,shuffle=False, num_workers=12, persistent_workers=True)
    test_loader = DataLoader(test_dataset,batch_size=b)

    model = LitNetwork(
                model_name=model_name,
                pretrained=False,
                freeze_backbone=False,
                lr=1e-4, #increased from 1e-4 since learning seems slow
                peak_lr=1e-3, # matches with lr?
                weight_decay=0.01, #was using 0.5 which is too high which was killing learning
                warmup_epochs=3, # passing this in shorter for pretrained
                rampup_epochs=7, #also passing in smaller rampup
                num_epochs=120,
                final_lr_fraction=0.1,
                num_blocks_to_keep=12,
                drop_path_rate=0.05, #adding this for stochastic depth the less the num_blocks_to_keep the smaller drop rate.
                use_dwconv_bypass=True, # test one at a time
                num_local_directional_blocks=0, # last blocks get local directional 0 means alternating
                window_size=7, # 7x7 windows - change to 4x4 because we start with 16x16
                altGlobal=True # alternating blocks global attention vs. local
                )
    checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_acc_epoch', save_top_k=1, mode='max')
    #stop training when validation stops improving:
    # early_stop = pl.callbacks.EarlyStopping(
    #     monitor="val_acc_epoch",
    #     patience = 10,
    #     mode='max'
    # )
    logger = pl_loggers.TensorBoardLogger(save_dir="localDirectionVit_base",name=model_name)
    #logger = pl_loggers.CSVLogger(save_dir="my_logs",name="my_csv_logs")

    #device = "gpu" # Use 'mps' for Mac M1 or M2 Core, 'gpu' for Windows with Nvidia GPU, or 'cpu' for Windows without Nvidia GPU
    device = "gpu"
    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(max_epochs=120, accelerator=device, callbacks=[checkpoint], logger=logger)
    trainer.fit(model,train_loader,val_loader)
    
    trainer.test(ckpt_path="best", dataloaders=test_loader)