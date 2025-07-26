import torch
import pytorch_lightning as pl
import Wayformer.wayformer_config as wf_cfg
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import os
from Wayformer.wayformer_config import config,load_config,batch_list_to_batch_tensors
from Wayformer.wayformer import WayformerPL
from Wayformer.wf_dataset import NuplanDataset
torch.set_float32_matmul_precision('high')
def train_model(model: WayformerPL):
    config = model.config
    logger = TensorBoardLogger(save_dir=f'{config.output_dir}',name=config.exp_name)
    callbacks = [
        LearningRateMonitor(),
        ModelCheckpoint(
            filename='{epoce}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_last=True
        ),
    ]
    trainer = pl.Trainer(
    accelerator="gpu",         
    devices=config.num_gpu,     
    logger=logger,
    log_every_n_steps=config.log_period,
    max_epochs=config.max_epochs,
    gradient_clip_val=5.0,      
    gradient_clip_algorithm="norm",
    callbacks=callbacks,
    accumulate_grad_batches=1,   
    precision=32,              
    detect_anomaly=True,        
)
    try:
        train_set = NuplanDataset(config=config,mode='train',num_workers=6,reuse_cache=True,val_ratio=0.2)
        val_set = NuplanDataset(config=config,mode='val',reuse_cache=True,num_workers=6,val_ratio=0.2)
        print(f"Dataset loaded successfully: train={len(train_set)}, val={len(val_set)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise e
    #ckpt_path = '/home/ethanyu/VectorNet_NuPlan/output/wayformer.1/version_0/checkpoints/last.ckpt'
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.data_workers,
        collate_fn=batch_list_to_batch_tensors,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,             
        prefetch_factor=2           
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.data_workers,
        collate_fn=batch_list_to_batch_tensors,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        #ckpt_path=ckpt_path
    )


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    parser = ArgumentParser(description="Wayformer NuPlan training entry point")
    parser.add_argument('--config_dir', type=str, default='/home/ethanyu/VectorNet_NuPlan/config/wayformer.1.json')
    args_run = parser.parse_args()
    args: config = load_config(args_run.config_dir)
    
    import time
    seed = int(time.time()) % 10000
    pl.seed_everything(seed, workers=False)  
    print(f"Using dynamic seed: {seed}")
    
    torch.autograd.set_detect_anomaly(True)
    
    try:
        model = WayformerPL(args)
        print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        train_model(model)
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise e