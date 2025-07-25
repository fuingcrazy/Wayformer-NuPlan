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
#torch.serialization.add_safe_globals([wf_cfg.config])
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
    # trainer = pl.Trainer(
    #     logger=logger,
    #     log_every_n_steps = config.log_period,
    #     max_epochs = config.max_epochs,
    #     callbacks = callbacks,
    #     gpus = config.num_gpu
    # )
    trainer = pl.Trainer(
    accelerator="gpu",          # 显式声明使用 GPU
    devices=config.num_gpu,     # 可以是 1 或 [0,1] 这样的列表
    # strategy="auto",          # 分布式策略自动选择，也可显式 ddp
    logger=logger,
    log_every_n_steps=config.log_period,
    max_epochs=config.max_epochs,
    gradient_clip_val=5.0,      # 增加梯度裁剪阈值，参考UniTraj
    gradient_clip_algorithm="norm",
    callbacks=callbacks,
    accumulate_grad_batches=1,   # 梯度累积
    precision=32,               # 使用32位精度确保数值稳定性
    detect_anomaly=True,        # 检测异常值
)
    try:
        train_set = NuplanDataset(config=config,mode='train',num_workers=6,reuse_cache=True,val_ratio=0.2)
        val_set = NuplanDataset(config=config,mode='val',reuse_cache=True,num_workers=6,val_ratio=0.2)
        print(f"Dataset loaded successfully: train={len(train_set)}, val={len(val_set)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise e
    ckpt_path = '/home/ethanyu/VectorNet_NuPlan/output/wayformer.1/version_0/checkpoints/last.ckpt'
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.data_workers,
        collate_fn=batch_list_to_batch_tensors,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,             # 丢弃最后一个不完整的batch
        prefetch_factor=2           # 预取因子
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
    # 设置CUDA内存分配策略
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    parser = ArgumentParser(description="Wayformer NuPlan training entry point")
    parser.add_argument('--config_dir', type=str, default='/home/ethanyu/VectorNet_NuPlan/config/wayformer.1.json')
    args_run = parser.parse_args()
    args: config = load_config(args_run.config_dir)
    
    # 设置随机种子确保可重现性，但允许一定的随机性用于轨迹多样性
    # 注释掉固定种子以增加轨迹生成的多样性
    # pl.seed_everything(0, workers=True)
    
    # 如果需要部分可重现性，可以使用时间戳作为种子
    import time
    seed = int(time.time()) % 10000
    pl.seed_everything(seed, workers=False)  # workers=False允许数据加载器有随机性
    print(f"Using dynamic seed: {seed}")
    
    # 启用异常检测
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