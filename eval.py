import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from utils.data_utils import create_map_raster, draw_trajectory
import os

# Add safe globals for PyTorch 2.6 compatibility
#from torch.serialization import add_safe_globals
from Wayformer.wayformer_config import config as wayformer_config

# Allow the custom global that's causing the issue
#add_safe_globals([wayformer_config])

from Wayformer.wayformer import WayformerPL
from Wayformer.utils import config, batch_list_to_batch_tensors
from Wayformer.wf_dataset import NuplanDataset
from torch.utils.data import DataLoader


if __name__ == "__main__":
    # 为评估添加随机性，避免每次都是相同的结果
    import time
    import random
    eval_seed = int(time.time()) % 1000
    torch.manual_seed(eval_seed)
    random.seed(eval_seed)
    print(f"Evaluation using seed: {eval_seed}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--use_sampling', action='store_true', default=True, help='Use sampling for trajectory generation')
    user_args = parser.parse_args()
    res_path = Path(user_args.ckpt).parent.parent / 'eval.txt'
    print(res_path)
    model: WayformerPL = WayformerPL.load_from_checkpoint(
        checkpoint_path=user_args.ckpt
    )
    model.to('cuda:0')
    model.eval()

    args = model.config
    val_data = NuplanDataset(args, 'val',reuse_cache=True, val_ratio=0.1)
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.data_workers//2,
        collate_fn=batch_list_to_batch_tensors
    )
    num_samples = 0
    ADEs = []
    FDEs = []
    missed = 0

    for i, batch in enumerate(tqdm(val_loader)):
        metrics = model.evaluate_model(batch, k=6)
        ADEs.append(metrics.minADE)
        FDEs.append(metrics.minFDE)
        missed += metrics.missed
        num_samples += len(batch) 


    missRate = missed / num_samples
    ADE = sum(ADEs) / len(ADEs)
    FDE = sum(FDEs) / len(FDEs)
    with open(res_path, 'w') as f:
        f.write(f'minADE: {ADE}\nminFDE: {FDE}\nmissRate: {missRate}\n')