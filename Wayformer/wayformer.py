import numpy as np
import torch
from torch import nn,Tensor
from torch.nn import functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import os

from typing import List, Dict, Tuple, Optional
from einops import rearrange,reduce,repeat
from math import pi, log

from .utils import EarlyFusion,Decoder
from .wayformer_config import get_from_mapping,config,Metrics

class Wayformer(nn.Module):

    def __init__(self,config:config) -> None:
        super(Wayformer,self).__init__()
        self.config = config
        self.encoder = EarlyFusion(config)   #history/interact/road -> Memory
        self.decoder = Decoder(config)    
    
    def forward(self, mappings: List, device) -> Tensor:
        agents = get_from_mapping(mappings,'agents')
        matrix = get_from_mapping(mappings, 'matrix')
        labels = get_from_mapping(mappings, 'labels')
        
        labels = rearrange(labels,'b t d -> b t d')
        labels = torch.as_tensor(labels,device=device,dtype=torch.float32)
        bs = labels.shape[0]
        
        memory,memory_mask = self.encoder(agents,matrix,device)
        
        # Extract ego history
        histories = [agent[0][:,:4] for agent in agents]    
        histories = torch.tensor(rearrange(histories, "A T D -> A T D"), device=device, dtype=torch.float32)
        
        weights, trajs = self.decoder(memory,memory_mask,histories=histories)   #trajs:[B,k,T,4], weights:[B,k]
        pred_labels = torch.zeros_like(weights)
        best_index = self.get_closest_traj(trajs,labels)
        pred_labels[torch.arange(bs),best_index] = 1
        best_trajs = trajs[torch.arange(bs), best_index][...,:3]   # [B,T,3]
        
        cls_loss = F.cross_entropy(weights,pred_labels.detach())
        reg_loss = F.smooth_l1_loss(best_trajs,labels)
        means,logvars = best_trajs, trajs[torch.arange(bs), best_index][...,-2:]   #[3+2]
        negative_log_prob = - self.log_prob_gaussian(means[...,:2], logvars, labels[...,:2])
        
        smoothness_loss = self.compute_smoothness_loss(best_trajs)
    
        loss = 0.6*cls_loss + 0.8*reg_loss + 0.3*negative_log_prob.mean() + 0.4*smoothness_loss
         
        return loss, best_trajs[..., :3]  
    
    def log_prob_gaussian(self, means: Tensor, logvars: Tensor, labels: Tensor) -> Tensor:
        var = torch.exp(logvars)
        log_prob = -0.5 * (log(2 * pi) + logvars + ((labels - means) ** 2) / var)
        return reduce(log_prob, 'b t d -> b', 'sum')

    def compute_smoothness_loss(self, pred_traj: Tensor) -> Tensor:

        if pred_traj.shape[1] < 3:
            return torch.tensor(0.0, device=pred_traj.device)
        
        velocity = pred_traj[:, 1:, :] - pred_traj[:, :-1, :]  # [B, T-1, 2]
        
        acceleration = velocity[:, 1:, :] - velocity[:, :-1, :]  # [B, T-2, 2]
        
        velocity_smoothness = F.mse_loss(velocity[:, 1:, :], velocity[:, :-1, :], reduction='mean')
        
        acceleration_magnitude = torch.norm(acceleration, dim=-1)  # [B, T-2]
        acceleration_loss = F.relu(acceleration_magnitude - 2.0).mean() 
        
        smoothness_loss = velocity_smoothness + 0.1 * acceleration_loss
        
        return smoothness_loss

    
    @torch.no_grad()
    def get_closest_traj(self,trajs: Tensor, labels: Tensor) -> Tensor:
        '''
        trajs:[B,k,T,C],C=5,
        '''
        k = trajs.shape[1]
        xy = trajs[...,:2]  # [B,k,T,2]
        xy_final = xy[..., -1, :]  # [B,k,2]
        labels_final = labels[..., -1, :2]  # [B,2] or [B,1,2]
        
        if labels_final.dim() == 3:
            labels_final = labels_final.squeeze(1)  # [B,2]
        
        labels_final = repeat(labels_final,'b d -> b k d',k=k)  # [B,k,2]
        final_distances = reduce((xy_final - labels_final)**2,'b k d -> b k','sum')
        indices = torch.min(final_distances,dim=1)[1]
        return indices
    
    @torch.no_grad()
    def predict(self,mappings: List, device, num_query: int=-1, use_sampling: bool=True, temperature: float=1.0) -> Tensor:
        agents = get_from_mapping(mappings, 'agents')
        matrix = get_from_mapping(mappings, 'matrix')
        memory,memory_mask = self.encoder(agents,matrix,device)
        
        histories = [agent[0][:,:4] for agent in agents]    
        histories = torch.tensor(rearrange(histories, "A T D -> A T D"), device=device, dtype=torch.float32)
        
        scores, trajs = self.decoder(memory,memory_mask,histories=histories)   #scores;[B,k], trajs:[B,k,t,d]

        k = scores.shape[1]
        if num_query > 0 and num_query < k:
            _, indices = torch.topk(scores, k=num_query, dim=1)
            trajs = trajs[torch.arange(len(trajs))[:, None], indices]
        
        return trajs

    
class WayformerPL(pl.LightningModule):

    def __init__(self,config:config) -> None:
        super(WayformerPL,self).__init__()
        self.config = config
        self.model = Wayformer(config)
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.MultiheadAttention):
                for name, param in m.named_parameters():
                    if param.dim() > 1:
                        if 'weight' in name:
                            torch.nn.init.xavier_normal_(param, gain=np.sqrt(2))
                        elif 'bias' in name:
                            torch.nn.init.constant_(param, 0)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)
            elif hasattr(m, 'weight') and m.weight is not None:
                if m.weight.dim() > 1:
                    torch.nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
                    if hasattr(m, 'bias') and m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0)
        
        self.model.apply(init_weights)
        self.save_hyperparameters()
    
    def forward(self,mappings:List) -> Tensor:

        return self.model(mappings,self.device)
    
    def training_step(self,batch,batch_idx):
        loss,_ = self(batch)
        
        if isinstance(batch, list):
            batch_size = len(batch)
        elif isinstance(batch, tuple):
            batch_size = batch[0].shape[0] if hasattr(batch[0], 'shape') else len(batch)
        else:
            batch_size = batch.shape[0] if hasattr(batch, 'shape') else 1
        
        self.log('train_loss', loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
        return loss
    
    def validation_step(self,batch,batch_idx) -> Tensor:
        loss, _ = self(batch)
        if isinstance(batch, list):
            batch_size = len(batch)
        elif isinstance(batch, tuple):
            batch_size = batch[0].shape[0] if hasattr(batch[0], 'shape') else len(batch)
        else:
            batch_size = batch.shape[0] if hasattr(batch, 'shape') else 1
        
        self.log('val_loss', loss, prog_bar=True, batch_size=batch_size)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=1e-4,
            eps=1e-4  
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.config.learning_rate*2,  
            steps_per_epoch=1, 
            epochs=self.config.max_epochs,
            pct_start=0.1,  
            div_factor=25.0,  
            final_div_factor=100.0  
        )
        
        return [optimizer], [scheduler]
    
    @torch.no_grad()
    def evaluate_model(self,mappings: List,k:int=6,threshold = 2.0, use_sampling: bool=True) -> Metrics:

        trajs = self.model.predict(mappings, self.device, num_query=k, use_sampling=use_sampling, temperature=1.2)   
        labels = get_from_mapping(mappings, 'labels')     #[16,40,3]
        labels = rearrange(labels, 'b t d -> b t d')
        labels = torch.as_tensor(labels, device=self.device)
        pred = trajs[0].cpu().numpy()             # [M, T, 3]
        gt   = labels[0].cpu().numpy()            # [T, 3]
        plt.figure(figsize=(8, 8))
        matrix = get_from_mapping(mappings, 'matrix')
        if matrix is not None:
            lanes = matrix[0] if isinstance(matrix, list) else matrix  
            for lane in lanes:
                x = lane[:, -3]          
                y = lane[:, -4]          
                tls = lane[0, -8:-4] if lane.shape[1] >= 12 else None
                if tls is not None and (tls > 0).any():
                    if tls[0] > 0:  
                     color = '#FF4444'  
                    elif tls[1] > 0:  
                     color = '#FFD700'  
                    elif tls[2] > 0:  
                     color = '#00FF44'  
                    else: 
                     color = '#FFA500'  
                    plt.plot(x, y, color=color, linewidth=2, alpha=0.7)
                else:
                     plt.plot(x, y, color='#6495ED', linewidth=2, alpha=0.5)  
                x_left = lane[:, -12]
                y_left = lane[:, -13]
                plt.plot(x_left, y_left, color='#A9A9A9', linewidth=1, alpha=0.5, linestyle='--')
                x_right = lane[:, -14]
                y_right = lane[:, -15]
                plt.plot(x_right, y_right, color='#A9A9A9', linewidth=1, alpha=0.5, linestyle='--')
        plt.plot(gt[:,0], gt[:,1], 'k-', label='gt')
        trajs_tensor = torch.as_tensor(pred, device=self.device)  # [M, T, 2]
        labels_expanded = labels[0:1]  # [1, T, 2] for single sample
        best_index = self.model.get_closest_traj(trajs_tensor.unsqueeze(0), labels_expanded.unsqueeze(0))[0].item()
        
        for m in range(pred.shape[0]):
            if m == best_index:
                plt.plot(pred[m,:,0], pred[m,:,1], 'b-', alpha=0.8, linewidth=2, label='best_pred')
            else:
                plt.plot(pred[m,:,0], pred[m,:,1], 'gray', alpha=0.4)
        agents = get_from_mapping(mappings, 'agents')
        # agents is a list of agent arrays for the batch; we visualize only the first sample in the batch
        agents_sample = agents[0]  # shape: (num_agents, T, D) or similar
        num_agents = len(agents_sample)
        plt.plot(agents_sample[0,:,0], agents_sample[0,:,1], color='#d33e4c', label='ego_hist', linewidth=2, alpha=0.8)
        plt.scatter(agents_sample[0,0,0], agents_sample[0,0,1], color='#d33e4c', marker='*', s=100, label='ego_start')

        agent_type_colors = {0: '#007672', 1: '#1f77b4', 2: '#ff7f0e'}  # VEHICLE, PEDESTRIAN, BICYCLE
        agent_type_labels = {0: 'vehicle', 1: 'pedestrian', 2: 'bicycle'}
        for i in range(1, num_agents):
            if not np.allclose(agents_sample[i,:,0:2], 0):
                agent_type = int(np.argmax(agents_sample[i,0,4:7]))  # 0: pedestrian, 1: vehicle, 2: bicycle

                color_map = {0: agent_type_colors[1], 1: agent_type_colors[0], 2: agent_type_colors[2]}
                label_map = {0: agent_type_labels[1], 1: agent_type_labels[0], 2: agent_type_labels[2]}
                color = color_map[agent_type]
                label = label_map[agent_type] if i == 1 else None
                plt.plot(agents_sample[i,:,0], agents_sample[i,:,1], color=color, alpha=0.5, linewidth=1)
                plt.scatter(agents_sample[i,0,0], agents_sample[i,0,1], color=color, marker='o', s=40, label=label)

        plt.axis('equal')
        plt.legend()
        plt.show()
        labels = torch.as_tensor(labels, device=self.device)
        ADE = self.get_minADE_k(trajs, labels)
        FDE, missed = self.get_minFDE_k(trajs, labels, threshold)

        return Metrics(trajs.shape[1], ADE, FDE, missed)

    @torch.no_grad()
    def get_minADE_k(self, trajs: Tensor, labels: Tensor) -> float:
        # trajs: (batch_size, k, t, d)
        # labels: (batch_size, t, d)
        k = trajs.shape[1]
        xy = trajs[..., :2]
        labels = repeat(labels[:,:,:2], 'b t d -> b k t d', k=k)
        ADEs = reduce((xy - labels)**2, 'b k t d -> b k t', 'sum')
        ADEs = reduce(torch.sqrt(ADEs), 'b k t -> b k', 'mean')
        minADEs = reduce(ADEs, 'b k -> b', 'min')
        return minADEs.mean().cpu().item()
    
    @torch.no_grad()
    def get_minFDE_k(self, trajs: Tensor, labels: Tensor, threshold: float) -> Tuple[float, int]:
        # get minFDE and missed predictions
        # trajs: (batch_size, k, t, d)
        # labels: (batch_size, t, d)
        k = trajs.shape[1]
        xy_final = trajs[..., -1, :2]
        labels = repeat(labels, 'b t d -> b k t d', k=k)
        labels = labels[..., -1, :2]
        FDEs = reduce((xy_final - labels)**2, 'b k d -> b k', 'sum')
        FDEs = (torch.sqrt(FDEs))
        minFDEs = reduce(FDEs, 'b k -> b', 'min')
        return minFDEs.mean().cpu().item(), (minFDEs > threshold).sum().cpu().item()






