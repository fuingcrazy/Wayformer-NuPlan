import torch
import torch.nn as nn
from .fourier_embedding import FourierEmbedding
from .wayformer_config import config

class StateAttnEncoder(nn.Module):

    def __init__(self,state_channel:int, dropout:float)  -> None:
        super().__init__()
        self.state_channel = state_channel
        self.dropout = dropout
        self.projection = nn.Linear(config.NUM_PAST_POSES,config.HIDDEN_SIZE)   #project timespan to hidden dim
        self.attn = nn.MultiheadAttention(config.HIDDEN_SIZE,
                                          num_heads=config.NUM_HEADS,
                                          dropout=self.dropout,
                                          batch_first=True)
        self.query = nn.Parameter(torch.Tensor(1,1,config.HIDDEN_SIZE))
        self.pos_emb = nn.Parameter(torch.Tensor(1,state_channel,config.HIDDEN_SIZE))

        nn.init.normal_(self.query,std=0.02)
        nn.init.normal_(self.pos_emb,std=0.02)
    
    def forward(self,histories: torch.Tensor):
        '''
         histories: tensor of shape [A,T,D],should be 16,20,9
         output: tensor of hist feature :[A,1,D]
        '''
        A,T,D_in = histories.shape
        query = self.query.repeat(A,1,1)   
        pos_emb = self.pos_emb.repeat(A,1,1)
        histories = self.projection(histories.transpose(1,2))   # [A,D_state,D_hidden]
        x_emb = histories + pos_emb
        if self.training and self.dropout > 0:
            visible_tokens = torch.zeros((histories.shape[0],3),device=histories.device, dtype=torch.bool)    #x,y,heading all visible
            dropout_tokens = ( torch.rand((histories.shape[0],5),device=histories.device) < self.dropout )    #acc,vel,etc. randomly dropout
            key_padding_mask = torch.cat([visible_tokens,dropout_tokens],dim=1)
        else:
            key_padding_mask = None
        feat, _ = self.attn(
            query = query,
            key = x_emb,
            value = x_emb,
            key_padding_mask = key_padding_mask
        )     #[A,1,D]
        return feat


