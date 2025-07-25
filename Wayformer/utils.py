import torch
import numpy as np
import torch.nn.functional as F
from .wayformer_config import BlockConfig,config
from .fourier_embedding import FourierEmbedding
from nuplan.database.nuplan_db.nuplan_scenario_queries import *

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.tracked_objects import TrackedObjects,TrackedObjectType
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import *

from torch import nn, Tensor
from einops import parse_shape, rearrange, repeat

def interpolate_polyline(points: np.ndarray, t: int) -> np.ndarray:
    """copy from av2-api"""

    if points.ndim != 2:
        raise ValueError("Input array must be (N,2) or (N,3) in shape.")

    # the number of points on the curve itself
    n, _ = points.shape

    # equally spaced in arclength -- the number of points that will be uniformly interpolated
    eq_spaced_points = np.linspace(0, 1, t)

    # Compute the chordal arclength of each segment.
    # Compute differences between each x coord, to get the dx's
    # Do the same to get dy's. Then the hypotenuse length is computed as a norm.
    chordlen: np.ndarray = np.linalg.norm(np.diff(points, axis=0), axis=1)  # type: ignore
    # Normalize the arclengths to a unit total
    chordlen = chordlen / np.sum(chordlen)
    # cumulative arclength

    cumarc: np.ndarray = np.zeros(len(chordlen) + 1)
    cumarc[1:] = np.cumsum(chordlen)

    # which interval did each point fall in, in terms of eq_spaced_points? (bin index)
    tbins: np.ndarray = np.digitize(eq_spaced_points, bins=cumarc).astype(int)  # type: ignore

    # #catch any problems at the ends
    tbins[np.where((tbins <= 0) | (eq_spaced_points <= 0))] = 1  # type: ignore
    tbins[np.where((tbins >= n) | (eq_spaced_points >= 1))] = n - 1

    chordlen[tbins - 1] = np.where(
        chordlen[tbins - 1] == 0, chordlen[tbins - 1] + 1e-6, chordlen[tbins - 1]
    )

    s = np.divide((eq_spaced_points - cumarc[tbins - 1]), chordlen[tbins - 1])
    anchors = points[tbins - 1, :]
    # broadcast to scale each row of `points` by a different row of s
    offsets = (points[tbins, :] - points[tbins - 1, :]) * s.reshape(-1, 1)
    points_interp: np.ndarray = anchors + offsets

    return points_interp


def sample_discrete_path(discrete_path: List[StateSE2], num_points: int):
    path = np.stack([point.array for point in discrete_path], axis=0)
    return interpolate_polyline(path, num_points)


def get_neighbor_lanes_polygon(map_api:AbstractMap,
                      map_features:List[str],
                      point:Point2D,
                      rot:float,
                      radius:float,
                      route_roadblock_ids: List[str],
                      sample_points:int,
                      traffic_light_status_data: List[TrafficLightStatusData]) -> Tuple[np.ndarray,np.ndarray]:
    """
    points: point in global frame, need to be transformed.
    returns a list of n polygons, each polygon is a [N,2] numpy array, n is the number of lanes
    """
    polygons : List[np.ndarray] = []
    polygons_left : List[np.ndarray] = []
    polygons_right : List[np.ndarray] = []
    feature_layer : List[VectorFeatureLayer] = []
    mat_rot = np.array([[np.cos(rot), -np.sin(rot)],
                        [np.sin(rot), np.cos(rot)]])
    map_objects = map_api.get_proximal_map_objects(point,radius,
                                                   [
                SemanticMapLayer.LANE,
                SemanticMapLayer.LANE_CONNECTOR,
                SemanticMapLayer.CROSSWALK,
            ],)
    lane_objects = (
            map_objects[SemanticMapLayer.LANE]
            + map_objects[SemanticMapLayer.LANE_CONNECTOR]
        )
    _, _, _, lane_ids = get_lane_polylines(map_api, point, radius)
    lanes_tls_vector = get_traffic_light_encoding(lane_ids,traffic_light_status_data).to_vector()
    assert len(lane_objects) == len(lanes_tls_vector)
    for lane in lane_objects:
        centerline = sample_discrete_path(     #[21,2]
                lane.baseline_path.discrete_path, sample_points + 1
            )
        left_bound = sample_discrete_path(
                lane.left_boundary.discrete_path, sample_points + 1
            )
        right_bound = sample_discrete_path(
                lane.right_boundary.discrete_path, sample_points + 1
            )
        shifted = centerline - np.array([point.x,point.y],dtype=np.float32)
        shifted_left = left_bound - np.array([point.x,point.y],dtype=np.float32)
        shifted_right = right_bound - np.array([point.x,point.y],dtype=np.float32)
        polygon = (mat_rot @ shifted.T).T
        polygon_left = (mat_rot @ shifted_left.T).T
        polygon_right = (mat_rot @ shifted_right.T).T
        polygons.append(polygon)
        polygons_left.append(polygon_left)
        polygons_right.append(polygon_right)
    
    return polygons, polygons_left, polygons_right, lanes_tls_vector
        
def merge_tensors(tensors: List[Tensor], device, max_length: int = None) -> Tuple[Tensor, float, List]:
    lengths = [t.shape[0] for t in tensors]
    if max_length is None:
        max_length = max(lengths)
    res = torch.zeros((len(tensors), max_length, *tensors[0].shape[1:]), device=device)
    for i, tensor in enumerate(tensors):
        res[i, :tensor.shape[0]] = tensor
    # res: (batch_size, max_length, embedding_dim)
    return res, max_length, lengths


def get_1D_padding_mask(lengths: List[int], max_length: int, device) -> Tensor:
    '''
    [B,max_len],with invalid position to be true
    '''
    mask = torch.zeros((len(lengths), max_length), dtype=torch.bool, device=device)
    for i, length in enumerate(lengths):
        mask[i, length:] = True
    return mask

def get_src_mask(src_key_padding_mask: Tensor, num_heads: int) -> Tensor:
    """
    input: [B, L], src_mask : [B*num_heads,L,L]
    """
    batch_size, src_len = src_key_padding_mask.shape
    batch_size, src_len = src_key_padding_mask.shape
    src_mask = torch.zeros((batch_size, src_len, src_len), device=src_key_padding_mask.device)
    src_mask.masked_fill_(src_key_padding_mask.unsqueeze(1), -1e5)
    src_mask = src_mask.repeat_interleave(num_heads, dim=0)
    return src_mask



def get_2D_padding_mask(lengths: List[List[int]], max_length1: int, max_length2: int, device) -> Tensor:
    mask = torch.zeros((len(lengths), max_length1, max_length2), dtype=torch.bool, device=device)
    for i, len_list in enumerate(lengths):
        length1 = len(len_list)
        mask[i, length1:, :] = True
        for j, length2 in enumerate(len_list):
            mask[i, j, length2:] = True
    return mask

def wrap_angle(angle:float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi    

def extract_agent_feature(type_list,tracked_objects,track_token_ids,timestamp):
    '''
    extract specific type agents, returns [num_agents,8] tensor
    '''
    agents = tracked_objects.get_tracked_objects_of_types(type_list)   #Filtered agents
    output = torch.zeros((len(agents),8),dtype=torch.float32)
    max_agent_ids = len(track_token_ids)
    for idx, agent in enumerate(agents):
        if agent.track_token not in track_token_ids:
            track_token_ids[agent.track_token] = max_agent_ids
            max_agent_ids += 1
        track_token_int = track_token_ids[agent.track_token]
        output[idx,0] = agent.center.x
        output[idx,1] = agent.center.y
        output[idx,2] = agent.center.heading 
        output[idx,3] = float(timestamp)  # Store agent index instead of timestamp
        output[idx,4] = agent.tracked_object_type == TrackedObjectType.PEDESTRIAN
        output[idx,5] = agent.tracked_object_type == TrackedObjectType.VEHICLE
        output[idx,6] = agent.tracked_object_type == TrackedObjectType.BICYCLE
        output[idx,7] = float(track_token_int)
    
    return output,track_token_ids

def sampled_tracked_ego_to_tensor(past_ego_states:List[EgoState],
                                  rot: float,
                                  ego_x:float,
                                  ego_y:float,
                                  isPast:bool,
                                  past_timestamps:List[TimePoint]) -> torch.Tensor:
    '''
    extract ego states and transform to [T,D] tensor
    '''
    #assert len(past_ego_states) == len(past_timestamps)     #both 20
    if isPast:
      output = torch.zeros((len(past_timestamps),8),dtype=torch.float32)
      initial_time = past_timestamps[0].time_s
    else:
        output = torch.zeros((config.NUM_FUTURE_POSES,8),dtype=torch.float32)
    c,s = np.cos(rot),np.sin(rot)
    rot_mat = np.array([[c, -s],
                        [s, c]])    
    for i, ego_state in enumerate(past_ego_states):
        dx ,dy = ego_state.rear_axle.x - ego_x, ego_state.rear_axle.y - ego_y
        output[i,0], output[i,1] = rot_mat @ np.array([dx,dy])
        output[i,2] = wrap_angle(ego_state.rear_axle.heading + rot)
        if isPast:
          output[i,3] = int(past_timestamps[i].time_s - initial_time)
    return output


def sampled_tracked_objects_to_agents(sampled_past_observations : List[TrackedObjects],
                                      rot: float,
                                      ego_x:float,
                                      ego_y:float,
                                      past_timestamp:List[TimePoint]) -> np.ndarray:
    '''
    output: agents history features for mapping['agents], shape:[B,T,D] 20 past+1 now
    '''
    object_type = [TrackedObjectType.PEDESTRIAN,TrackedObjectType.VEHICLE,TrackedObjectType.BICYCLE]
    agents_per_timestamp = list()
    track_token_ids = {}
    initial_time = past_timestamp[0].time_s
    flat_timestamp = [t.time_s - initial_time for t in past_timestamp]
    assert len(flat_timestamp) == len(sampled_past_observations)
    c,s = np.cos(rot),np.sin(rot)
    rot_mat = np.array([[c, -s],
                        [s, c]])    
    
    # First pass: collect all agents and build track_token_ids
    for i in range(len(sampled_past_observations)):
        tensorized_agent,track_token_ids = extract_agent_feature(object_type,sampled_past_observations[i],track_token_ids,flat_timestamp[i])
        agents_per_timestamp.append(tensorized_agent)
    
    T = len(sampled_past_observations)
    num_agents = len(track_token_ids)
    # Limit the number of agents to prevent memory issues
    max_agents = 100  # Set a reasonable limit
    if num_agents > max_agents:
        num_agents = max_agents
    
    output_agents = torch.ones((num_agents,T,8),dtype=torch.float32)
    
    for t,features in enumerate(agents_per_timestamp):
        for row in features:
            agent_idx = int(row[7].item())
            # Ensure agent_idx is within bounds
            if agent_idx < num_agents:
                dx,dy = row[0] - ego_x, row[1] - ego_y
                row[0], row[1] = rot_mat @ np.array([dx,dy])   
                row[2] = wrap_angle(row[2] + rot)
                output_agents[agent_idx,t,:] = row
    #print(output_agents.shape)
    last_timestamps = output_agents[:, -1, 3] 
    firt_timestamps = output_agents[:,0,3]

# only record agents with full timestamps(from 0 to 2.0s)
    mask = (last_timestamps >= 1.8) & (firt_timestamps < 0.5)
    masked = output_agents[mask]
    return masked



def batch_list_to_batch_tensors(batch):
    return [each for each in batch]


class LayerNorm(nn.Module):
    """
    layer normalization
    Input normalization, N(0,1),eps: stablize the input in case the denominator is 0
    bias, weight are learnable params, this layer can reduce offsets between different inputs, accelerate convergence
    """
    def __init__(self,hidden_size:int, eps:float=1e-5):
        super(LayerNorm,self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self,x:Tensor):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias

class MLP(nn.Module):
    def __init__(self,hidden_size:int, out_features=None):
        super(MLP,self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size,out_features)
        self.layer_norm = LayerNorm(out_features)
    
    def forward(self,hidden_states: Tensor):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = torch.nn.functional.relu(hidden_states)
        return hidden_states

class Projection(nn.Module):

    def __init__(self,dim_in:int, dim_out:int) ->None:
        super(Projection,self).__init__()
        self.linear = nn.Linear(dim_in,dim_out)
    
    def forward(self,x: Tensor) ->Tensor:
        x = self.linear(x)
        x = F.relu(x)
        return x

class RoadGraph(nn.Module):

    """
    Encode S polylines of V segments into (B,S,hidden_dim) polyline features 
    """
    def __init__(self,config:config) -> None:
        super(RoadGraph,self).__init__()
        self.config = config
        layer = nn.TransformerEncoderLayer(
            d_model=config.GRAPH_HIDDEN_SIZE,
            nhead=config.GRAPH_HEADS,
            dim_feedforward=config.GRAPH_DIM_FFN,
            dropout=config.DROPOUT,
            batch_first=True
        )
        self.model = nn.TransformerEncoder(layer,num_layers=config.GRAPH_DEPTH)
        self.projection = Projection(config.GRAPH_HIDDEN_SIZE,config.HIDDEN_SIZE)    #32 -> 256
           
    def forward(self,x:Tensor) ->Tensor:
        #x:(S,V,D) V - num of vectors
        x = self.model(x)    # multi-head self attention on 9-element sequence, stack 2 times
        x = torch.max(x,dim=1)[0]    #(S,D), pooling,
        return self.projection(x)    #(s,hidden_size), project 128 road features onto 256 

class LatentQuery(nn.Module):

    def __init__(self,config:config) ->None:
        super(LatentQuery,self).__init__()
        return NotImplementedError

class EncoderBlock(nn.Module):

    def __init__(self,config:BlockConfig) -> None:
        super().__init__()
        
        option = config.option
        dim_D = config.dim_D
        dim_T = config.dim_T
        dim_S = config.dim_S
        n_head = config.n_head
        dim_feedforward = config.dim_feedforward
        self.latent_query = config.latent_query
        self.lq_ratio = config.lq_ratio   
        dropout = config.dropout
        

        #[A, T, S, D]
        assert not self.latent_query
        if option == 'temporal':
            self.in_pattern = 'A T S D -> (A S) T D'    
            self.in_pattern_mask = 'A T S -> (A S) T'  
            self.out_pattern = '(A S) T D -> A T S D'   
            self.L_in = dim_T                           
        elif option == 'spacial':
            self.in_pattern = 'A T S D -> (A T) S D'
            self.in_pattern_mask = 'A T S -> (A T) S'
            self.out_pattern = '(A T) S D -> A T S D'
            self.L_in = dim_T
        elif option == 'multi_axis':
            self.in_pattern = 'A T S D -> A (T S) D'
            self.in_pattern_mask = 'A T S -> A (T S)'
            self.out_pattern = 'A (T S) D -> A T S D'
            self.L_in = dim_T * dim_S
        else:
            raise NotImplementedError(f'EncoderBlock with {option} not implemented')
        self.apply_out_rearrange = (option == 'spacial' or option == 'temporal')
        if self.latent_query:
            self.L_out = int(self.L_in * self.lq_ratio)
            self.latent_pattern = 'A L D -> A D L'   
            self.latent_mapping = nn.Linear(self.L_in, self.L_out)
        
        self.tfm_layer = nn.TransformerEncoderLayer(
            d_model=dim_D,   
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LayerNorm for better stability
        )
        self.n_head = n_head

    def forward(self, x:Tensor, mask: Tensor = None) ->Tensor:
        if x.dim() == 4:   # x:[A T S D]
            axes_lengths = parse_shape(x,'A T S D')    
            x = rearrange(x,self.in_pattern)    
        else:
            assert x.dim() == 3
        if mask is not None:
            mask = rearrange(mask,self.in_pattern_mask)
            all_true_rows = mask.all(dim=1)
            
            if all_true_rows.any():
                mask[all_true_rows] = False
            
            #src_key_padding_mask = mask
            src_mask = get_src_mask(mask,config.NUM_HEADS)
        if self.latent_query:    #L_in -> L_out
            x = rearrange(x,self.latent_pattern)
            x = self.latent_mapping(x)
            x = rearrange(x,self.latent_pattern)
            axes_lengths = {}
        # x : [A, L, D]
        x = self.tfm_layer(x, src_mask = src_mask)
        if self.apply_out_rearrange:    
            x = rearrange(x,self.out_pattern,**axes_lengths)
        return x
    

class Encoder(nn.Module):
    """
    [B,N,D],N = T*S or N = Lq
    """
    def __init__(self, config:config) -> None:
        super().__init__()
        block_configs = config.get_block_config()
        self.blocks = nn.ModuleList([EncoderBlock(block_cfg) for block_cfg in block_configs])
        self.num_blocks = len(self.blocks)
        self.norm = nn.LayerNorm(config.HIDDEN_SIZE)    

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        for i, block in enumerate(self.blocks):
            x = block(x, mask)
            if (i == self.num_blocks - 1) and (x.dim() == 4):    #rearrange to [A,N,D] at the last encoder block if not multi-axis
                x = rearrange(x,'A T S D -> A (T S) D')     
        x = self.norm(x)
        return x

class Decoder(nn.Module):

    def __init__(self,config:config):
        super(Decoder,self).__init__()
        self.config = config
        
        # lateral query: random learnable parameters 
        lateral_qs = torch.randn(config.k_components, config.HIDDEN_SIZE // 2)
        self.lateral_queries = nn.Parameter(lateral_qs)
        
        # Longitudinal query: encode history trajectory features
        self.hist_encoder = FourierEmbedding(3, config.HIDDEN_SIZE // 2, 64)
        self.projection = nn.Linear(config.dim_D_h, config.HIDDEN_SIZE // 2)
        self.longitudinal_proj = nn.Linear(config.DIM_T * (config.HIDDEN_SIZE // 2), config.HIDDEN_SIZE // 2)
        
        # Fusion projection
        self.query_fusion = nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE)
        
        layer = nn.TransformerDecoderLayer(
            d_model=config.HIDDEN_SIZE,
            nhead=config.NUM_HEADS,
            dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT,
            batch_first=True,
            norm_first=True  # Pre-LayerNorm for better stability
        )
        self.model = nn.TransformerDecoder(layer,num_layers=config.num_decoder_blocks)
        
        self.classifier = nn.Linear(config.HIDDEN_SIZE, 1)
        self.regressor = nn.Linear(config.HIDDEN_SIZE,config.pred_horizon*5)
        
        nn.init.normal_(self.lateral_queries, mean=0.0, std=0.01)
    
    def forward(self,memory: Tensor, mask: Tensor = None, histories: Tensor = None) -> Tuple[Tensor]:
        batch_size = memory.shape[0]
        lateral_queries = repeat(self.lateral_queries, 'k d -> b k d', b=batch_size)  # [B, k, D//2]
        if histories is not None:
            coords = histories[:,:,:3]    # x,y,heading [B,T,3]
            pos_enc = self.hist_encoder(coords)   # [B,T,hidden_dim//2]
            hist_emb = self.projection(histories) + pos_enc  # [B,T,hidden_dim//2]
            
            hist_flat = rearrange(hist_emb, 'b t d -> b (t d)')  # [B, T*(D//2)]
            longitudinal_query = self.longitudinal_proj(hist_flat)  # [B, D//2], time fusion
            #Repeat k times to get k modes
            longitudinal_queries = repeat(longitudinal_query, 'b d -> b k d', k=self.config.k_components)  # [B,k,D//2]
        else:
            longitudinal_queries = torch.zeros(batch_size, self.config.k_components, self.config.HIDDEN_SIZE // 2, 
                                             device=memory.device, dtype=memory.dtype)
        
        combined_queries = torch.cat([lateral_queries, longitudinal_queries], dim=-1)  # [B, k, D]
        target = self.query_fusion(combined_queries)  # [B, k, D]
        if mask is not None and mask.dtype != torch.bool:
            mask = mask.bool()
        embeddings = self.model(target, memory, memory_key_padding_mask=mask)  # [B, k, D]

        cls_head = self.classifier(embeddings).squeeze(-1)   # [B, k]
        reg_head = self.regressor(embeddings)  # [B, k, T*5]
        reg_head = rearrange(reg_head,'b k (t c) -> b k t c', c=5)   # [B, k, T, 5]
        
        return cls_head, reg_head


class EarlyFusion(nn.Module):

    def  __init__(self,config:config) -> None:
        super(EarlyFusion,self).__init__()
        self.config = config
        self.projection1 = Projection(config.dim_D_h,config.HIDDEN_SIZE)
        self.projection2 = Projection(config.dim_D_i,config.HIDDEN_SIZE)
        self.roadgraph = RoadGraph(config=config) 
        self.encoder = Encoder(config)   
        self.pos_emb = FourierEmbedding(
            input_dim=3,  # x, y, heading
            hidden_dim=config.HIDDEN_SIZE,
            num_freq_bands=getattr(config, 'num_freq_bands', 64)
        )
        self.pos_emb_interact = FourierEmbedding(
            input_dim=3,  # x, y, heading
            hidden_dim=config.HIDDEN_SIZE,
            num_freq_bands=getattr(config, 'num_freq_bands', 64)
        )
        self.pos_emb_road = FourierEmbedding(
            input_dim=2,  # x, y
            hidden_dim=config.HIDDEN_SIZE,
            num_freq_bands=getattr(config, 'num_freq_bands', 64)
        )
    
    def forward_road_embedding(self,matrix:List[np.ndarray],device) -> Tuple[Tensor]:
        batch_size = len(matrix)
        batch_embedding = []
        for i in range(batch_size):
            tensor = torch.tensor(matrix[i],device=device,dtype=torch.float32)  
            embedding = self.roadgraph(tensor)  # [S, HIDDEN_SIZE=256]
            
            if tensor.shape[2] >= 2 and tensor.shape[0] > 0 and embedding.shape[0] > 0:
                coords = tensor[:, :, -2:]  #[S, 20, 2]
                center = torch.mean(coords, dim=1)  # [S, 2] 
                pos_encoding = self.pos_emb_road(center.unsqueeze(0)).squeeze(0)  # [S, HIDDEN_SIZE]
                if pos_encoding.shape[0] == embedding.shape[0]:
                    embedding = embedding + pos_encoding
            
            batch_embedding.append(embedding)
        
        embeddings, max_len,lengths = merge_tensors(batch_embedding,device)
        padding_mask = get_1D_padding_mask(lengths,max_len,device)   #shape:[B,max_len]

        embeddings = repeat(embeddings,'A S D -> A T S D',T=self.config.DIM_T)  
        padding_mask = repeat(padding_mask,'A S -> A T S',T=self.config.DIM_T).to(torch.bool)
        return embeddings,padding_mask
    
    def forward_interact_embedding(self,agents:List[List[np.ndarray]],device) -> Tuple[Tensor]:
        '''
        Input: [B,S_i,D],S_i is not fixed
        '''
        length_list = list()
        embedding_list = list()
        for agent in agents:
            input_list = [torch.tensor(a,device=device) for a in agent]

            embeddings,_,lengths = merge_tensors(input_list,device,max_length=self.config.DIM_T)    
            length_list.append(lengths)
            embedding_list.append(embeddings)
        embeddings,max_len,lengths = merge_tensors(embedding_list,device)

        # Positional encoding before projection
        coords = embeddings[:, :, :, :3]  # [A, S, T, 3], xy heading
        A, S, T, _ = coords.shape
        coords_reshaped = coords.view(A*S, T, 3)
        pos_encoding = self.pos_emb_interact(coords_reshaped)  # [A*S, T, HIDDEN_SIZE]
        pos_encoding = pos_encoding.view(A, S, T, -1)
        
        embeddings = self.projection2(embeddings)
        embeddings = embeddings + pos_encoding 
        
        padding_mask = get_2D_padding_mask(length_list,max_len,self.config.DIM_T,device)   #shape[B,S_i,T]
        if padding_mask.dtype != torch.bool:
            padding_mask = padding_mask.bool()
        embeddings = rearrange(embeddings, 'A S T D -> A T S D')
        padding_mask = rearrange(padding_mask, 'A S T -> A T S')
        
        return embeddings,padding_mask
    
    def forward_history_embedding(self,agents: List[List[np.ndarray]],device) ->Tuple[Tensor]:
        batch_size = len(agents)
        histories = [agent[0][:,:4] for agent in agents]   
        histories = torch.tensor(rearrange(histories, "A T D -> A T D"),device=device)
        
        coords = histories[:, :, :3]  
        pos_encoding = self.pos_emb(coords)  
        
        embeddings = self.projection1(histories)    
        embeddings = embeddings + pos_encoding  
        embeddings = repeat(embeddings, 'A T D -> A T 1 D')
        padding_mask = torch.zeros(batch_size,self.config.DIM_T,1,dtype=torch.bool,device=device)
        if padding_mask.dtype != torch.bool:
            padding_mask = padding_mask.bool()
        return embeddings,padding_mask
    

    def forward(self,agents: List[List[np.ndarray]],matrix: List[np.ndarray],device) -> Tensor:
        '''
        Early fusion: concatenate all features to be a [A,T,S_total,D] token,
        where S_total = S_i + S_r + 1 ,memory is [B,N,D], N=T*S
        '''
        try:
            road, road_mask = self.forward_road_embedding(matrix,device)
            interact, interact_mask = self.forward_interact_embedding(agents,device)
            hist,hist_mask = self.forward_history_embedding(agents,device)
            
            embedding = torch.cat([road,interact,hist],dim=2)  
            padding_mask = torch.cat([road_mask,interact_mask,hist_mask],dim=2)
            
            assert embedding.shape[:3] == padding_mask.shape   #[A T S]
            memory_mask = rearrange(padding_mask, 'A T S -> A (T S)')   
            
            memory = self.encoder(embedding,padding_mask)   
                
            return memory,memory_mask
            
        except Exception as e:
            print(f"Error in EarlyFusion forward: {e}")
            batch_size = len(agents)
            memory = torch.zeros(batch_size, 1, self.config.HIDDEN_SIZE, device=device)
            memory_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
            return memory, memory_mask




        
