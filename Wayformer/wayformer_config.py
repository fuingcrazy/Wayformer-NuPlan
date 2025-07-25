from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import List,Tuple, Dict

import numpy as np
import os,time,math
eps = 1e-5

def rotate(x, y, angle):
    res_x = x * math.cos(angle) - y * math.sin(angle)
    res_y = x * math.sin(angle) + y * math.cos(angle)
    return res_x, res_y


def get_unit_vector(point_a, point_b):
    der_x = point_b[0] - point_a[0]
    der_y = point_b[1] - point_a[1]
    scale = 1 / math.sqrt(der_x ** 2 + der_y ** 2)
    der_x *= scale
    der_y *= scale
    return (der_x, der_y)


def larger(a, b):
    return a > b + eps


def get_dis(points: np.ndarray, point_label):
    return np.sqrt(np.square((points[:, 0] - point_label[0])) + np.square((points[:, 1] - point_label[1])))


def get_angle(x, y):
    return math.atan2(y, x)


def get_time():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


time_begin = get_time()


def get_name(name: str = '', mode: str = 'train', append_time=False):
    if name.endswith(time_begin):
        return name
    if mode == 'test' or mode == 'val':
        prefix = f'{mode}.'
    elif mode == 'train':
        prefix = ''
    else:
        raise NotImplementedError
    suffix = '.' + time_begin if append_time else ''
    return prefix + str(name) + suffix


def get_from_mapping(mappings: List[Dict], key: str):
    return [each_map[key] for each_map in mappings]


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        args = config.from_json(f.read())
    return args


def batch_list_to_batch_tensors(batch):
    return [each for each in batch]

@dataclass
class Metrics:
   k:int
   minADE: float    #Average displacement error
   minFDE: float    #Final displacement error
   missed: int

@dataclass_json
@dataclass
class BlockConfig:
    option:str
    dim_T : int
    dim_S : int
    dim_D : int
    n_head : int
    dim_feedforward : int
    dropout: float = 0.1
    latent_query : bool = False
    lq_ratio : float = 0.5


@dataclass_json
@dataclass
class config:
  exp_name: str
  DATA_PATH = '/home/ethanyu/nuplan/dataset/nuplan-v1.1/splits/mini'
  MAP_PATH  = '/home/ethanyu/nuplan/dataset/maps'
  SAVE_PATH = '/home/ethanyu/VectorNet_NuPlan/processed_data'
  MAP_VERSION = "nuplan-maps-v1.0"    
  temp_file_dir : str = '/home/ethanyu/VectorNet_NuPlan/config'
  SCENARIOS_PER_TYPE = 100

  DISCRETE_SIZE = 10
  PAST_TIME_HORIZON = 2 # [seconds]
  NUM_PAST_POSES = DISCRETE_SIZE * PAST_TIME_HORIZON 
  FUTURE_TIME_HORIZON = 4 # [seconds]
  NUM_FUTURE_POSES = DISCRETE_SIZE * FUTURE_TIME_HORIZON
  NUM_AGENTS :int = 19    # +1(ego)=20
  NUM_HIST = 1   #ego's history trajectory
  # encoding
  OPTION : str = 'factorized'
  HIDDEN_SIZE : int = 256 #dim_D, global hidden dimension for all input tokens
  DIM_T : int =  20        #history window width
  DIM_FEEDFORWARD :int = 512
  DROPOUT : float = 0.1
  LATENT_QUERY : bool = False
  LQ_RATIO = 0.5
  NUM_HEADS = 2
  NUM_BLOCKS = 6    #number of encoder blocks
  LANE_NUM = 60
  ROUTE_LANES_NUM = 10
  CROSSWALKS_NUM = 5
  FACTORIZED : str = 'interleaved'

  LANE_POINTS_NUM = 41
  ROUTE_LANES_POINTS_NUM = 51
  CROSSWALKS_POINTS_NUM = 31


  QUERY_RADIUS:int = 100

  GRAPH_HIDDEN_SIZE = 32
  GRAPH_DEPTH = 1
  GRAPH_HEADS : int = 4
  GRAPH_DIM_FFN : int = 64

  #history embedding
  dim_D_h : int = 4

  #interaction embedding
  dim_D_i : int = 8 
  
  #decoding
  radius:int = 50
  k_components : int = 6
  num_decoder_blocks : int = 4
  pred_horizon : int = 40
  output_dir: str = 'output'
  
  # Fourier embedding parameters
  num_freq_bands: int = 64 

  #training
  batch_size : int = 32
  do_test: bool = False
  do_eval: bool = False
  do_train: bool = True
  learning_rate: float = 6e-4
  num_gpu: int = 1
  max_epochs: int = 40
  data_workers: int = 8
  log_period: int = 2

  def __post_init__(self):   
     output_dir = os.path.join(self.output_dir,self.exp_name)
     if not os.path.exists(output_dir):
        os.makedirs(output_dir)
     if not os.path.exists(self.temp_file_dir):
        os.makedirs(self.temp_file_dir)
  
  def get_block_config(self) -> List[BlockConfig] : 
      if self.OPTION == 'multi_axis':    # O(S^2 T^2)
        blocks, lqs = self.get_multi_axis_blocks()
      elif self.OPTION == 'factorized':   #O(S^2) + O(T^2)
        blocks, lqs = self.get_factorized_blocks()
      result = [BlockConfig(
         option=opt,
         dim_T=self.DIM_T,
         dim_S=self.NUM_AGENTS + self.LANE_NUM + self.NUM_HIST,
         dim_D=self.HIDDEN_SIZE,
         n_head=self.NUM_HEADS,
         dim_feedforward=self.DIM_FEEDFORWARD,
         dropout=self.DROPOUT,
         latent_query=lq,
         lq_ratio=self.LQ_RATIO
      ) for opt, lq in zip(blocks,lqs)]
      return result
  
  def get_factorized_blocks(self) -> Tuple[List[str],List[bool]]:
     assert self.NUM_BLOCKS > 1 and self.NUM_BLOCKS % 2 == 0
     repeat = self.NUM_BLOCKS // 2
     if self.FACTORIZED == 'sequential':
        blocks = ['temporal'] * repeat + ['spacial'] * repeat
     elif self.FACTORIZED == 'interleaved':   
        blocks = ['temporal', 'spacial'] * repeat
     else:
        raise NotImplementedError(f'Unknown option {self.FACTORIZED}')
     if self.LATENT_QUERY:
        if self.FACTORIZED == 'sequential':
           lqs = ([True] + [False] * (repeat - 1)) * 2
        elif self.FACTORIZED == 'interleaved':
           lqs = [True] * 2 + [False] * (self.NUM_BLOCKS - 2)
        else:
           raise NotImplementedError(f'Unknown option {self.FACTORIZED}')
     else:
        lqs = [False]* self.NUM_BLOCKS    
     return blocks, lqs
  

  def get_multi_axis_blocks(self) -> Tuple[List[str],List[bool]]:
     assert self.NUM_BLOCKS > 0
     blocks = ['multi_axis'] * self.NUM_BLOCKS
     if self.LATENT_QUERY: 
        lqs = [True] + [False] * (self.NUM_BLOCKS - 1)
     else:
        lqs = [False] * self.NUM_BLOCKS
     return blocks, lqs
  
  def save_config(self, save_dir: str):
        with open(os.path.join(save_dir, f'{self.exp_name}.json'), 'w') as f:
            f.write(self.to_json(indent=2))