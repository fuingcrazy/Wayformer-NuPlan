import os
import numpy as np
import matplotlib.pyplot as plt
import pickle,random,multiprocessing as mp,zlib

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in cast')

from tqdm import tqdm
from .utils import *
from .wayformer_config import config
from utils.common_utils import (
    get_scenario_map,
    get_filter_parameters
)
from einops import rearrange

from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.common.actor_state.state_representation import Point2D


VECTOR_PRE_X = 0
VECTOR_PRE_Y = 1
VECTOR_X = 2
VECTOR_Y = 3
COMPRESS = {"zlib": (zlib.compress, zlib.decompress)}
compress, decompress = COMPRESS["zlib"]

class data_processor(object):

    def __init__(self,scenario):
        
        self.scenario = scenario
        self.map_api = scenario.map_api

        self.discrete_size       = config.DISCRETE_SIZE
        self.past_time_horizon   = config.PAST_TIME_HORIZON
        self.num_past_poses      = config.NUM_PAST_POSES
        self.future_time_horizon = config.FUTURE_TIME_HORIZON
        self.num_future_poses    = config.NUM_FUTURE_POSES
        self.num_agents          = config.NUM_AGENTS

        # name of map features to be extracted.
        self._map_features = [
            'LANE', 
            'ROUTE_LANES', 
            'CROSSWALK'
        ] 
        
        # maximum number of elements to extract per feature layer.
        self._max_elements = {
            'LANE'       : config.LANE_NUM, 
            'ROUTE_LANES': config.ROUTE_LANES_NUM, 
            'CROSSWALK'  : config.CROSSWALKS_NUM
        } 
        
        # maximum number of points per feature to extract per feature layer.
        self._max_points = {
            'LANE'       : config.LANE_POINTS_NUM, 
            'ROUTE_LANES': config.ROUTE_LANES_POINTS_NUM, 
            'CROSSWALK'  : config.CROSSWALKS_POINTS_NUM
        } 
        
        # [m] query radius scope relative to the current pose.
        self._radius = config.QUERY_RADIUS 
        
        self._interpolation_method = 'linear' # interpolation method for points in the map features.
    
    
    def get_min_distance(self,polygon : np.ndarray) ->float:
        """
        polygon: aray of [N,2]
        """
        dists = polygon[:,0] ** 2 + polygon[:,1] ** 2
        return np.min(dists)
    
    def sort_agents_hist_by_distance(self,agents_hist: torch.Tensor) -> torch.Tensor:

        dist2 = (agents_hist[:, :, 0:2] ** 2).sum(dim=-1)

        pad_mask = (agents_hist[:, :, 0] == 0) & (agents_hist[:, :, 1] == 0)
        dist2[pad_mask] = 1e8

        min_dist, _ = dist2.min(dim=1)        # [N]

        order = min_dist.argsort()           
        return agents_hist[order]             
    
    def get_map_features(self) -> np.ndarray:
        """
        extract lanes close to ego car
        """
        ego_state = self.scenario.initial_ego_state
        ego_head = ego_state.rear_axle.heading
        ego_rot = -ego_head + np.pi / 2.0     #rotate to +Y direction
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        route_roadblock_ids = self.scenario.get_route_roadblock_ids()
        traffic_light_data = self.scenario.get_traffic_light_status_at_iteration(0)
        vectors = list()
        polyline_span = list()
        sample_points = 20
        
        lane_polygons, lane_polygons_left, lane_polygons_right, tls_data = get_neighbor_lanes_polygon(
            self.map_api,
            self._map_features,
            ego_coords,
            ego_rot,
            self._radius,
            route_roadblock_ids,
            sample_points,
            traffic_light_data
        )
        # tls_data: green [1, 0, 0, 0] yellow [0, 1, 0, 0], red [0, 0, 1, 0], unknown [0, 0, 0, 1]
        #print(f'lanes number:{len(lane_polygons)}, traffic light shape:{tls_data.shape}')
        lane_polygons = sorted(lane_polygons,key=self.get_min_distance)
        lane_polygons_left = sorted(lane_polygons_left,key=self.get_min_distance)
        lane_polygons_right = sorted(lane_polygons_right,key=self.get_min_distance)
        
        for idx_polygon, polygon in enumerate(lane_polygons):
            start = len(vectors)
            poly_idx = idx_polygon
            for i in range(1,len(polygon)):
                    point_pre,point = polygon[i-1], polygon[i]
                    vector = [0] * config.GRAPH_HIDDEN_SIZE
                    vector[-1 - VECTOR_PRE_X], vector[-1 - VECTOR_PRE_Y] = point_pre[0],point_pre[1]
                    vector[-1 - VECTOR_X], vector[-1 - VECTOR_Y] = point[0],point[1]
                    vector[-5] = 1.0   #lane flag
                    vector[-6] = i
                    vector[-7] = poly_idx
                    point_pre_pre = (2 * point_pre[0] - point[0], 2 * point_pre[1] - point[1])
                    vector[-8] = tls_data[poly_idx][0]
                    vector[-9] = tls_data[poly_idx][1]
                    vector[-10] = tls_data[poly_idx][2]
                    vector[-11] = tls_data[poly_idx][3]
                    if(i >= 2):
                      point_pre_pre = polygon[i-2]
                    vector[-12] = lane_polygons_left[poly_idx][i][0]
                    vector[-13] = lane_polygons_left[poly_idx][i][1]
                    vector[-14] = lane_polygons_right[poly_idx][i][0]
                    vector[-15] = lane_polygons_right[poly_idx][i][1]   #add left/right boundaries
                    vector[-17] = point_pre_pre[0]
                    vector[-18] = point_pre_pre[1]

                    vectors.append(vector)
            end = len(vectors)
            if(start < end):
                polyline_span.append([start, end])
            if len(polyline_span) > self._max_elements['LANE']:
                break
        vectors = np.array(vectors, dtype=np.float32)
        lane_count = len(polyline_span)
        matrix = rearrange(vectors, '(S V) D -> S V D', V=20)

        if lane_count > config.LANE_NUM:
             matrix = matrix[:config.LANE_NUM]
        return matrix
    
    def collect_agents(self):
       """
       collect agents from scenario (history)s
       returns a np array of shape (num_agents, 20, 7)
       """
       initial_ego_state = self.scenario.initial_ego_state
       angle = initial_ego_state.rear_axle.heading     
       ego_x, ego_y = initial_ego_state.rear_axle.x, initial_ego_state.rear_axle.y
       rot = -angle + np.pi / 2.0

       past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_past_tracked_objects(
                iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses
            )
        ]
       past_tracked_ego = self.scenario.get_ego_past_trajectory(     #read ego past
            iteration=0, 
            num_samples=self.num_past_poses, 
            time_horizon=self.past_time_horizon
        )
       past_timestamp = list(
            self.scenario.get_past_timestamps(
                iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
            )
       )

       sampled_past_observations = past_tracked_objects 
       agents_hist = sampled_tracked_objects_to_agents(sampled_past_observations, rot,ego_x,ego_y,past_timestamp)
       ego_hist = sampled_tracked_ego_to_tensor(past_tracked_ego,rot,ego_x,ego_y,isPast=True, past_timestamps=past_timestamp)
       agents_hist = self.sort_agents_hist_by_distance(agents_hist)
       if(len(agents_hist) > config.NUM_AGENTS):
           agents_hist = agents_hist[:config.NUM_AGENTS]    #[19,20,8]
       
       agents_list = torch.cat([ego_hist.unsqueeze(0),agents_hist],dim=0)     #concatenate ego and other agents to form a [20,20,8] array
       agents = np.array(agents_list,dtype=np.float32)
       return agents

    def collect_future(self):
       """
       collect ego's future trajectory, returns a np array of [num_future_poses, 2], in initial ego frame 
       """
       initial_ego_state = self.scenario.initial_ego_state   #absolute pose
       angle = initial_ego_state.rear_axle.heading
       ego_x, ego_y = initial_ego_state.rear_axle.x, initial_ego_state.rear_axle.y
       rot = -angle + np.pi / 2.0
       trajectory_absolute_states = self.scenario.get_ego_future_trajectory(
            iteration=0, 
            num_samples=self.num_future_poses, 
            time_horizon=self.future_time_horizon
        )

       ego_future = sampled_tracked_ego_to_tensor(trajectory_absolute_states,rot,ego_x,ego_y,isPast=False,past_timestamps= None)                                                     
       future = ego_future[:,:3]   #x,y,heading,no timestamp
       return np.array(future, dtype=np.float32)

    def build_mapping(self,debug=False):

       agents = self.collect_agents()
       matrix = self.get_map_features()
       labels = self.collect_future()
       
       mapping = {
           'agents': agents,
           'matrix': matrix,     
           'labels': labels,
           'token' : self.scenario.token,
           'map_api': self.map_api,
       }
       if debug:
           self.plot_scenario(mapping)
       return mapping
    

class NuplanDataset(torch.utils.data.Dataset):
    '''
    extract and dump scenarios
    '''
    def __init__(self,
                 config:config,
                 mode:str='train',
                 to_screen=True,
                 shuffle:bool = False,
                 reuse_cache: bool = True,
                 num_workers : int = 4,
                 val_ratio: float = 0.2):  
        os.makedirs(config.temp_file_dir,exist_ok=True)
        cache_file = os.path.join(config.temp_file_dir,f"ex_list_nuplan_{mode}.pkl")
        if reuse_cache and os.path.exists(cache_file):
            with open(cache_file,"rb") as f:
                self.ex_list = pickle.load(f)
            print(f"Load cache Nuplan files,{len(self.ex_list)} in total.")
            return
        scenario_mapping = ScenarioMapping(
            scenario_map=get_scenario_map(),
            subsample_ratio_override=0.5
        )
        print('Building scenarios...')
        builder = NuPlanScenarioBuilder(
            data_root=config.DATA_PATH,
            map_root=config.MAP_PATH,
            sensor_root=None,
            db_files=None,
            map_version=config.MAP_VERSION,
            scenario_mapping=scenario_mapping
        )
        print('Filtering scenario...')
        
        if mode == 'val':
            scenarios_per_type = max(1, int(config.SCENARIOS_PER_TYPE * val_ratio))
            limit_total_scenarios = None
        else:
            scenarios_per_type = config.SCENARIOS_PER_TYPE
            limit_total_scenarios = None
            
        scenario_filter = ScenarioFilter(
            *get_filter_parameters(
                scenarios_per_type,
                limit_total_scenarios,
                shuffle
            )
        )
        #Enable parallel process
        worker = SingleMachineParallelExecutor(use_process_pool=True)
        scenarios = builder.get_scenarios(scenario_filter,worker)
        print(f"Processing {len(scenarios)} scenarios for {mode} mode...")
        queue_in,queue_out = mp.Queue(num_workers),mp.Queue()
        def _worker(q_in, q_out):
            while True:
                sc = q_in.get()
                if sc is None: break
                try:
                    mapping = data_processor(sc).build_mapping()
                    q_out.put(compress(pickle.dumps(mapping)))
                except Exception as e:
                    print(f"[Warn] skip scenario {sc.token}:{e}")
        
        procs = [mp.Process(target=_worker,args=(queue_in,queue_out))
                 for _ in range(num_workers)]
        for p in procs : p.start()
        for sc in scenarios : queue_in.put(sc)
        for _ in procs : queue_in.put(None)    
        
        self.ex_list = []
        pbar = tqdm(total=len(scenarios))
        finished = 0
        processed_count = 0
        
        while finished < len(scenarios):
            try:
                item = queue_out.get(timeout=30) 
                finished += 1
                if item is not None:
                    self.ex_list.append(item)
                    processed_count += 1
                pbar.update(1)
            except Exception as e:
                print(f"Queue timeout or error: {e}")
                break
                
        pbar.close()
        
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join()
                
        print(f"Processed {processed_count} valid scenarios for {mode} mode, dumping...")
        with open(cache_file,"wb") as f:
            pickle.dump(self.ex_list,f)
        
    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):

        data_compress = self.ex_list[idx]
        while True:
            try:
                instance = pickle.loads(decompress(data_compress))
                if 'agents' in instance:
                    agents = instance['agents']
                    if np.isnan(agents).any() or np.isinf(agents).any():
                        print(f"Warning: NaN/Inf in agents data for sample {idx}")
                if 'matrix' in instance:
                    matrix = instance['matrix'] 
                    if np.isnan(matrix).any() or np.isinf(matrix).any():
                        print(f"Warning: NaN/Inf in matrix data for sample {idx}")
                if 'labels' in instance:
                    labels = instance['labels']
                    if np.isnan(labels).any() or np.isinf(labels).any():
                        print(f"Warning: NaN/Inf in labels data for sample {idx}")
                break
            except:
                # print(f"error {idx}")
                num = random.randint(0, idx)
                data_compress = self.ex_list[num]
        return instance


    

 
