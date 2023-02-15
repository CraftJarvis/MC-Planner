import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from rich import print
from src.utils.mlp import build_mlp

def discrete_horizon(horizon):
    '''
    0 - 10: 0
    10 - 20: 1
    20 - 30: 2
    30 - 40: 3
    40 - 50: 4
    50 - 60: 5
    60 - 70: 6
    70 - 80: 7
    80 - 90: 8
    90 - 100: 9
    100 - 120: 10
    120 - 140: 11
    140 - 160: 12
    160 - 180: 13
    180 - 200: 14
    200 - ...: 15
    '''
    # horizon_list = [0]*25 + [1]*25 + [2]*25 + [3]*25 +[4]* 50 + [5]*50 + [6] * 700
    horizon_list = []
    for i in range(10):
        horizon_list += [i] * 10
    for i in range(10, 15):
        horizon_list += [i] * 20
    horizon_list += [15] * 700
    if type(horizon) == torch.Tensor:
        return torch.Tensor(horizon_list, device=horizon.device)[horizon]
    elif type(horizon) == np.ndarray:
        return np.array(horizon_list)[horizon]
    elif type(horizon) == int:
        return horizon_list[horizon]
    else:
        assert False

class Concat(nn.Module):
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(2*input_dim, output_dim)
    
    def forward(self, model_x, model_y):
        return self.fc(torch.cat([model_x, model_y], dim=-1))

class Bilinear(nn.Module):
    
    def __init__(self, input_dim: int, output_dim:int):
        super().__init__()
        self.U = nn.Linear(input_dim, output_dim)
        self.V = nn.Linear(input_dim, output_dim)
        self.P = nn.Linear(input_dim, output_dim)
    
    def forward(self, model_x, model_y):
        return self.P(torch.tanh(self.U(model_x) * self.V(model_y)))


class FiLM(nn.Module):
    def __init__(self, input_dim: int, output: int):
        super().__init__()
        self.U = nn.Linear(input_dim, output_dim)
        self.V = nn.Linear(input_dim, output_dim)
    
    def forward(self, model_x, model_y):
        return self.U(model_y) * model_x + self.V(model_y)


class PrevActionEmbedding(nn.Module):
    
    def __init__(self, output_dim: int, action_space):
        super().__init__()
        self.output_dim = output_dim
        self.action_space = action_space
        embed_dim = output_dim // len(action_space)
        self._embed = nn.ModuleList([nn.Embedding(voc_size, embed_dim) for voc_size in self.action_space])
        self._fc = nn.Linear(len(self.action_space) * embed_dim, output_dim)
    
    def forward(self, prev_action):
        categorical = prev_action.shape[-1]
        action_list = []
        for i in range(categorical):
            action_list.append(self._embed[i](prev_action[..., i].long()))
        output = self._fc(torch.cat(action_list, dim=-1))
        return output


class ExtraObsEmbedding(nn.Module):
    
    def __init__(self, embed_dims: dict, output_dim: int):
        super().__init__()
        self.embed_dims = embed_dims 
        self.embed_biome = nn.Embedding(168, embed_dims['biome_hiddim'])
        self.embed_compass = build_mlp(
            input_dim=2,
            hidden_dim=embed_dims['compass_hiddim'],
            output_dim=embed_dims['compass_hiddim'],
            hidden_depth=2,
        )
        self.embed_gps = build_mlp(
            input_dim=3,
            hidden_dim=embed_dims['gps_hiddim'],
            output_dim=embed_dims['gps_hiddim'],
            hidden_depth=2,
        )
        self.embed_voxels = nn.Embedding(32,  embed_dims['voxels_hiddim']//4)
        self.embed_voxels_last = build_mlp(
            input_dim=12*embed_dims['voxels_hiddim']//4,
            hidden_dim=embed_dims['voxels_hiddim'],
            output_dim=embed_dims['voxels_hiddim'],
            hidden_depth=2,
        )
        sum_dims = sum(v for v in embed_dims.values())
        self.fusion = build_mlp(
            input_dim=sum_dims,
            hidden_dim=sum_dims//2,
            output_dim=output_dim,
            hidden_depth=2,
        )
    
    def forward(self, obs: dict):
        
        biome = obs['biome']
        compass = obs['compass']
        gps = obs['gps']
        voxels = obs['voxels']
        
        with_time_dimension = len(biome.shape) == 2
        
        if with_time_dimension:
            B, T = biome.shape
            biome = biome.view(B*T, *biome.shape[2:])
            compass = compass.view(B*T, *compass.shape[2:])
            gps = gps.view(B*T, *gps.shape[2:])
            voxels = voxels.view(B*T, *voxels.shape[2:])
        
        biome = self.embed_biome(biome)
        compass = self.embed_compass(compass)
        gps = self.embed_gps(gps)
        voxels = self.embed_voxels(voxels)
        voxels = self.embed_voxels_last(voxels.view(voxels.shape[0], -1))
        
        output = self.fusion(torch.cat([biome, compass, gps, voxels], dim = -1))
        
        if with_time_dimension:
            output = output.view(B, T, *output.shape[1:])

        return output