import cv2
import os
import time
import gym
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import argparse
import multiprocessing as mp
import hydra
import pickle
import random
import sys
from copy import deepcopy
from functools import partial
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import torch.distributed as dist
from datetime import datetime
from hydra.utils import get_original_cwd, to_absolute_path
from pathlib import Path
from rich import print
from tqdm import tqdm
from functools import partial
import minedojo
from minedojo.sim.mc_meta import mc as MC
from typing import List, Dict, Tuple
from itertools import chain
from ray.rllib.models.torch.torch_action_dist import TorchMultiCategorical
from minedojo.minedojo_wrapper import MineDojoEnv
from src.models.simple import SimpleNetwork
from src.utils.vision import create_backbone, resize_image

class CraftAgent: 
    '''
    Craft agent based on 'craft' action space.
    '''
    def __init__(self, env):
        self.env = env
        self.craft_smelt_items = MC.ALL_CRAFT_SMELT_ITEMS
        self.history = {
            'craft_w_table': None, 
            'craft_wo_table': None,
            'smelt_w_furnace': None,
        }
    
    def no_op(self, times = 20):
        for i in range(times):
            act = self.env.action_space.no_op()
            yield act
    
    def take_forward(self, times=3):
        for _ in range(times):
            yield self.env.action_space.no_op()
            
    def index_slot(self, goal):
        #! accquire info 
        obs, reward, done, info = self.env.step(self.env.action_space.no_op())
        slot = -1
        for item in info['inventory']:
            if goal == item['name']:
                slot = item['index']
                break
        return slot

    def equip(self, goal):
        obs, reward, done, info = self.accquire_info()
        for item in info['inventory']:
            if item['name'] == goal and item['index'] > 0:
                act = self.env.action_space.no_op()
                act[5] = 5
                act[7] = item['index']
                yield act
                return 

    def pillar_jump(self, stepping_stone="cobblestone"):
        for act in chain(
            self.look_to(-85), 
            self.attack(40),
            self.place_down(stepping_stone),
            self.place_down(stepping_stone),
            self.place_down(stepping_stone),
        ):
            yield act

    def go_surface(self):
        while True:
            obs, reward, done, info = self.env.step(self.env.action_space.no_op())
            if info['can_see_sky']:
                return 
            candidates = ['dirt', 'stone', 'cobblestone']
            insufficient = True
            for stepping_stone in candidates:
                quantity = sum([item['quantity'] for item in info['inventory'] if item['name'] == stepping_stone])
                if quantity >= 1:
                    insufficient = False
                    for act in self.pillar_jump(stepping_stone):
                        yield act
                    break
            if insufficient:
                return
        
    def accquire_info(self):
        return self.env.step(self.env.action_space.no_op())

    def use(self):
        act = self.env.action_space.no_op()
        act[5] = 1
        yield act
        yield self.env.action_space.no_op()

    def look_to(self, deg = 0):
        #! accquire info 
        obs, reward, done, info = self.accquire_info()
        while obs['compass'][1] < deg:
            act = self.env.action_space.no_op()
            act[3] = 10
            act[5] = 3
            yield act
            obs, reward, done, info = self.accquire_info()
        while obs['compass'][1] > deg:
            act = self.env.action_space.no_op()
            act[5] = 3
            act[3] = 0
            yield act
            obs, reward, done, info = self.accquire_info()

    def jump(self):
        act = self.env.action_space.no_op()
        act[2] = 1
        yield act
        yield self.env.action_space.no_op()

    def place(self, goal):
        slot = self.index_slot(goal)
        if slot == -1:
            return False
        act = self.env.action_space.no_op()
        act[5] = 6
        act[7] = slot
        yield act

    def place_down(self, goal):
        if self.index_slot(goal) == -1:
            return None
        for act in chain(
            self.look_to(deg=87),
            self.attack(2),
            self.jump(),
            self.place(goal),
            self.use(),
        ):
            yield act

    def attack(self, times = 20):
        for i in range(times):
            act = self.env.action_space.no_op()
            act[5] = 3
            yield act
        yield self.env.action_space.no_op()

    def recycle(self, goal, times = 20):
        for i in range(times):
            act = self.env.action_space.no_op()
            act[5] = 3
            obs, reward, done, info = self.env.step(act)
            if any([item['name'] == goal for item in info['inventory']]):
                break
        yield self.env.action_space.no_op()
        for act in chain(
            self.look_to(0),
            self.take_forward(3),
        ):
            yield act

    def craft_wo_table(self, goal):
        act = self.env.action_space.no_op()
        act[5] = 4
        act[6] = self.craft_smelt_items.index(goal)
        yield act
    
    def forward(self, times=5):
        for i in range(times):
            act = self.env.action_space.no_op()
            act[0] = 1
            yield act
        
    def craft_w_table(self, goal):
        if self.index_slot('crafting_table') == -1:
            return None
        for act in chain(
            self.forward(5),
            self.look_to(-87), 
            self.attack(40),
            self.place_down('crafting_table'),
            self.craft_wo_table(goal),
            self.recycle('crafting_table', 200),
        ):
            # print(f"{goal}: {act}")
            yield act

    def smelt_w_furnace(self, goal):
        if self.index_slot('furnace') == -1:
            return None
        for act in chain(
            self.look_to(-87), 
            self.attack(40),
            self.place_down('furnace'),
            self.craft_wo_table(goal),
            self.recycle('furnace', 200),
        ):
            yield act

    def smelt_wo_furnace(self, goal):
        for act in self.craft_wo_table(goal):
            yield act

    def get_action(self, preconditions, goal_type, goal):
        if goal_type == 'craft':
            use_crafting_table = ('crafting_table' in preconditions)
            if use_crafting_table:
                if self.history['craft_w_table'] is None:
                    self.history['craft_w_table'] = self.craft_w_table(goal)
                try:
                    act = next(self.history['craft_w_table'])
                    return act, False
                except:
                    self.history['craft_w_table'] = None
                    return self.env.action_space.no_op(), True
            else:
                if self.history['craft_wo_table'] is None:
                    self.history['craft_wo_table'] = self.craft_wo_table(goal)
                try:
                    act = next(self.history['craft_wo_table'])
                    return act, False
                except:
                    self.history['craft_wo_table'] = None
                    return self.env.action_space.no_op(), True
        elif goal_type == 'smelt':
                if self.history['smelt_w_furnace'] is None:
                    self.history['smelt_w_furnace'] = self.smelt_w_furnace(goal)
                try:
                    act = next(self.history['smelt_w_furnace'])
                    return act, False
                except:
                    self.history['smelt_w_furnace'] = None
                    return self.env.action_space.no_op(), True

torch.backends.cudnn.benchmark = True

def making_exp_name(cfg):
    component = []
    if cfg['model']['use_horizon']:
        component.append('p:ho')
    else:
        component.append('p:bc')
    
    component.append("b:" + cfg['model']['backbone_name'][:4])
    
    today = datetime.now()
    
    component.append(f"{today.month}-{today.day}#{today.hour}-{today.minute}")
    
    return "@".join(component)

from ray.rllib.models.torch.mineclip_lib.mineclip_model import MineCLIP
def accquire_goal_embeddings(clip_path, goal_list, device="cuda"):
    clip_cfg = {'arch': 'vit_base_p16_fz.v2.t2', 'hidden_dim': 512, 'image_feature_dim': 512, 'mlp_adapter_spec': 'v0-2.t0', 
               'pool_type': 'attn.d2.nh8.glusw', 'resolution': [160, 256]}
    clip_model = MineCLIP(**clip_cfg)
    clip_model.load_ckpt(clip_path, strict=True)
    clip_model = clip_model.to(device)
    res = {}
    with torch.no_grad():
        for goal in goal_list:
            res[goal] = clip_model.encode_text([goal]).cpu().numpy()
    return res

class MineAgentWrapper:
    '''
    Shell agent for goal: mine_cobblestone, mine_stone, mine_coal, mine_iron_ore, mine_diamond
    '''
    script_goals = ['cobblestone', 'stone', 'coal', 'iron_ore', 'diamond']
    
    def __init__(self, env, mine_agent, max_ranking: int=15) -> None:
        self.env = env
        self.mine_agent = mine_agent
        self.max_ranking = max_ranking
    
    def get_action(self, goal: str, goals: torch.Tensor, states: dict) -> Tuple[int, torch.Tensor]:
        if goal in MineAgentWrapper.script_goals:
            act = self.env.action_space.no_op()
            if random.randint(0, 20) == 0:
                act[4] = 1
            if random.randint(0, 20) == 0:
                act[0] = 1
            if goal in ['stone', 'coal', 'cobblestone']:
                if states['compass'][-1][1] < 83:
                    act[3] = 9
                    return self.max_ranking, act
                else:
                    act[5] = 3
                    return self.max_ranking, act
            elif goal in ['iron_ore', 'diamond']:
                if goal == 'iron_ore':
                    depth = 30
                elif goal == 'diamond':
                    depth = 10
                if states['gps'][-1][1] * 100 > depth:
                    if states['compass'][-1][1] < 80:
                        act[3] = 9
                        return self.max_ranking, act
                    else:
                        act[5] = 3
                        return self.max_ranking, act
                else:
                    if states['compass'][-1][1] > 50:
                        act[3] = 1
                        return self.max_ranking, act
                    elif states['compass'][-1][1] < 40:
                        act[3] = 9
                        return self.max_ranking, act
                    else:
                        act[0] = 1
                        act[5] = 3
                        return self.max_ranking, act
            else:
                raise NotImplementedError
        else:
            # Neural Network Agent
            action_preds, mid_info = self.mine_agent.get_action(
                goals=goals,
                states=states, 
                horizons=None,
            )
            action_dist = TorchMultiCategorical(action_preds[:, -1],  None, self.mine_agent.action_space)
            action = action_dist.sample().squeeze(0)
            goal_ranking = mid_info['pred_horizons'][0, -1].argmax(-1)
            return goal_ranking, action

class MineAgent:
    def __init__(self, cfg, device, local_rank=0, only_base=False):
        self.action_space = [3, 3, 4, 11, 11, 8, 1, 1]
        self.cfg = cfg
        self.device = device
        self.local_rank = local_rank
        self.exp_name = making_exp_name(cfg)

        #! accquire goal embeddings
        print("[Progress] [red]Computing goal embeddings using MineClip's text encoder...")
        self.embedding_dict = accquire_goal_embeddings(cfg['pretrains']['clip_path'], cfg['data']['filters'])
        
        backbone = create_backbone(
            name=cfg['model']['backbone_name'], 
            model_path=cfg['model']['model_path'], 
            weight_path=cfg['model']['weight_path'],
            goal_dim=cfg['model']['embed_dim'],
        )
        
        if cfg['model']['name'] == 'simple':
            self.model = SimpleNetwork(
                action_space=self.action_space,
                state_dim=cfg['model']['state_dim'],
                goal_dim=cfg['model']['goal_dim'],
                action_dim=cfg['model']['action_dim'],
                num_cat=len(cfg['data']['filters']),
                hidden_size=cfg['model']['embed_dim'],
                fusion_type=cfg['model']['fusion_type'],
                max_ep_len=cfg['model']['max_ep_len'],
                backbone=backbone,
                frozen_cnn=cfg['model']['frozen_cnn'],
                use_recurrent=cfg['model']['use_recurrent'],
                use_extra_obs=cfg['model']['use_extra_obs'],
                use_horizon=cfg['model']['use_horizon'],
                use_prev_action=cfg['model']['use_prev_action'],
                extra_obs_cfg=cfg['model']['extra_obs_cfg'],
                use_pred_horizon=cfg['model']['use_pred_horizon'],
                c=cfg['model']['c'],
                transformer_cfg=cfg['model']['transformer_cfg']
            )
        else:
            raise NotImplementedError
        
        # self.iter_num = -1
        
        if cfg['model']['load_ckpt_path'] != "":
            state_dict = torch.load(cfg['model']['load_ckpt_path'])
            print(f"[MAIN] load checkpoint from {cfg['model']['load_ckpt_path']}. ")
            # print(f"[MAIN] iter_num: {state_dict['iter_num']}, loss: {state_dict['loss']}")
            if cfg['model']['only_load_cnn']:
                backbone_state_dict = self.model.state_dict()
                backbone_state_dict.update({
                    k: v for k, v in state_dict['model_state_dict'].items() if 'backbone' in k
                })
                self.model.load_state_dict(backbone_state_dict)
            else:
                self.model.load_state_dict(state_dict['model_state_dict'])
                self.iter_num = state_dict['iter_num']
        
        self.model = self.model.to(self.device)
    


