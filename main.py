'''
@zihao:  
'''

from controller import *
from planner import *
from selector import *

from mineclip import MineCLIP
from transformers import CLIPProcessor, CLIPModel
from omegaconf import OmegaConf

import os
import json
import random
from datetime import datetime 
import time 

from typing import List, Dict, Tuple

from PIL import Image, ImageDraw
import cv2

import warnings
warnings.filterwarnings('ignore')

def resize_image_numpy(img, target_resolution = (128, 128)):
    img = cv2.resize(img, dsize=target_resolution, interpolation=cv2.INTER_LINEAR)
    return img

prefix = os.getcwd()
goal_mapping_json = os.path.join(prefix, "data/goal_mapping.json")
task_info_json = os.path.join(prefix, "data/task_info.json")
goal_lib_json = os.path.join(prefix, "data/goal_lib.json")
logging_folder = ""


# env_name = "crafting"
# task = "obtain_wooden_slab"      

class Evaluator:
    def __init__(self, cfg):
        device = torch.device("cuda", 0)
        self.device = device
        self.cfg = cfg
        # super().__init__(cfg, device=device, only_base=True)
        self.num_workers = 0
        self.env = MineDojoEnv(
                name=cfg['eval']['env_name'], 
                img_size=(cfg['simulator']['resolution'][0], cfg['simulator']['resolution'][1]),
                rgb_only=False,
            )
        log_file = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.task = cfg['eval']['task_name']
        self.task_obj, self.max_ep_len, self.task_question = self.load_task_info(self.task)
        
        self.use_ranking_goal = cfg["goal_model"]["use_ranking_goal"]
        
        self.goal_mapping_cfg = self.load_goal_mapping_config()
        self.mineclip_prompt_dict = self.goal_mapping_cfg["mineclip"]
        self.clip_prompt_dict = self.goal_mapping_cfg["clip"] # unify the mineclip and clip 
        self.goal_mapping_dct = self.goal_mapping_cfg["horizon"]

        print("[Progress] [red]Computing goal embeddings using MineClip's text encoder...")
        rely_goals = [val for val in self.goal_mapping_dct.values()]
        self.embedding_dict = accquire_goal_embeddings(cfg['pretrains']['clip_path'], rely_goals)
        
        self.goal_model_freq = cfg["goal_model"]["freq"]
        self.goal_list_size = cfg["goal_model"]["queue_size"]

        self.record_frames = cfg["record"]["frames"]
        
        self.mine_agent = MineAgent(cfg, device).model
        self.mine_wrapper = MineAgentWrapper(self.env, self.mine_agent, max_ranking=15)
        self.craft_agent = CraftAgent(self.env)

        self.planner = Planner()

        self.selector = Selector()

        plan = self.planner.initial_planning(self.task_question)
        self.goal_list = self.planner.generate_goal_list(plan)
        print(self.goal_list)
        self.curr_goal = self.goal_list[0]
        self.goal_eps = 0

    def load_task_info(self, task):
        with open(task_info_json, 'r') as f:
            task_info = json.load(f)
        target_item = task_info[task]['object']
        episode_length = task_info[task]["episode"]
        task_question = task_info[task]['question']
        return target_item, episode_length, task_question

    def load_goal_mapping_config(self):
        with open(goal_mapping_json, "r") as f:
            goal_mapping_dct = json.load(f)
        return goal_mapping_dct 

    # check if the inventory has the object items
    def check_inventory(self, inventory, items:dict): # items: {"planks": 4, "stick": 2}
        for key in items.keys(): # check every object item 
            # item_flag = False
            if sum([item['quantity'] for item in inventory if item['name'] == key]) < items[key]:
                return False
        return True
    
    def check_precondition(self, inventory, precondition:dict): 
        for key in precondition.keys(): # check every object item 
            # item_flag = False
            if sum([item['quantity'] for item in inventory if item['name'] == key]) < precondition[key]:
                return False
        return True
    
    def check_done(self, inventory, task_obj:str):
        for item in inventory:
            if task_obj == item['name']:
                return True
        return False

    def update_goal(self, inventory):
        while self.check_inventory(inventory, self.curr_goal["object"]):
            print(f"[INFO]: finish goal {self.curr_goal['name']}.")
            self.planner.generate_success_description(self.curr_goal["ranking"])
            self.goal_list.remove(self.goal_list[0])
            self.curr_goal = self.goal_list[0]
            self.goal_eps = 0
        
        if self.curr_goal["type"]== 'mine' and not self.check_precondition(inventory, self.curr_goal["precondition"]):
            self.goal_eps = 3000

    def replan_task(self, inventory, task_question):
        self.planner.generate_inventory_description(inventory)
        self.planner.generate_failure_description(self.curr_goal['ranking'])
        self.planner.generate_explanation()
        plan = self.planner.replan(task_question)
        
        self.goal_list = self.planner.generate_goal_list(plan)
        self.curr_goal = self.goal_list[0]
        self.goal_eps = 0 


    @torch.no_grad()
    def eval_step(self, fps=100):
        
        self.mine_agent.eval()

        obs = self.env.reset() 

        # target_item = self.mapping_goal[goal]
        print(f"[INFO]: Evaluating the task is ", self.task)
        
        if self.record_frames:
            video_frames = [obs['rgb']]
            goal_frames = ["start"] 
        
        def preprocess_obs(obs: dict):
            res_obs = {}
            rgb = torch.from_numpy(obs['rgb']).unsqueeze(0).to(device=self.device, dtype=torch.float32).permute(0, 3, 1, 2)
            res_obs['rgb'] = resize_image(rgb, target_resolution=(120, 160))
            res_obs['voxels'] = torch.from_numpy(obs['voxels']).reshape(-1).unsqueeze(0).to(device=self.device, dtype=torch.long)
            res_obs['compass'] = torch.from_numpy(obs['compass']).unsqueeze(0).to(device=self.device, dtype=torch.float32)
            res_obs['gps'] = torch.from_numpy(obs['gps'] / np.array([1000., 100., 1000.])).unsqueeze(0).to(device=self.device, dtype=torch.float32)
            res_obs['biome'] = torch.from_numpy(obs['biome_id']).unsqueeze(0).to(device=self.device, dtype=torch.long)
            return res_obs

        def stack_obs(prev_obs: dict, obs: dict):
            stacked_obs = {}
            stacked_obs['rgb'] = torch.cat([prev_obs['rgb'], obs['rgb']], dim = 0)
            stacked_obs['voxels'] = torch.cat([prev_obs['voxels'], obs['voxels']], dim = 0)
            stacked_obs['compass'] = torch.cat([prev_obs['compass'], obs['compass']], dim = 0)
            stacked_obs['gps'] = torch.cat([prev_obs['gps'], obs['gps']], dim = 0)
            stacked_obs['biome'] = torch.cat([prev_obs['biome'], obs['biome']], dim = 0)
            return stacked_obs

        def slice_obs(obs: dict, slice: torch.tensor):
            res = {}
            for k, v in obs.items():
                res[k] = v[slice]
            return res

        def add_obs(video, image):
            video = np.concatenate((video, image.reshape(1, 1, 3, 160, 256)), axis = 1)
            if video.shape[1] > self.clip_frames:
                video = video[:,1:,:,:,:]
            return video
        
        obs = preprocess_obs(obs)

        states = obs
        actions = torch.zeros(1, self.mine_agent.action_dim, device=self.device)
        timesteps = torch.tensor([0], device=self.device, dtype=torch.long)

        acquire = []
        curr_goal = None
        prev_goal = None
        seek_point = 0
        history_gps = []
        
        obs, reward, env_done, info = self.env.step(self.env.action_space.no_op())
        init_deaths = info['stat']['deaths']
        
        # max_ep_len = task_eps[self.task]
        for t in range(0, self.max_ep_len):
            time.sleep(1/fps)
            
            sf = 5 # self.cfg['data']['skip_frame']
            wl = 16 # self.cfg['data']['window_len']
            
            # extract frame by every <skip_frame> frames
            end = actions.shape[0] - 1
            rg = torch.arange(end, min(max(end-sf*(wl-1)-1, seek_point-1), end-1), -sf).flip(0)

            self.update_goal(info['inventory'])
            # print(self.curr_goal)

            # take the current goal type
            curr_goal_type = self.curr_goal["type"]

            if not prev_goal == curr_goal:
                print(f"[INFO]: Episode Step {t}, Current Goal {curr_goal}")
            prev_goal = curr_goal
            # choose for actions 

            # DONE: change the craft agent into craft actions
            if curr_goal_type in ['craft', 'smelt']:
                action_done = False
                curr_goal = self.curr_goal['name']
                preconditions = self.curr_goal["precondition"].keys()
                goal = list(self.curr_goal['object'].keys())[0]
                curr_actions, action_done = self.craft_agent.get_action(preconditions, curr_goal_type, goal)

            elif curr_goal_type == "mine":
                action_done = True
                # key = self.candidate_goal_list[0]
                key = self.curr_goal['name']
                goal = self.goal_mapping_dct[list(self.curr_goal["object"].keys())[0]]
                goal_embedding = self.embedding_dict[goal]
                goals = torch.from_numpy(goal_embedding).to(self.device).repeat(len(rg), 1)
                complete_states = slice_obs(states, rg)
                complete_states['prev_action'] = actions[rg]
                _ranking, _action = self.mine_wrapper.get_action(goal, goals, complete_states)
                
                curr_goal = key
                curr_actions = _action
            else:
                print("Undefined action type !!")
            
            if self.curr_goal['precondition'] is not None:
                for cond in self.curr_goal['precondition'].keys():
                    if cond not in ['wooden_pickaxe', 'stone_pickaxe', 'iron_pickaxe', "diamond_pickaxe", 
                                "wooden_axe", "stone_axe", "iron_axe", "diamond_axe"]:
                        continue
                    if info['inventory'][0]['name'] != cond:
                        for item in info['inventory']:
                            if item['name'] == cond and item['quantity'] > 0 and item['index'] > 0:
                                act = self.env.action_space.no_op()
                                act[5] = 5
                                act[7] = item['index']
                                self.env.step(act)
                                break
            #! indent change
            action = curr_actions
            if hasattr(action, 'is_cuda') and action.is_cuda:
                action = action.cpu().numpy()
            obs, reward, env_done, info = self.env.step(action)
            
            # history_gps.append(obs['gps'])
            # if curr_goal != prev_goal:
            #     print(f"Iteration {iter_num} | Step {t} - current goal is {curr_goal}")

            if self.record_frames:
                video_frames.append(obs['rgb'])
                goal_frames.append(curr_goal)
            obs = preprocess_obs(obs)

            if type(action) != torch.Tensor:
                action = torch.from_numpy(action)
            if action.device != self.device:
                action = action.to(self.device)

            states = stack_obs(states, obs)
            actions = torch.cat([actions, action.unsqueeze(0)], dim = 0)
            timesteps = torch.cat([timesteps, torch.tensor([t], device=self.device, dtype=torch.long)], dim=0)

            self.goal_eps += 1
            if curr_goal_type == 'mine' and self.goal_eps > 2000:
                self.replan_task(info["inventory"], self.task_question)
            elif curr_goal_type == 'craft' and self.goal_eps > 100:
                self.replan_task(info["inventory"], self.task_question)
            elif curr_goal_type == 'smelt' and self.goal_eps > 100:
                self.replan_task(info["inventory"], self.task_question)

            
            if len(history_gps) > 1000 and (history_gps[-1] == history_gps[-1000]).all():
                break
            
            if self.check_done(info['inventory'], self.task_obj):  # check if the task is done?
                env_done = True
                print(f"[INFO]: finish goal {self.curr_goal['name']}.")
                self.planner.generate_success_description(self.curr_goal["ranking"])
                break

        # record the video
        if self.record_frames:
            print("[INFO]: saving the frames")
            imgs = []
            for id, frame in enumerate(video_frames):
                frame = resize_image_numpy(frame, (320,240)).astype('uint8')
                cv2.putText(
                    frame,
                    f"FID: {id}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Goal: {goal_frames[id]}",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255), 
                    2,
                )
                imgs.append(Image.fromarray(frame))
            imgs = imgs[::3]
            print(f"record imgs length: {len(imgs)}")
            now = datetime.now()
            timestamp = f"{now.hour}_{now.minute}_{now.second}"
            file_name = os.path.join(prefix, "recordings/"+timestamp)
            imgs[0].save(file_name + ".gif", save_all=True, append_images=imgs[1:], optimize=False, quality=0, duration=150, loop=0)
        
        return env_done, t # True or False, episode length

    def single_task_evaluate(self):
        if self.num_workers == 0:
            succ_flag, min_episode = self.eval_step()
            if succ_flag: 
                print(f'Succeed on {min_episode} step.')
            else:
                print("Failed.")
    
    # TODO: add multi-task evaluation


@hydra.main(config_path="configs", config_name="defaults")
def main(cfg):
    print(cfg)
    evaluator = Evaluator(cfg) 
    evaluator.single_task_evaluate()


if __name__ == '__main__':
    main()
