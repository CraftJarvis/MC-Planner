import torch
import numpy as np
from itertools import accumulate

def negtive_sample(goals: torch.Tensor, num_cat: int):
    '''
    goals: B x T
    '''
    B, T = goals.shape
    goals = goals.long()
    dist = torch.zeros(B, num_cat).to(device=goals.device, dtype=torch.float32)
    dist = dist.scatter(1, goals[:, -1:], -torch.inf)
    dist = dist.softmax(-1)
    new_goals = torch.multinomial(dist, 1)
    new_goals = new_goals.repeat(1, T)
    return new_goals


class EvalMetric:

    def __init__(self, goal_names: list, max_ep_len=600):
        self.goal_names = goal_names
        self.max_ep_len = max_ep_len
        self.num_goal = len(goal_names)
        self.reset()

    def reset(self):
        self.acquires = {goal: [] for goal in self.goal_names}
        self.nb_eps = {}

    def add(self, goal_name, acquire: list):
        self.acquires[goal_name].append(acquire)
        self.nb_eps[goal_name] = self.nb_eps.get(goal_name, 0) + 1

    def precision(self, k = 1):
        pos = {}
        tot = {}
        suc = {}
        hor = {}
        suc_per_step = {}
        for i in range(self.num_goal):
            goal = self.goal_names[i]
            if goal not in suc_per_step:
                suc_per_step[goal] = [0] * (self.max_ep_len + 10)
            for eps in self.acquires[goal]:
                for item, t in eps:
                    if item == goal:
                        suc_per_step[goal][t] = 1
                        break
            for eps in self.acquires[goal]:
                acq = eps[:k]
                pos[goal] = pos.get(goal, 0) + sum([a[0] == goal for a in acq])
                tot[goal] = tot.get(goal, 0) + len(acq)
                #! compute the first achieved horizon
                f_eps = list(filter(lambda x : x[0] == goal, eps))
                if len(f_eps) > 0:
                    hor[goal] =  hor.get(goal, []) + [f_eps[0][1]]
                    suc[goal] = suc.get(goal, 0) + 1
        
        res = {}
        for goal in self.goal_names:
            
            suc_per_step[goal] = np.array(list(accumulate(suc_per_step[goal]))) / self.nb_eps[goal]
            
            res[goal] = {
                'precision': (pos[goal] / tot[goal]) if tot[goal] != 0 else 0,
                'pos': pos[goal], 
                'neg': tot[goal] - pos[goal],
                'hor': sum(hor.get(goal, [])) / (len(hor.get(goal, [])) + 1e-6),
                'tot': tot[goal],
                'success': suc.get(goal, 0) / len(self.acquires[goal]) if len(self.acquires[goal]) != 0 else 0,
                'suc_per_step': suc_per_step[goal],
            }
            
        return res


if __name__ == '__main__':
    goals = torch.Tensor([
        [0, 0, 0, 1, 1, 1],
        [0, 0, 2, 2, 2, 2],
        [0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
    ])
    new_goals = negtive_sample(goals, 3)
    print(new_goals)