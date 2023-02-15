import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_action_dist import TorchMultiCategorical

def horizon_loss(params: dict):
    
    mid_info = params['mid_info']
    pred_horizons = mid_info['pred_horizons']
    horizons = params['horizons']
    attention_mask = params['attention_mask']
    # print(horizons, mid_horizons)
    # import ipdb; ipdb.set_trace()
    horizons = horizons.reshape(-1)[attention_mask.reshape(-1) > 0]
    pred_horizons = pred_horizons.reshape(-1, pred_horizons.shape[-1])[attention_mask.reshape(-1) > 0]
    fn = nn.CrossEntropyLoss()
    loss = fn(pred_horizons, horizons)
    return loss

def action_loss(params: dict):
    
    # prepare parameters
    actions = params['actions']
    horizons = params['horizons']
    state_preds = params['state_preds']
    action_preds = params['action_preds']
    horizon_preds = params['horizon_preds']
    attention_mask = params['attention_mask']
    action_space = params['action_space']
    gamma = params['gamma']
    
    action_dim = action_preds.shape[2]
    action_target = torch.clone(actions)
    act_label_dim = action_target.shape[2]
    action_preds = action_preds.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]
    masked_horizons = horizons.reshape(-1)[attention_mask.reshape(-1) > 0]
    action_target = action_target.reshape(-1, act_label_dim)[attention_mask.reshape(-1) > 0]

    def fn(action_preds, action_target, masked_horizons, action_space, gamma):
        weight = gamma ** masked_horizons
        action_dist = TorchMultiCategorical(action_preds,  None, action_space)
        neg_logp = -action_dist.logp(action_target.to(action_preds.device))
        loss = torch.mean(neg_logp * weight)
        return loss

    return fn(action_preds, action_target, masked_horizons, action_space, gamma)

def state_loss(params: dict):

    # prepare parameters
    states = params['states']
    state_preds = params['state_preds']
    horizons = params['horizons']
    horizon_preds = params['horizon_preds']
    attention_mask = params['attention_mask']
    gamma = params['gamma']
    
    attention_mask = attention_mask[:, :-1]
    state_target = torch.clone(states)[:, 1:, :].reshape(-1, states.shape[-1])
    state_preds = state_preds[:, :-1, :].reshape(-1, state_preds.shape[-1])
    state_target = state_target[attention_mask.reshape(-1) > 0]
    state_preds = state_preds[attention_mask.reshape(-1) > 0]
    masked_horizons = horizons[:, :-1, :].reshape(-1)[attention_mask.reshape(-1) > 0]
    
    def fn(state_preds, state_target, masked_horizons, gamma = 1.):
        return F.mse_loss(state_preds, state_target)

    return fn(state_preds, state_target, masked_horizons, gamma)

def goal_loss(params: dict):
    
    # prepare parameters
    goals = params['goals']
    goal_preds = params['goal_preds']
    horizons = params['horizons']
    attention_mask = params['attention_mask']
    gamma = params['gamma']
    
    # import ipdb; ipdb.set_trace()
    masked_horizons = horizons.reshape(-1)[attention_mask.reshape(-1) > 0]

    goal_pred_dim = goal_preds.shape[-1]
    goal_targets = torch.clone(goals).reshape(-1)[attention_mask.reshape(-1) > 0]
    goal_preds = goal_preds.reshape(-1, goal_pred_dim)[attention_mask.reshape(-1) > 0]
    
    loss = torch.nn.CrossEntropyLoss(reduction='none')(goal_preds, goal_targets.long())
    loss *= gamma ** masked_horizons
    return loss.mean()

def kl_loss(params: dict):
    action_preds = params['action_preds']
    neg_action_preds = params['neg_action_preds']
    horizons = params['horizons']
    attention_mask = params['attention_mask']
    gamma = params['gamma']
    masked_horizons = horizons.reshape(-1)[attention_mask.reshape(-1) > 0]
    action_dim = action_preds.shape[2]
    action_preds = action_preds.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]
    neg_action_preds = neg_action_preds.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]
    action_space = [3, 3, 4, 11, 11, 8, 1, 1]
    loss = []
    for i in range(len(action_space)):
        s = sum(action_space[:i])
        e = s + action_space[i]
        pos = action_preds[:, s:e]
        neg = neg_action_preds[:, s:e]
        this_loss = nn.KLDivLoss(reduction="none")(F.log_softmax(neg, dim = -1), F.softmax(pos, dim = -1))
        this_loss += nn.KLDivLoss(reduction="none")(F.log_softmax(pos, dim = -1), F.softmax(neg, dim = -1))
        this_loss = this_loss.sum(dim=-1) * (gamma ** masked_horizons) / 2
        loss += [this_loss.mean()]
        
    return - sum(loss) / len(loss)

def cl_loss(params: dict, tau = 1.0):
    action_g = params['actions'].long()
    action_p = params['action_preds']
    action_n = params['neg_action_preds']
    attention_mask = params['attention_mask']
    action_g = action_g.reshape(-1, action_g.shape[-1])[attention_mask.reshape(-1) > 0]
    action_p = action_p.reshape(-1, action_p.shape[-1])[attention_mask.reshape(-1) > 0]
    action_n = action_n.reshape(-1, action_n.shape[-1])[attention_mask.reshape(-1) > 0]

    action_space = [3, 3, 4, 11, 11, 8, 1, 1]
    loss = []
    for i in range(len(action_space)):
        s = sum(action_space[:i])
        e = s + action_space[i]
        
        _a_g = action_g[:, i]
        _a_p = action_p[:, s:e]
        _a_n = action_n[:, s:e]
        
        ce_gp = nn.CrossEntropyLoss(reduction='none')(_a_p, _a_g) / tau
        ce_gn = nn.CrossEntropyLoss(reduction='none')(_a_n, _a_g) / tau
        
        cl = -torch.log(torch.exp(-ce_gp)/(torch.exp(-ce_gp) + torch.exp(-ce_gn)))
        
        loss.append(cl.mean())
    
    return sum(loss) / len(loss)


def get_loss_fn(name: str):
    assert name in ['horizon_loss', 'action_loss', 'state_loss', 'goal_loss', 'kl_loss', 'cl_loss']
    return globals()[name]