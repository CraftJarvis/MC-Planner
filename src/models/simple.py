import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from rich import print
from src.utils.mlp import build_mlp
from src.utils.foundation import ExtraObsEmbedding, PrevActionEmbedding, Bilinear, FiLM, Concat


class SimpleNetwork(nn.Module):
    
    def __init__(
            self,
            action_space: list, 
            state_dim: int,
            goal_dim: int,
            action_dim: int,
            num_cat: int,
            hidden_size: int,
            fusion_type: str,
            max_ep_len:int =4096,
            backbone: nn.Module=None,
            frozen_cnn: bool=False,
            use_recurrent: str=None,
            use_extra_obs: bool=False, 
            use_horizon: bool=False,
            use_pred_horizon: bool=False,
            use_prev_action: bool=False,
            extra_obs_cfg: dict=None,
            c: int=1,
            **kwargs,
    ):
        super().__init__()
        
        types = ['rgb', 'concat', 'bilinear', 'film']
        
        assert fusion_type in types, \
            f"ERROR: [{fusion_type}] is not in {types}"
        
        # print(f"fusion_type is: {fusion_type}")
        
        # self.action_space = action_space
        self.action_space = action_space
        self.num_cat = num_cat
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.fusion_type = fusion_type
        self.frozen_cnn = frozen_cnn
        self.hidden_size = hidden_size
        self.act_pred_dim = sum(self.action_space)
        self.use_recurrent = use_recurrent
        self.use_extra_obs = use_extra_obs
        self.use_horizon = use_horizon
        self.use_prev_action = use_prev_action
        self.use_pred_horizon = use_pred_horizon
        self.c = c
        
        assert (not use_pred_horizon) or use_horizon, "use_pred_horizon is based on use_horizon!"
        
        self.backbone = backbone
        if frozen_cnn:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False
        
        self.extra_dim = 0

        # self.embed_goal = nn.Embedding(num_cat, hidden_size)
        self.embed_goal = nn.Linear(self.goal_dim, hidden_size)
        self.embed_rgb = nn.Linear(self.state_dim, hidden_size)

        if self.use_extra_obs:
            assert extra_obs_cfg is not None, f"ExtraObsEmbedding class arguments are required!"
            self.embed_extra = ExtraObsEmbedding(embed_dims=extra_obs_cfg, output_dim=hidden_size)
            self.extra_dim += hidden_size
        
        if self.use_prev_action:
            self.embed_prev_action = PrevActionEmbedding(output_dim=hidden_size, action_space=self.action_space)
            self.extra_dim += hidden_size


        if fusion_type == "rgb":
            concat_input_dim = hidden_size+self.extra_dim
        elif fusion_type in ["concat"]:
            concat_input_dim = hidden_size*2+self.extra_dim
        elif fusion_type == "bilinear":
            concat_input_dim = hidden_size+self.extra_dim
            self.f_rgb_goal = Bilinear(hidden_size, hidden_size)
        elif fusion_type == "film":
            concat_input_dim = hidden_size+self.extra_dim
            self.f_rgb_goal = FiLM(hidden_size, hidden_size)
        else:
            assert False

        self.concat_input = build_mlp(
            input_dim=concat_input_dim,
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            hidden_depth=3,
        )

        if self.use_recurrent == 'gru':
            self.recurrent = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=0.1,
            )
        elif self.use_recurrent == 'transformer':
            self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
            self.embed_ln = nn.LayerNorm(hidden_size)
            transformer_cfg = kwargs['transformer_cfg']
            config = transformers.GPT2Config(
                vocab_size=1,  # doesn't matter -- we don't use the vocab
                n_embd=hidden_size,
                n_layer=transformer_cfg['n_layer'],
                n_head=transformer_cfg['n_head'],
                n_inner=transformer_cfg['n_head']*hidden_size,
                activation_function= transformer_cfg['activation_function'],
                resid_pdrop=transformer_cfg['resid_pdrop'],
                attn_pdrop=transformer_cfg['attn_pdrop'],
            )
            self.recurrent = transformers.GPT2Model(config)

        if self.use_horizon:
            self.embed_horizon = nn.Embedding(16, hidden_size)
            self.fuse_horizon = Concat(hidden_size, hidden_size)
        
        if self.use_pred_horizon:
            self.pred_horizon = build_mlp(
                input_dim=hidden_size,
                hidden_dim=hidden_size,
                output_dim=16,
                hidden_depth=2,
            )
        
        action_modules = [build_mlp(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=self.act_pred_dim,
            hidden_depth=2,
        )]
        
        self.action_head = nn.Sequential(*action_modules)

    def _img_feature(self, img, goal_embeddings):
        '''
        do the normalization inside the backbone
        '''
        if img.shape[-1] == 3:
            img = img.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = img.shape
        img = img.view(B * T, *img.shape[2:])
        goal_embeddings = goal_embeddings.view(B * T, *goal_embeddings.shape[2:])
        feat = self.backbone(img, goal_embeddings)
        feat = feat.view(B, T, -1)
        return feat

    def forward(self, goals, states, horizons, timesteps=None, attention_mask=None):

        mid_info = {}
        raw_rgb = states['rgb']
        batch_size, seq_length = raw_rgb.shape[:2]
        assert self.use_recurrent or seq_length == 1 , "simple network only supports length = 1 if use_recurrent = None. "
        goal_embeddings = self.embed_goal(goals)
        rgb_embeddings = self.embed_rgb(self._img_feature(raw_rgb, goal_embeddings))
        
        #! compute rgb embeddings based on goal information
        if self.fusion_type == "rgb":
            body_embeddings = rgb_embeddings
        elif self.fusion_type == "concat":
            body_embeddings = torch.cat([rgb_embeddings, goal_embeddings], dim = -1)
        elif self.fusion_type == "bilinear":
            body_embeddings = self.f_rgb_goal(rgb_embeddings, goal_embeddings)
        elif self.fusion_type == "film":
            body_embeddings = self.f_rgb_goal(rgb_embeddings, goal_embeddings)
        elif self.fusion_type == "multihead":
            body_embeddings = rgb_embeddings
        else:
            assert False, f"unknown fusion type. "
        
        #! add extra observation embeddings
        if self.use_extra_obs:
            extra_obs = states
            extra_embeddings = self.embed_extra(extra_obs)
            body_embeddings = torch.cat([body_embeddings, extra_embeddings], dim=-1)
        
        #! add prev action embeddings
        if self.use_prev_action:
            prev_action_embeddings = self.embed_prev_action(states['prev_action'])
            body_embeddings = torch.cat([body_embeddings, prev_action_embeddings], dim=-1)
        
        obs_feature = self.concat_input(body_embeddings)

        #! recurrent network is used to 
        if self.use_recurrent in ['gru', 'lstm']:
            obs_feature, hids = self.recurrent(obs_feature)
        elif self.use_recurrent in ['transformer']:
            time_embeddings = self.embed_timestep(timesteps)
            inputs_embeds = obs_feature + time_embeddings
            # inputs_embeds = self.embed_ln(obs_feature + time_embeddings)
            transformer_outputs = self.recurrent(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            obs_feature = transformer_outputs['last_hidden_state']

        #! add horizon embeddings
        if self.use_horizon:
            if self.use_pred_horizon:
                pred_horizons = self.pred_horizon(obs_feature)
                mid_info['pred_horizons'] = pred_horizons
                if not self.training:
                    mid_horizons = pred_horizons.argmax(-1)
                    mid_horizons = (mid_horizons - self.c).clip(0)
                else:
                    mid_horizons = horizons
            else:
                mid_horizons = horizons
            horizon_embeddings = self.embed_horizon(mid_horizons)
            mid_feature = self.fuse_horizon(obs_feature, horizon_embeddings)
        else:
            mid_feature = obs_feature
        
        action_preds = self.action_head(mid_feature)
        
        return action_preds, mid_info


    def get_action(self, goals, states, horizons):
        # augment the batch dimension
        goals = goals.unsqueeze(0) # 1xLxH
        B, L, _ = goals.shape
        for k, v in states.items():
            states[k] = v.unsqueeze(0)
        if horizons is not None:
            horizons = horizons.unsqueeze(0) # 1xL
        timesteps = torch.arange(L).unsqueeze(0).to(goals.device)
        attention_mask = torch.ones((B, L)).to(goals.device)
        return self.forward(goals, states, horizons, timesteps, attention_mask)
