import cv2
import torch
import pickle
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F
from copy import deepcopy
from rich import print
from mineclip import MineCLIP
from ray.rllib.models.torch.vpt_backbone import VPTBackbone
from src.utils.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from src.utils.impala_lib.impala_cnn import ImpalaCNN
from src.utils.impala_lib.goal_impala_cnn import GoalImpalaCNN
from src.utils.impala_lib.util import FanInInitReLULayer

def resize_image(img, target_resolution=(128, 128)):
    if type(img) == np.ndarray:
        img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)
    elif type(img) == torch.Tensor:
        img = F.interpolate(img, size=target_resolution, mode='bilinear')
    else:
        raise ValueError
    return img

class MineCLIPWrapper(MineCLIP):
    
    MINECLIP_AGENT_RESOLUTION = (160, 256)
    cfg = {'arch': 'vit_base_p16_fz.v2.t2', 'hidden_dim': 512, 'image_feature_dim': 512, 'mlp_adapter_spec': 'v0-2.t0', 
            'pool_type': 'attn.d2.nh8.glusw', 'resolution': [160, 256]}
    def __init__(self, weight_path, **kwargs):
        MineCLIP.__init__(self, **MineCLIPWrapper.cfg)
        if weight_path != "":
            print(f"[VISION] loading weights from {weight_path}. ")
            self.load_ckpt(weight_path, strict=True)
        for name, param in self.named_parameters():
            if "bias" not in name:
                param.requires_grad = False
    
    def forward(self, img, goal_embeddings):
        assert len(img.shape) == 4
        B, C, H, W = img.shape
        inp = resize_image(img.float(), MineCLIPWrapper.MINECLIP_AGENT_RESOLUTION)
        inp = inp.reshape(B, 1, C, MineCLIPWrapper.MINECLIP_AGENT_RESOLUTION[0], MineCLIPWrapper.MINECLIP_AGENT_RESOLUTION[1])
        image_feats = super().forward_image_features(inp)
        # image_feats = MineCLIP.forward_image_features(self, inp)
        # features = MineCLIP.forward_video_features(self, image_feats)
        return image_feats


class VPTWrapper(VPTBackbone):

    def __init__(self, model_path, weight_path = "", **kwargs):
        agent_parameters = pickle.load(open(model_path, "rb"))
        # ! remember to restore it
        # agent_parameters["model"]["args"]["net"]["args"]["img_shape"] = [128, 128, 4]
        print(f"parameters cfg: ", agent_parameters["model"]["args"]["net"]["args"])
        VPTBackbone.__init__(self, net_config=agent_parameters["model"]["args"]["net"]["args"])
        if weight_path != "":
            print("[VISION] using pretrained weights to initialize vpt backbone (impala cnn)! ")
            weights = torch.load(weight_path)
            self.load_weights(weights)
        else:
            print("[VISION] train vpt backbone (impala cnn) from scratch! ")
            
        for name, param in self.named_parameters():
            if "bias" not in name:
                param.requires_grad = False
    
    def forward(self, img, goal_embeddings):
        assert len(img.shape) == 4
        img = resize_image(img, (128, 128))
        img = img.permute(0, 2, 3, 1)
        opt = VPTBackbone.forward(self, img)
        return opt


class ImpalaCNNWrapper(nn.Module):
    
    def __init__(self, scale='1x', **kwargs):
        super().__init__()
        if scale == '1x':
            net_config = {
                'hidsize': 1024,
                'img_shape': [128, 128, 3],
                'impala_chans': [16, 32, 32],
                'impala_kwargs': {'post_pool_groups': 1},
                'impala_width': 4,
                'init_norm_kwargs': {'batch_norm': False, 'group_norm_groups': 1},
            }
        elif scale == '3x':
            net_config = {
                'hidsize': 3072,
                'img_shape': [128, 128, 3],
                'impala_chans': [16, 32, 32],
                'impala_kwargs': {'post_pool_groups': 1},
                'impala_width': 12,
                'init_norm_kwargs': {'batch_norm': False, 'group_norm_groups': 1},
            }
        else:
            assert False
        
        hidsize = net_config['hidsize']
        img_shape = net_config['img_shape']
        impala_width = net_config['impala_width']
        impala_chans = net_config['impala_chans']
        impala_kwargs = net_config['impala_kwargs']
        init_norm_kwargs = net_config['init_norm_kwargs']
        
        chans = tuple(int(impala_width * c) for c in impala_chans)
        self.dense_init_norm_kwargs = deepcopy(init_norm_kwargs)
        if self.dense_init_norm_kwargs.get("group_norm_groups", None) is not None:
            self.dense_init_norm_kwargs.pop("group_norm_groups", None)
            self.dense_init_norm_kwargs["layer_norm"] = True
        if self.dense_init_norm_kwargs.get("batch_norm", False):
            self.dense_init_norm_kwargs.pop("batch_norm", False)
            self.dense_init_norm_kwargs["layer_norm"] = True
        
        self.cnn = ImpalaCNN(
            outsize=256,
            output_size=hidsize,
            inshape=img_shape,
            chans=chans,
            nblock=2,
            init_norm_kwargs=init_norm_kwargs,
            dense_init_norm_kwargs=self.dense_init_norm_kwargs,
            first_conv_norm=False,
            **impala_kwargs,
            **kwargs,
        )

        self.linear = FanInInitReLULayer(
            256,
            hidsize,
            layer_type="linear",
            **self.dense_init_norm_kwargs,
        )

    def forward(self, img, goal_embeddings):
        assert len(img.shape) == 4
        img = resize_image(img, (128, 128))
        # print(img)
        img = img.to(dtype=torch.float32)  / 255.
        return self.linear(self.cnn(img))

class GoalImpalaCNNWrapper(nn.Module):
    
    def __init__(self, scale='1x', **kwargs):
        super().__init__()
        if scale == '1x':
            net_config = {
                'hidsize': 1024,
                'img_shape': [128, 128, 3],
                'impala_chans': [16, 32, 32],
                'impala_kwargs': {'post_pool_groups': 1},
                'impala_width': 4,
                'init_norm_kwargs': {'batch_norm': False, 'group_norm_groups': 1},
            }
        elif scale == '3x':
            net_config = {
                'hidsize': 3072,
                'img_shape': [128, 128, 3],
                'impala_chans': [16, 32, 32],
                'impala_kwargs': {'post_pool_groups': 1},
                'impala_width': 12,
                'init_norm_kwargs': {'batch_norm': False, 'group_norm_groups': 1},
            }
        else:
            assert False
        
        hidsize = net_config['hidsize']
        img_shape = net_config['img_shape']
        impala_width = net_config['impala_width']
        impala_chans = net_config['impala_chans']
        impala_kwargs = net_config['impala_kwargs']
        init_norm_kwargs = net_config['init_norm_kwargs']
        
        chans = tuple(int(impala_width * c) for c in impala_chans)
        self.dense_init_norm_kwargs = deepcopy(init_norm_kwargs)
        if self.dense_init_norm_kwargs.get("group_norm_groups", None) is not None:
            self.dense_init_norm_kwargs.pop("group_norm_groups", None)
            self.dense_init_norm_kwargs["layer_norm"] = True
        if self.dense_init_norm_kwargs.get("batch_norm", False):
            self.dense_init_norm_kwargs.pop("batch_norm", False)
            self.dense_init_norm_kwargs["layer_norm"] = True
        
        self.cnn = GoalImpalaCNN(
            outsize=256,
            output_size=hidsize,
            inshape=img_shape,
            chans=chans,
            nblock=2,
            init_norm_kwargs=init_norm_kwargs,
            dense_init_norm_kwargs=self.dense_init_norm_kwargs,
            first_conv_norm=False,
            **impala_kwargs,
            **kwargs,
        )

        self.linear = FanInInitReLULayer(
            256,
            hidsize,
            layer_type="linear",
            **self.dense_init_norm_kwargs,
        )

    def forward(self, img, goal_embeddings):
        '''
        img: BxT, 3, H, W, without normalization
        goal_embeddings: BxT, C
        '''
        img = resize_image(img, (128, 128))
        img = img.to(dtype=torch.float32) / 255.
        return self.linear(self.cnn(img, goal_embeddings))

    def get_cam_layer(self):
        return [ block.conv1 for block in self.cnn.stacks[-1].blocks ] + [ block.conv0 for block in self.cnn.stacks[-1].blocks ]


class ResNetWrapper(nn.Module):
    
    def __init__(self, type="resnet18", pretrained=False, **kwargs):
        super().__init__()
        if type == "resnet18":
            self.model = resnet18(pretrained=pretrained)
        elif type == "resnet34":
            self.model = resnet34(pretrained=pretrained)
        elif type == "resnet50":
            self.model = resnet50(pretrained=pretrained)
        elif type == "resnet101":
            self.model = resnet101(pretrained=pretrained)
        print("[VISION] train resnet from scratch! ")
        
    def forward(self, img, goal_embeddings):
        assert len(img.shape) == 4
        img = img / 255.
        img = resize_image(img, target_resolution=(128, 128))
        opt = self.model(img)
        return opt
        

def create_backbone(name, model_path = "", weight_path = "", **kwargs):
    assert name in ['impala_cnn', 'vpt', 'mineclip', 'impala_1x', 'impala_2x', 'impala_3x', \
                    'goal_impala_1x', 'goal_impala_2x', 'goal_impala_3x', \
                    'resnet18', 'resnet34', 'resnet50', 'resnet101'], \
                    f"[x] backbone {name} is not surpported!"
    if name == 'vpt':
        assert model_path != "", "[VISION] vpt backbone (impala cnn) need model path!"
        return VPTWrapper(model_path, weight_path, **kwargs)
    elif name == 'mineclip':
        assert weight_path != "", "[VISION] mineclip model need weight path!"
        return MineCLIPWrapper(weight_path, **kwargs)
    elif name == 'resnet18':
        return ResNetWrapper('resnet18', **kwargs)
    elif name == 'resnet34':
        return ResNetWrapper('resnet34', **kwargs)
    elif name == 'resnet50':
        return ResNetWrapper('resnet50', **kwargs)
    elif name == 'resnet101':
        return ResNetWrapper('resnet101', **kwargs)
    elif name == 'impala_1x':
        return ImpalaCNNWrapper('1x', **kwargs)
    elif name == 'goal_impala_1x':
        return GoalImpalaCNNWrapper('1x', **kwargs)
    elif name == 'goal_impala_3x':
        return GoalImpalaCNNWrapper('3x', **kwargs)


if __name__ == '__main__':
    vpt = create_backbone("impala_cnn", "/home/caishaofei/workspace/MODEL_BASE/models/3x.model", "/home/caishaofei/workspace/MODEL_BASE/weights/foundation-model-3x.weights")
    inp = torch.ones(2, 3, 128, 128)
    opt = vpt(inp)
    print(opt.shape)
    
    mineclip = create_backbone("mineclip", weight_path="/home/caishaofei/workspace/CODE_BASE/MineCLIP-lucky/pretrained/attn.pth")
    inp = torch.ones(2, 3, 128, 128)
    opt = mineclip(inp)
    print(opt.shape)