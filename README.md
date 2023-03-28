# Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents

<div align="center">

[[Website]](http://www.craftjarvis.org)
[[Arxiv Paper]](https://arxiv.org/pdf/2302.01560.pdf)
[[Team]](https://github.com/CraftJarvis)

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/MineDojo)](https://pypi.org/project/MineDojo/)
[<img src="https://img.shields.io/badge/Framework-PyTorch-red.svg"/>](https://pytorch.org/)
[![GitHub license](https://img.shields.io/github/license/MineDojo/MineCLIP)](https://github.com/MineDojo/MineCLIP/blob/main/license)
</div>

## Updates

* [2023.03.28] Due to the cancellation of access to Codex by OpenAI, planning based on Codex is no longer supported by this repository. We will update to the latest OpenAI model, ChatGPT, which has better performance, as soon as possible.

## Prepare Packages
Our codebase require Python ≥ 3.9. 
Please run the following commands to prepare the environments. 
```sh
conda create -n planner python=3.9 
conda activate planner
python -m pip install numpy torch==2.0.0.dev20230208+cu117 --index-url https://download.pytorch.org/whl/nightly/cu117
python -m pip install -r requirements.txt
python -m pip install git+https://github.com/MineDojo/MineCLIP
```

## Prepare Environment

It also requires a [modified version of MineDojo](https://github.com/CraftJarvis/MC-Simulator) as the simulator and a [goal-conditioned controller](https://github.com/CraftJarvis/MC-Controller). 
```sh
git clone https://github.com/CraftJarvis/MC-Simulator.git
cd MC-Simulator
pip install -e .
```

## Prepare controller checkpoints
Below are the configures and weights of models. 
|Configure|Download| Biome| Number of goals|
|---|---|---|---|
|Transformer| [weights](https://pkueducn-my.sharepoint.com/:f:/g/personal/zhwang_stu_pku_edu_cn/Ev7WGWHL5PpCjMKil0dYrOUBEwfc-y9gM7EPtsQqqp5d0A?e=UdB5gf) | Plains | 4 |

## Prepare OpenAI keys
Our planner depends on Large Language Model like InstructGPT, Codex or ChatGPT. So we need support the OpenAI keys in the file `data/openai_keys.txt`. An OpenAI key list is also accepted.

## Running agent models
To run the code, call 
```sh
python main.py model.load_ckpt_path=<path/to/ckpt>
```
After loading, you should see a window where agents are playing Minecraft. 

|painting|wooden_slab|stone_stairs|
|---|---|---|
|<img src="imgs/obtain_painting.gif" width="200" />|<img src="imgs/obtain_wooden_slab.gif" width="200" />|<img src="imgs/obtain_stone_stairs.gif" width="200" />|

Note: Our planner depends on stable OpenAI API connection. If meeting connection error, please retry it.

## Paper and Citation
Our paper is posted on [Arxiv](https://arxiv.org/pdf/2301.10034.pdf). If it helps you, please consider citing us!
```bib
@article{wang2023describe,
  title={Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents},
  author={Wang, Zihao and Cai, Shaofei and Liu, Anji and Ma, Xiaojian and Liang, Yitao},
  journal={arXiv preprint arXiv:2302.01560},
  year={2023}
}
```
