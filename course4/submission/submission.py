# -*- coding:utf-8  -*-
# Time  : 2022/8/10 下午4:14
# Author: Yahui Cui

"""
# =================================== Important =========================================
Notes:
1. this agents is random agents , which can fit any env in Jidi platform.
2. if you want to load .pth file, please follow the instruction here:
https://github.com/jidiai/ai_lib/blob/master/examples/demo
"""
from pathlib import Path
import os

import numpy as np
import torch
import sys

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
from dqn_agent import DQNAgent

agent = DQNAgent(
    num_actions=1,
    state_shape=36,
    mlp_layers=[64, 64]
)
model_pth = os.path.dirname(os.path.abspath(__file__)) + '/model.pth'
agent = torch.load(model_pth)


def my_controller(observation, action_space, is_act_continuous=True):
    actions = np.zeros(4)
    if observation is None or observation["obs"] is None:
        return [actions.tolist()]
    q_vals = agent.q_estimator.qnet(torch.tensor(np.array([observation["obs"]["observation"]])))
    legal_actions = observation["obs"]["action_mask"]
    for idx, action in enumerate(legal_actions):
        if action is 0:
            q_vals[0][idx] = -999999.
    actions[torch.argmax(q_vals[0])] = 1
    return [actions.tolist()]
