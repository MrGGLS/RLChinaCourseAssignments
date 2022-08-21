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

import numpy
import numpy as np
import torch
import torch.nn.functional as F


class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        # self.fc3 = torch.nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.fc4 = torch.nn.Linear(hidden_dim // 2, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        return self.fc4(x)


current_path = Path(__file__).resolve().parent
q_params_path = os.path.join(current_path, 'q_net_state_dict.pt')
loaded_q_params = torch.load(q_params_path)
q_net = Qnet(3, 64, 24)
q_net.load_state_dict(loaded_q_params)


def dis_to_con(discrete_action, action_dim):  # 离散动作转回连续的函数
    action_lowbound = -2.  # 连续动作的最小值
    action_upbound = 2. # 连续动作的最大值
    return action_lowbound + (discrete_action /
                              (action_dim - 1)) * (action_upbound -
                                                   action_lowbound)


def my_controller(observation, action_space, is_act_continuous=True):
    return [np.array([dis_to_con(q_net(torch.tensor(observation["obs"])).argmax().item(), 24)], dtype=np.float32)]
    # print(action)
    # return action
