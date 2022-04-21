import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import utils


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, skill_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.Q1 = utils.mlp(obs_dim + action_dim + skill_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(obs_dim + action_dim + skill_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action, skill):
        assert obs.size(0) == action.size(0)
        assert obs.size(0) == skill.size(0)

        obs_action_skill = torch.cat([obs, action, skill], dim=-1)
        # print(f"OBSACTIONSKILL INFO is: {obs_action_skill}, its shape is : {obs_action_skill.shape}, its type is: {type(obs_action_skill)}")


        #CRITIC 1 AND 2.
        q1 = self.Q1(obs_action_skill)
        q2 = self.Q2(obs_action_skill)

        # print(f"OBS is : {obs}, its shape is : {obs.shape}")
        # print(f"action is : {action}, its shape is : {action.shape}")

        # print(f"skill is : {skill}, its shape is : {skill.shape}")


        # print(f"CRITIC OUTPUTS ARE: {q1}, q2: {q2}")

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2


    #Functions for GHER Acceleration 

    def returnQ(self, obs, action, skill):
        q1 = self.Q1(obs_action_skill)
        q2 = self.Q2(obs_action_skill)
        return q1, q2 


    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)
