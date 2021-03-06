#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl

from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils

import dmc2gym
import hydra

# Import the environments

# envs
import gym
from gym.spaces import Discrete, MultiBinary
from rlkit.envs.point_robot_new import PointEnv as PointEnv2
from rlkit.envs.point_reacher_env import PointReacherEnv
from rlkit.envs.updated_half_cheetah import HalfCheetahEnv
from rlkit.envs.wrappers import NormalizedBoxEnv, TimeLimit
from rlkit.envs.fetch_reach import FetchReachEnv
# from rlkit.envs.updated_ant import AntEnv

NUM_GPUS_AVAILABLE = 4  # change this to the number of gpus on your system




def make_env(cfg):
    """Helper function to create dm_control environment"""
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])

    gym_envList = ["AntEnv", "HalfCheetahEnv", "PointEnv2", "PointReacherEnv"]

    if cfg.env in gym_envList:
        env = gym.make(domain_name=domain_name,
                        task_name=task_name,
                        seed=cfg.seed,
                        visualize_reward=True)
    else:
        env = dmc2gym.make(domain_name=domain_name,
                        task_name=task_name,
                        seed=cfg.seed,
                        visualize_reward=True)
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)


        #THE ENVIRONMENT INFORMATION IS ADDED HERE IN THE PARAMETERS. 
        self.env = utils.make_env(cfg)

        # TODO(Mahi): Set up the skill space here.

        #DIAYN AGENT IS SETUP HERE.
        # print("New Information")
        # print(self.env.observation_space.shape[0])
        # print(self.env.action_space.shape[0])
        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        # print("Env info: ")
        # print(f"Observation space: {self.env.observation_space.shape[0]}")


        # print(f"The observation shape in DIAYN is self.env.observation_space.shape[0], {self.env.observation_space.shape[0]}")

        # print(f"The replay buffer env shape self.env.observation_space.shape : {self.env.observation_space.shape}")
        # print(f"Replay buffer capacity: {cfg.replay_buffer_capacity}")
    
        self.agent = hydra.utils.instantiate(cfg.agent)

        # TODO(Mahi): Set up the discriminator here


        # TODO(Mahi): Augment the replay buffer with the skill information
        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          (cfg.agent.params.skill_dim, ),
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode < 3))
            done = False
            episode_reward = 0
            skill = self.agent.skill_dist.sample()

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, skill, sample=False)
                obs, reward, done, _ = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}_{episode}_skill_{skill.argmax().cpu().item()}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
                # TODO(Mahi): Sample a skill here.

                #SAMPLE SKILL
                skill = utils.to_np(self.agent.skill_dist.sample())

                self.logger.log('train/episode', episode, self.step)


            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    # print(f"Shape of skill before in eval mode act is : {skill.shape}, obs shape is : {obs.shape}")
                    action = self.agent.act(obs, skill, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                #HERE IS WHERE THE REPLAY BUFFER IS ADDED. 

                #ADDING INFORMATION TO THE UPDATE FUNCTION FOR DIAYN

                self.agent.update(self.replay_buffer, self.logger, self.step)


            # print(f"The size of the action after act is: {action.size}")
            # print(f"The action is : {action}")

            next_obs, reward, done, _ = self.env.step(action)
            # print(f"The next_obs is : {next_obs}")

            # allow infinite bootstrap
            done = float(done)


            # DONENOMAX, the critic does not get updated in the last step?
            
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            # print(f"Values in DIAYN: Done: {done}, done_no_max: {done_no_max}")
            episode_reward += reward


            """

                ADDING THE INFORMATION ON THE REPLAY BUFFER. 

            """

            self.replay_buffer.add(obs, action, reward, next_obs, skill,
                                   done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1




#CFG comes from the configuration path. 
@hydra.main(config_path='/home/yb1025/Research/GRAIL/relabeler-irl/accelerate-skillDiscovery/library-algo/diayn-main/config/train2.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
