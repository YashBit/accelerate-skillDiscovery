"""
Different encoding priors that are supplied to the incremental agents. The
prior used here are simple, in two ways. First, the observation that the agent
uses to learn is kept the same as before. But, the transformed observation
that is used to form the entropy-based reward is modified in each case. One
rule of thumb that we use is this: any extra dimension that is added as the
prior for the agent reward must be added to the front of standard observation.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def obs_transform_quad(env, obs=None):
    t_obs = env.env._env._physics._named.data.xpos['torso'][:2]
    return np.copy(t_obs)


def obs_transform_ant(env, obs=None):
    xposbefore = env.get_body_com("torso")[0]
    yposbefore = env.get_body_com("torso")[1]
    t_obs = np.array([xposbefore, yposbefore])
    return t_obs


def obs_transform_hopper(env, obs=None):
    data = env.get_body_com("torso")[0]
    return np.array([data, 0])


def record_transform_hopper(env, obs=None):
    xpos = env.get_body_com("torso")[0]
    zpos = env.get_body_com("torso")[2]
    return np.array([xpos, zpos])


def obs_transform_cheetah(env, obs=None):
    data = env.env._env._physics._named.data

    return np.array([
        data.xpos['torso'][0],  # X position
        data.xpos['torso'][2],  # Z position
    ])


def record_transform_swimmer(env, obs=None):
    data = env.env._env._physics._named.data
    head_position = data.xpos['head']

    return np.array([
        head_position[0],  # X position
        head_position[1],  # Z position
    ])


def obs_transform_swimmer(env, obs=None):
    physics = env.env._env._physics
    data = physics._named.data
    center_of_mass = data.subtree_com['head']
    return np.array([
        center_of_mass[0],  # X position
        center_of_mass[1],  # Y position
    ])


def obs_transform_gym_swimmer(env, obs=None):
    position = env.sim.data.qpos.flat[0:2]
    return position.copy()


def obs_transform_gym_swimmer_com(env, obs=None):
    position = env.sim.data.subtree_com.flat[0:2]
    return position.copy()


def obs_transform_default(env, obs):
    return obs


def record_transform_default(env, obs):
    return np.array([])


def transform_obs_weights(env_name, transformed_obs):
    weight_coeffs = TRANSFORMED_OBS_WEIGHTS.get(env_name, 1)
    if type(weight_coeffs) in [int, float]:
        return weight_coeffs * transformed_obs
    else:
        ones = len(weight_coeffs)
        zeros = transformed_obs.shape[0] - ones
        actual_weight = np.concatenate([
            weight_coeffs,
            np.zeros(zeros)
        ])
        return actual_weight * transformed_obs


OBS_TRANSFORMS = {
    'Ant': obs_transform_ant,
    'HalfCheetah': record_transform_hopper,
    'Swimmer': obs_transform_gym_swimmer_com,
    'Hopper': record_transform_hopper,
}

RECORD_TRANSFORMS = {
    'Ant': obs_transform_ant,
    'HalfCheetah': record_transform_hopper,
    'Swimmer': obs_transform_gym_swimmer_com,
    'Hopper': record_transform_hopper,
}


TRANSFORMED_OBS_WEIGHTS = {
    'Ant': [1, 1],
    'HalfCheetah': 1,
}
