from lib.tf2rl.tf2rl.algos.gail import Discriminator
from gail.train import config_env

import numpy as np
import tensorflow as tf
import h5py

class DiscReward:
    def __init__(self, state_shape, action_dim, units, model_name):
        self.disc = Discriminator(
                state_shape=state_shape, 
                action_dim=action_dim,
                units=units, 
                enable_sn=False)
        ## load weights of Discriminator
        self.disc.load_weights(model_name)

    def inference(self, states, actions, next_states):
        if states.ndim == actions.ndim == 1:
            states = np.expand_dims(states, axis=0)
            actions = np.expand_dims(actions, axis=0)
        with tf.device("/cpu:0"):
            return self.disc.compute_reward([states, actions])

def train():
    env = config_env()
    state_shape=env.observation_space.shape
    action_dim=env.action_space.high.size
    units=[400, 300]
    file = "log/results/20201111T142620.525755_DDPG_GAIL/model_30S2489.h5"
    # disc_rew = DiscReward(state_shape, action_dim, units, file)
    # disc_rew.disc
    f = h5py.File(file)
    for key, value in f.attrs.items():
        print("{}: {}".format(key, value))


if __name__ == "__main__":
    train()