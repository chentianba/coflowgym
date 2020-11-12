from lib.tf2rl.tf2rl.algos.gail import Discriminator
from lib.tf2rl.tf2rl.algos.ddpg import DDPG
from gail.train import config_env
from coflowgym.algo.ddpg import OUNoise
from coflowgym.util import get_h_m_s, get_now_time

import numpy as np
import tensorflow as tf
import h5py
import time, sys, os

class ReplayBuffer():
    def __init__(self, s_dim, a_dim, capacity=10000, batch_size=64):
        self.MEMORY_CAPACITY = capacity
        self.BATCH_SIZE = batch_size
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.memory = np.zeros((self.MEMORY_CAPACITY, s_dim * 2 + a_dim + 2), dtype=np.float32)
        self.pointer = 0
    
    def store(self, s, a, r, s_, done):
        transition = np.hstack((s, a, [r], s_, [1 if done else 0]))
        print("transition: ", transition)
        index = self.pointer % self.MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def sample(self, ):
        indices = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, self.s_dim + self.a_dim: self.s_dim + self.a_dim + 1]
        bs_ = bt[:, self.s_dim + self.a_dim + 1:self.s_dim*2 + self.a_dim + 1]
        bd = bt[:, -1:]

        return {
            "obs": bs,
            "act": ba,
            "next_obs": bs_,
            "done": bd,
            "reward": br
        }

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
    state_dim = state_shape[0]
    action_dim=env.action_space.high.size
    print(state_shape, action_dim)
    units=[400, 300]
    file = "log/results/20201112T141844.150251_DDPG_GAIL/model_test_2.0S3200.h5"
    disc_rew = DiscReward(state_shape, action_dim, units, file)
    rs = disc_rew.inference(np.zeros((4,40)), np.zeros((4, 9)), None)
    print(np.array(rs))

    if not os.path.exists("log/models"):
        os.mkdir("log/models")
    MODEL_DIR = "log/models/"+get_now_time()

    LOG_FILE = "log/log_ddpg.txt"

    a_bound = 1

    agent = DDPG(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        max_action=env.action_space.high[0],
        gpu=-1, # -1 is only cpu
        actor_units=units,
        critic_units=units,
        n_warmup=10, # default is 10000
        batch_size=100)
    
    PERIOD_SAVE_MODEL = True
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    print("In loop!")
    print("log file:", LOG_FILE)
    print("agent:", agent)
    print("directory of model: ", MODEL_DIR)

    ave_rs = []

    begin_time = time.time()

    for episode in range(1, 1000):
        obs = env.reset()

        ep_reward = 0
        mlfqs = []
        sentsize = []

        ep_time = time.time()
        for i in range(int(1e10)):
            ## Add exploration noise
            action_original = agent
            if episode < 3:
                action = env.action_space.sample()
            else:
                action = agent.get_action(obs)

            ## because of `tanh` activation which valued in [-1, 1], we need to scale
            obs_n, _, done, info = env.step(action)
            print("episode %s step %s"%(episode, i))
            print("obs_next:", obs_n.reshape(-1, env.UNIT_DIM), "done:", done)
            print("action:", action.tolist())
            # mlfqs.append(info["mlfq"])
            ac = [coflow[2] for coflow in eval(info["obs"].split(":")[-1])]
            sentsize.extend(ac)
            print("active coflow:", np.array(sorted(ac)))

            ## calculate the reward before storing data
            rewards = disc_rew.inference(np.array([obs], dtype="float32"), np.array([action]), None)
            reward = np.array(rewards[0])[0]
            replay_buffer.store(obs, action, reward, obs_n, False)

            start_learning = episode >= 3
            start_learning = True
            if start_learning:
                samples = replay_buffer.sample()
                agent.train(samples["obs"], samples["act"], samples["next_obs"], samples["reward"], samples["done"], None)

            
            obs = obs_n
            ep_reward += reward
            if done:
                ## print stats
                result, cf_info = env.getResult()
                print("episodic sentsize:", sorted(sentsize))
                print("cf_info:", cf_info)
                print("\nepisode %s: step %s, ep_reward %s"%(episode, i, ep_reward))
                print("result: ", result)
                print("time: total-%s, episode-%s"%(get_h_m_s(time.time()-begin_time), get_h_m_s(time.time()-ep_time)))
                sys.stdout.flush()
                break
        if PERIOD_SAVE_MODEL and episode%10 == 0:
            model_name = "%s/model_%s.ckpt"%(MODEL_DIR, episode)
            # agent.save(model_name)

    env.close()
    print("Game is over!")

if __name__ == "__main__":
    train()