from jpype import *
import os
from gym import Env, spaces
from algo.ddpg import DDPG
from algo.ddpg import OUNoise
import numpy as np
import json
import math


class CoflowSimEnv(Env):
    def __init__(self, gym):
        self.coflowsim = gym
        self.NUM_COFLOW = self.coflowsim.MAX_COFLOW # 10
        self.UNIT_DIM = 4 # id, width/1000, sent_bytes, duration_time/1000
        self.STATE_DIM = self.NUM_COFLOW*self.UNIT_DIM
        self.ACTION_DIM = 9

        self.low_property = np.zeros((self.UNIT_DIM,))
        assert self.UNIT_DIM == 4, "UNIT_DIM != 4"
        self.high_property = np.array([526, 21170, 8501205*1048576, 0]) # B, ms
        
        self.observation_space = spaces.Box(0, 1, (self.STATE_DIM,))
        self.action_space = spaces.Box(0, 1, (self.ACTION_DIM,))

        self.MB = 1024*1024 # 1MB = ? B
        self.old_throughout = 0

    def step(self, action):
        # correction for action
        action = np.clip(action, 0.01, 100)

        res = self.coflowsim.toOneStep(action)
        # print("res", res)
        result = json.loads(str(res))
        obs = result["observation"]
        obs = self.__parseObservation(obs)
        done = result["done"]
        # obs = self.coflowsim.printStats()
        # obs = np.zeros(self.observation_space.shape)

        ### calculate the throughout
        completed = result["completed"]
        c_coflows = eval(completed.split(":")[-1])
        throughout = 0
        for coflow in c_coflows:
            ## id, width, total bytes(B), duration time(ms)
            throughout += (coflow[-2]/(coflow[-1]*1024)) # unit is Mbs
        ### calculate the reward
        reward = 0
        if self.old_throughout == 0:
            if throughout == 0:
                reward = 0
            else:
                reward = 100
        else:
            rate = throughout/self.old_throughout
            reward = np.clip(rate if rate <= 1 else rate*10, 0, 100)
        self.old_throughout = throughout
        # print("throughout: ", throughout, "reward: ", reward)
        
        return obs, reward, done, {}
    
    def __parseObservation(self, obs):
        arr = obs.split(":")[-1]
        arrs = sorted(eval(arr), key=lambda x: x[2], reverse=True) # sort according to sent bytes
        arrs = arrs[:self.NUM_COFLOW]
        state = []
        for a in arrs:
            ## id, width, already bytes(B), duration time(ms)
            # power = int(math.log(a[2], 10)) if a[2] != 0 else 0
            # b = (a[0], a[1]/1000, a[2]/(10**power), power, a[3]/1000)
            if a[3] > self.high_property[3]:
                self.high_property[3] = a[3]
            # print("parse: ", a, self.low_property, self.high_property)
            b = [(a[i]-self.low_property[i])/(self.high_property[i]-self.low_property[i]) for i in range(self.UNIT_DIM)]
            # print(b)
            state.extend(b)
        if len(arrs) < self.NUM_COFLOW:
            state.extend([0]*(self.NUM_COFLOW-len(arrs))*self.UNIT_DIM)
        return np.array(state)

    def getResult(self):
        return self.coflowsim.printStats()
    
    def reset(self):
        obs = self.coflowsim.reset()
        return self.__parseObservation(str(obs))
    
    def render(self):
        pass
    
    def close(self):
        pass

def loop(env):
    """Coflow Environment
    """
    # thresholds = [1.0485760E7*(10**i) for i in range(9)]
    thresholds = np.array([10]*9)
    a_dim = env.action_space.shape[0]
    s_dim = env.observation_space.shape[0]
    a_bound = env.action_space.high

    print("a_dim:", a_dim, "s_dim:", s_dim, "a_bound:", a_bound)
    agent = DDPG(a_dim, s_dim, a_bound)
    oun = OUNoise(a_dim, mu=0.4)

    epsilon = 1
    EXPLORE = 70
    TH = 10 # threshold MULT default is 10
    ave_rs = []
    for episode in range(1000):
        obs = env.reset()
        ep_reward = 0
        oun.reset()
        epsilon -= (epsilon/EXPLORE)

        for i in range(int(1e10)):
            ## Add exploration noise
            action_original = agent.choose_action(obs)
            # action = np.array(thresholds)
            action = action_original + max(0.01, epsilon)*oun.noise()

            # because of `tanh` activation which valued in [-1, 1], we need to scale
            obs_n, reward, done, _ = env.step( (action+1)*TH/2 )
            print("episode %s step %s"%(episode, i))
            print("obs_next: ", obs_n.reshape(-1, env.UNIT_DIM), "reward: ", reward, "done: ", done, "action: ", action)
            agent.store_transition(obs, action, reward, obs_n)

            if agent.pointer > agent.BATCH_SIZE:
                agent.learn()
            
            obs = obs_n
            ep_reward += reward
            if done:
                print("\nepisode %s: step %s, ep_reward %s"%(episode, i, ep_reward))
                # print("Observation: ", obs)
                result = env.getResult()
                print("result: ", result, type(result))
                break
        # if episode % 50 == 0 and False:
        #     total_reward = 0
        #     for i in range(int(3)):
        #         state = env.reset()
        #         for j in range(int(1e10)):
        #             action = agent.choose_action(state)
        #             state, reward, done, _ = env.step((action+1)*TH/2)
        #             total_reward += reward
        #             if done:
        #                 break
        #     ave_reward = total_reward/5
        #     print("episode: ", episode, "Evaluation Average Reward: ", ave_reward)
        #     ave_rs.append(ave_reward)
        #     print(ave_rs)
        #     if False:
        #         return
    
    env.close()
    print("Game is over!")


if __name__ == "__main__":
    # Configure the jpype environment
    jarpath = os.path.join(os.path.abspath("."))
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=%s/target/coflowsim-0.2.0-SNAPSHOT.jar"%(jarpath), convertStrings=False)

    java.lang.System.out.println("Hello World!")
    testfile = "/home/chentb/project/coflowgym/scripts/100coflows.txt"
    benchmark = "/home/chentb/tmp/git/coflow-benchmark/FB2010-1Hr-150-0.txt"
    args = ["dark", "COFLOW-BENCHMARK", benchmark] # 2.4247392E7
    # args = ["dark", "COFLOW-BENCHMARK", testfile] # 326688.0
    CoflowGym = JClass("coflowsim.CoflowGym")
    gym = CoflowGym(args)

    # main loop
    loop(CoflowSimEnv(gym))

    shutdownJVM()
