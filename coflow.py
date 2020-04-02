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
        self.STATE_DIM = self.coflowsim.MAX_COFLOW # 10
        self.UNIT_DIM = 5 # id, width/1000, sent_bytes, power, duration_time/1000
        self.ACTION_DIM = 9

        self.observation_space = spaces.Box(0, 1, (self.STATE_DIM*self.UNIT_DIM,))
        self.action_space = spaces.Box(0, 1, (self.ACTION_DIM,))
        
        self.old_throughout = 0

    def step(self, action):
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
            ## id, width, total bytes, duration time
            throughout += (coflow[-2]/coflow[-1])
        ### calculate the reward
        reward = 0
        if self.old_throughout == 0:
            if throughout == 0:
                reward = 0
            else:
                reward = 1
        else:
            rate = throughout/self.old_throughout
            reward = rate if rate <= 1 else 1
        self.old_throughout = throughout
        # print("throughout: ", throughout, "reward: ", reward)
        
        return obs, reward, done, {}
    
    def __parseObservation(self, obs):
        arr = obs.split(":")[-1]
        arrs = sorted(eval(arr), key=lambda x: x[2], reverse=True) # sort according to sent bytes
        arrs = arrs[:self.STATE_DIM]
        state = []
        for a in arrs:
            ## id, width, already bytes, duration time
            power = int(math.log(a[2], 10)) if a[2] != 0 else 0
            b = (a[0], a[1]/1000, a[2]/(10**power), power, a[3]/1000)
            state.extend(b)
        if len(arrs) < self.STATE_DIM:
            state.extend([0]*(self.STATE_DIM-len(arrs))*self.UNIT_DIM)
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
    thresholds = [1.0485760E7*(10**i) for i in range(9)]
    a_dim = env.action_space.shape[0]
    s_dim = env.observation_space.shape[0]
    a_bound = env.action_space.high

    agent = DDPG(a_dim, s_dim, a_bound)
    oun = OUNoise(a_dim, mu=0.4)
    TH = 1e9

    var = 3 # control exploration
    ave_rs = []
    for episode in range(1000):
        obs = env.reset()
        ep_reward = 0
        for i in range(int(1e10)):
            ## Add exploration noise
            action = agent.choose_action(obs)
            # action = thresholds
            # action = np.clip(np.random.normal(action, var), -1, 1)

            n_action = (action+1)*TH/2
            n_action = thresholds
            obs_n, reward, done, _ = env.step(n_action)
            print("episode %s step %s"%(episode, i), "obs: ", obs, "reward: ", reward, "done: ", done, "action: ", action)
            agent.store_transition(obs, action, reward, obs_n)

            if agent.pointer > agent.MEMORY_CAPACITY:
                var *= .995
                agent.learn()
            
            obs = obs_n
            ep_reward += reward
            if done:
                print("\nepisode %s step %s:"%(episode, i))
                # print("Observation: ", obs)
                print("result: ", env.getResult())
                break
        if episode % 10 == 0 and False:
            total_reward = 0
            for i in range(int(5)):
                state = env.reset()
                for j in range(int(1e10)):
                    action = agent.choose_action(state)
                    state, reward, done, _ = env.step((action+1)*TH/2)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/5
            print("episode: ", episode, "Evaluation Average Reward: ", ave_reward)
            ave_rs.append(ave_reward)
            print(ave_rs)
            if False:
                return
    
    env.close()


if __name__ == "__main__":
    # Configure the jpype environment
    jarpath = os.path.join(os.path.abspath("."))
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=%s/target/coflowsim-0.2.0-SNAPSHOT.jar"%(jarpath), convertStrings=False)

    java.lang.System.out.println("Hello World!")
    testfile = "100coflows.txt"
    benchmark = "/home/chentb/tmp/git/coflow-benchmark/FB2010-1Hr-150-0.txt"
    args = ["dark", "COFLOW-BENCHMARK", benchmark]
    # args = ["dark", "COFLOW-BENCHMARK", testfile] # 326688.0
    CoflowGym = JClass("coflowsim.CoflowGym")
    gym = CoflowGym(args)

    # main loop
    loop(CoflowSimEnv(gym))

    shutdownJVM()
