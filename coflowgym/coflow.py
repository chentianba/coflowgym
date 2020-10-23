from jpype import *
import os
from gym import Env, spaces
from coflowgym.algo.ddpg import DDPG
from coflowgym.algo.ddpg import OUNoise
import numpy as np
import json
import math, sys, time
from coflowgym.util import Logger, KDE

# logger = Logger("log/mlfq.txt")
# logger = Logger("log/result.txt")

class CoflowSimEnv(Env):
    def __init__(self, gym, debug=True):
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
        self.__initialize()

        self.debug = debug

    def __initialize(self):
        self.old_throughout = 0
        self.old_ave_duration = 0
        self.ep_f_coflows = []

    def step(self, action):
        # correction for action
        # action = np.clip(action, 1e-10, 100)

        res = self.coflowsim.toOneStep(action)
        # print("res", res)
        result = json.loads(str(res))
        obs = result["observation"]
        obs = self.__parseObservation(obs)
        done = result["done"]
        mlfq = eval(result["MLFQ"])
        # if self.debug:
        #     logger.print("MLFQ: "+str(mlfq))
        # obs = self.coflowsim.printStats()
        # obs = np.zeros(self.observation_space.shape)

        ### calculate the reward
        # print("result: ", result)
        completed = result["completed"]
        a_coflows = eval(result["observation"].split(":")[-1])
        c_coflows = eval(completed.split(":")[-1])
        # reward = self.__calculate_reward(a_coflows, c_coflows)
        # reward = self.__cal_reward_2(a_coflows, c_coflows)
        # reward = self.__cal_reward_3(a_coflows, c_coflows)
        reward = self.__cal_reward_4(a_coflows, c_coflows) ## best
        # reward = self.__cal_reward_5(a_coflows, c_coflows, mlfq)

        # print("completed: ", [coflow[0] for coflow in c_coflows])
        
        return obs, reward, done, {"mlfq":mlfq, "obs":result["observation"]}
    
    def __cal_reward_5(self, a_coflows, c_coflows, mlfq):
        r1 = self.__cal_reward_4(a_coflows, c_coflows)
        r2 = -np.std(mlfq)
        alpha = 0.2
        print("r1:", r1, "r2:", r2)
        return alpha*r1 + (1-alpha)*r2

    def __cal_reward_4(self, a_coflows, c_coflows):
        n = len(self.ep_f_coflows)
        old_ave = sum(self.ep_f_coflows)/n if n != 0 else 0
        for coflow in c_coflows:
            self.ep_f_coflows.append(coflow[-1]/1024)
        total_t = 0
        for coflow in a_coflows:
            total_t += (coflow[-1]/1024)
        total_t += sum(self.ep_f_coflows)
        n = (len(self.ep_f_coflows)+len(a_coflows))
        ave_cct = total_t / n if n != 0 else 0
        diff = ave_cct - old_ave
        return -diff

    def __cal_reward_3(self, a_coflows, c_coflows):
        total_time = 0
        for coflow in a_coflows:
            total_time += (coflow[-1]/1024)
        for coflow in c_coflows:
            total_time += (coflow[-1]/1024)
        n = len(a_coflows)+len(c_coflows)
        if n == 0:
            acct = 0
        else:
            acct = total_time/n
        diff = acct - self.old_ave_duration
        self.old_ave_duration = acct
        if diff >= 0:
            r = math.log(diff+1)*10
        else:
            r = -math.log(-diff+1)
        return r
    
    def __cal_reward_2(self, a_coflows, c_coflows):
        ### calculate the average duration about active coflows
        total_duration = 0
        for coflow in c_coflows:
            total_duration += coflow[-1]/1024
        for coflow in a_coflows:
            total_duration += coflow[-1]/1024 # unit is second
        n_coflow = len(a_coflows) + len(c_coflows)
        if n_coflow == 0:
            ave_duration = 0
        else:
            ave_duration = total_duration / n_coflow
        diff = ave_duration - self.old_ave_duration
        self.old_ave_duration = ave_duration

        ### 2. calculate the reward about active coflows
        ### range is [-5, 5]
        r_a = 0
        if diff >= 0:
            r_a = -np.clip(math.log(diff + 1), 0, 10)
        else:
            r_a = np.clip(math.log(-diff + 1), 0, 100)
        return r_a    

    def __calculate_reward(self, a_coflows, c_coflows):
        # print("active: ", a_coflows, "completed: ", c_coflows)

        ## calculate the throughout
        throughout = 0
        for coflow in c_coflows:
            ## id, width, total bytes(B), duration time(ms)
            throughout += (coflow[-2]/(coflow[-1]*1024)) # unit is Mbs
        
        ### 1. calculate the reward about completed coflows
        ### range is [-1, 20]
        r_t = 0
        if self.old_throughout == 0:
            if throughout == 0:
                r_t = 0
            else:
                r_t = 5 # equal to rate = 2
        else:
            rate = throughout/self.old_throughout
            r_t = np.clip(rate-1 if rate <= 1 else math.log(rate, 1.15), -1, 20)
        self.old_throughout = throughout
        
        # **************************************************#

        ### calculate the average duration about active coflows
        total_duration = 0
        for coflow in a_coflows:
            total_duration += coflow[-1]/1024 # unit is second
        if len(a_coflows) == 0:
            ave_duration = 0
        else:
            ave_duration = total_duration / len(a_coflows)
        diff = ave_duration - self.old_ave_duration
        self.old_ave_duration = ave_duration

        ### 2. calculate the reward about active coflows
        ### range is [-5, 5]
        r_a = 0
        if diff >= 0:
            r_a = -np.clip(math.log(diff + 1), 0, 5)
        else:
            r_a = np.clip(math.log(-diff + 1), 0, 5)

        alpha = 0.6 # discount for throughout reward
        reward = alpha*r_t + (1-alpha)*r_a
        return reward

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
        stats = str(self.coflowsim.printStats())
        lines = stats.split("\n")
        result = eval(lines[-1]) # unit is milli second(ms)
        cf_info = lines[:-1]
        return result, cf_info
    
    def reset(self):
        self.__initialize()

        obs = self.coflowsim.reset()
        return self.__parseObservation(str(obs))
    
    def render(self):
        pass
    
    def close(self):
        pass


class CoflowKDEEnv(Env):
    def __init__(self, gym, debug=True, isTest=False):
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
        self.TEST = isTest
        self.__initialize()
        if self.TEST:
            self.test_logger = Logger("log/test_log.txt")

        self.debug = debug

    def __initialize(self):
        self.old_throughout = 0
        self.old_ave_duration = 0
        self.ep_f_coflows = []

        self.kde = KDE(list(range(15))*667)
    
    def get_proto_actions(self, mlfq_actions):
        for mlfq in mlfq_actions:
            for i in range(len(mlfq)):
                mlfq[i] = self.kde.get_prob(mlfq[i])
        return mlfq_actions

    def step(self, action):
        ## action is from nerual network and we need to convert it into MLFQ thresholds
        action = np.clip(action, -1, 1)
        action = sorted(action)
        # sent_s = np.log10([e for e in sentsize if e != 0])
        acts = [self.kde.get_val((a+1)/2) for a in action]
        action = np.power(10, acts)

        return self.__step(action)


    def __step(self, action):
        # correction for action
        # action = np.clip(action, 1e-10, 100)

        res = self.coflowsim.toOneStep(action)
        # print("res", res)
        result = json.loads(str(res))
        obs = result["observation"]
        obs = self.__parseObservation(obs)
        done = result["done"]
        mlfq = eval(result["MLFQ"])
        # if self.debug:
        #     logger.print("MLFQ: "+str(mlfq))
        # obs = self.coflowsim.printStats()
        # obs = np.zeros(self.observation_space.shape)

        ### calculate the reward
        # print("result: ", result)
        completed = result["completed"]
        a_coflows = eval(result["observation"].split(":")[-1])
        c_coflows = eval(completed.split(":")[-1])
        reward = self.__cal_reward_4(a_coflows, c_coflows) ## best

        # print("completed: ", [coflow[0] for coflow in c_coflows])
        ac = [coflow[2] for coflow in eval(result["observation"].split(":")[-1])]
        self.kde.push(np.log10([e for e in ac if e != 0]))
        self.kde.update()
        
        return obs, reward, done, {"mlfq":mlfq, "obs":result["observation"]}

    def __cal_reward_4(self, a_coflows, c_coflows):
        n = len(self.ep_f_coflows)
        old_ave = sum(self.ep_f_coflows)/n if n != 0 else 0
        for coflow in c_coflows:
            self.ep_f_coflows.append(coflow[-1]/1024)
        total_t = 0
        for coflow in a_coflows:
            total_t += (coflow[-1]/1024)
        total_t += sum(self.ep_f_coflows)
        n = (len(self.ep_f_coflows)+len(a_coflows))
        ave_cct = total_t / n if n != 0 else 0
        diff = ave_cct - old_ave
        return -diff

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
        stats = str(self.coflowsim.printStats())
        lines = stats.split("\n")
        result = eval(lines[-1]) # unit is milli second(ms)
        cf_info = lines[:-1]
        return result, cf_info
    
    def reset(self):
        result, cf_info = self.getResult()
        print("Result: ", result)
        if self.TEST:
            self.test_logger.print("Test/Result: %s"%(result))
            self.test_logger.print("Test/CoflowInfo: %s"%(cf_info))

        self.__initialize()

        obs = self.coflowsim.reset()
        return self.__parseObservation(str(obs))
    
    def render(self):
        pass
    
    def close(self):
        pass



if __name__ == "__main__":
    pass
    print(sys.path)
    print("ok")