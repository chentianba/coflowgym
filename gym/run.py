import sys, time, os
from jpype import *
import numpy as np
import json
import random

from algo.ddpg import DDPG, OUNoise
from coflow import CoflowSimEnv
from train import makeMLFQVal
from util import chengji

kb, mb, gb, tb = 1024**1, 1024**2, 1024**3, 1024**4

def run(env):
    a_dim = env.action_space.shape[0]
    s_dim = env.observation_space.shape[0]
    a_bound = env.action_space.high

    print("a_dim:", a_dim, "s_dim:", s_dim, "a_bound:", a_bound)
    agent = DDPG(a_dim, s_dim, a_bound)

    ################ hyper parameter ##############
    agent.LR_A = 0.001
    agent.LR_C = 0.0001
    agent.GAMMA = 0.99
    agent.TAU = 0.001

    epsilon = 1
    EXPLORE = 400
    TH = 10 # threshold MULT default is 10
    PERIOD_SAVE_MODEL = True
    ###############################################
    agent.load("models/2020-4-24-11-3-27/model_550.ckpt")

    for episode in range(5):
        obs = env.reset()
        ep_reward = 0

        for i in range(int(1e10)):
            action = agent.choose_action(obs)
            obs_n, reward, done, _ = env.step( makeMLFQVal(env, action) )
            obs = obs_n
            ep_reward += reward
            if done:
                print("\nepisode %s: step %s, ep_reward %s"%(episode, i, ep_reward))
                result = env.getResult()
                print("result: ", result, type(result))
                break
    env.close()
    print("Game is over!")

def run_coflowsim(env):

    t_actions = np.array([
        [10*mb, 100*mb, 1*gb, 10*gb, 100*gb, 1*tb],
        # [1*mb, 5*mb, 13*mb, 33*mb, 165*mb, 4212*mb],
    ])
    # t_actions = sample(env.action_space.shape[0])
    f = open("log/sample.json", "r")
    t_actions = json.load(f)["actions"]
    random.shuffle(t_actions)
    print(len(t_actions))
    assert len(t_actions[0]) == env.action_space.shape[0], "ActionDim Error!"
    for action in t_actions[:1000]:
        env.reset()
        for i in range(int(1e10)):
            _ , _, done, _ = env.step(action)
            if done:
                print("Action:", action, "result:", env.getResult())
                break
        sys.stdout.flush()
    env.close()
    print("It's over!")

def sample(a_dim):
    init_limit = 1*kb
    mult = list(range(2, 10))+list(range(10, 110, 10))
    print(init_limit, mult, a_dim)
    count = [-1]*a_dim
    N = len(mult)
    k = 0
    actions = []
    while k >= 0:
        while count[k] < N-1:
            count[k] += 1
            if k == a_dim-1:
                # print(count)
                act = [mult[count[i]] for i in range(a_dim)]
                if chengji(act) >= 1e8:
                    actions.append(act)
            else:
                k += 1
        count[k] = -1
        k -= 1
    print(actions[:10])
    for i in range(len(actions)):
        p = init_limit
        for j in range(len(actions[i])):
            actions[i][j] = actions[i][j]*p
            p = actions[i][j]
    print(actions[:10])
    with open("log/sample.json", "w") as f:
        json.dump({"actions":actions}, f)
    return actions
        

def config_env():
    # Configure the jpype environment
    jarpath = os.path.join(os.path.abspath("."))
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=%s/target/coflowsim-0.2.0-SNAPSHOT.jar"%(jarpath), convertStrings=False)

    java.lang.System.out.println("Hello World!")
    testfile = "./scripts/100coflows.txt"
    benchmark = "./scripts/FB2010-1Hr-150-0.txt"
    args = ["dark", "COFLOW-BENCHMARK", benchmark] # 2.4247392E7
    args = ["dark", "COFLOW-BENCHMARK", testfile] # 326688.0
    CoflowGym = JClass("coflowsim.CoflowGym")
    gym = CoflowGym(args)
    return CoflowSimEnv(gym, False)

def destroy_env():
    shutdownJVM()
    
if __name__ == "__main__":

    env = config_env()

    # record
    # print("training begins: %s"%(time.asctime(time.localtime(time.time()))))

    # main loop
    # run(env)
    run_coflowsim(env)
    # sample(6)

    destroy_env()