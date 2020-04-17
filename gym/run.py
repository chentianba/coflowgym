import sys, time, os
from jpype import *
import numpy as np

from algo.ddpg import DDPG, OUNoise
from coflow import CoflowSimEnv

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
    agent.load("models/2020-4-16-21-0-4/model_40.ckpt")

    for episode in range(5):
        obs = env.reset()
        ep_reward = 0

        for i in range(int(1e10)):
            action = agent.choose_action(obs)
            obs_n, reward, done, _ = env.step( (action+1)*TH/2 )
            obs = obs_n
            ep_reward += reward
            if done:
                print("\nepisode %s: step %s, ep_reward %s"%(episode, i, ep_reward))
                result = env.getResult()
                print("result: ", result, type(result))
                break
    env.close()
    print("Game is over!")

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
    return CoflowSimEnv(gym)

def destroy_env():
    shutdownJVM()
    
if __name__ == "__main__":

    env = config_env()

    # record
    print("training begins: %s"%(time.asctime(time.localtime(time.time()))))
    sys.stdout.flush()

    # main loop
    run(env)

    destroy_env()