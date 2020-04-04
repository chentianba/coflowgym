import sys, time, os
from jpype import *
import numpy as np

from algo.ddpg import DDPG, OUNoise
from coflow import CoflowSimEnv
from util import get_h_m_s, get_now_time

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
    PERIOD_SAVE_MODEL = True

    ave_rs = []

    begin_time = time.time()
    dir = "./models/"+get_now_time()

    for episode in range(1, 1000):
        obs = env.reset()
        ep_reward = 0
        oun.reset()
        epsilon -= (epsilon/EXPLORE)

        ep_time = time.time()
        for i in range(int(1e10)):
            ## Add exploration noise
            action_original = agent.choose_action(obs)
            # action_original = np.array(thresholds)
            action = action_original + max(0.01, epsilon)*oun.noise()

            ## because of `tanh` activation which valued in [-1, 1], we need to scale
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
                result = env.getResult()
                print("result: ", result, type(result))
                print("time: total-%s, episode-%s"%(get_h_m_s(time.time()-begin_time), get_h_m_s(time.time()-ep_time)))
                sys.stdout.flush()
                break
        if PERIOD_SAVE_MODEL and episode%20 == 0:
            model_name = "%s/model_%s.ckpt"%(dir, episode)
            agent.save(model_name)

    env.close()
    print("Game is over!")

def config_env():
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
    return CoflowSimEnv(gym)

def destroy_env():
    shutdownJVM()

if __name__ == "__main__":
    file = open("log/log.txt", "w")
    sys.stdout = file

    env = config_env()

    # record
    print("training begins: %s"%(time.asctime(time.localtime(time.time()))))
    sys.stdout.flush()

    # main loop
    loop(env)

    destroy_env()