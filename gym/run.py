import sys, time, os
from jpype import *
import numpy as np
import json, math
import random

from algo.ddpg import DDPG, OUNoise
from coflow import CoflowSimEnv
from train import makeMLFQVal, action_with_kde
from util import chengji, get_h_m_s, KDE, prepare_pm

kb, mb, gb, tb = 1024**1, 1024**2, 1024**3, 1024**4

def run(env):
    # thresholds = [1.0485760E7*(10**i) for i in range(9)]
    thresholds = np.array([10]*9)
    a_dim = env.action_space.shape[0]
    s_dim = env.observation_space.shape[0]
    a_bound = env.action_space.high

    print("a_dim:", a_dim, "s_dim:", s_dim, "a_bound:", a_bound)
    agent = DDPG(a_dim, s_dim, a_bound)

    ################ hyper parameter ##############
    agent.LR_A = 1e-4
    agent.LR_C = 1e-3
    agent.GAMMA = 0.99
    agent.TAU = 0.001

    epsilon = 1
    EXPLORE = 200
    TH = 20 # threshold MULT default is 10
    PERIOD_SAVE_MODEL = True
    IS_OU = True
    var = 3
    ###############################################

    kde = KDE(prepare_pm())
    models = [220, 230, 240, 250, 260, 270, 280, 290]
    random.shuffle(models)
    print("models:", models)

    for model in models:
        print("Model:", model)
        agent.load("log/models/2020-5-21-23-42-43/model_%s.ckpt"%(model))

        for episode in range(1):

            obs = env.reset()
            ep_reward = 0
            begin = time.time()

            for i in range(int(1e10)):
                action = agent.choose_action(obs)
                obs_n, reward, done, _ = env.step( action_with_kde(kde, action) )
                obs = obs_n
                ep_reward += reward
                if done:
                    print("episode %s: step %s, ep_reward %s, consume time: %s"%(episode, i, ep_reward, get_h_m_s(time.time()-begin)))
                    result = env.getResult()
                    print("result: ", result, type(result))
                    break
        print()
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

def human_inst(env, obs):
    unit_dim = env.UNIT_DIM
    num_coflow = env.NUM_COFLOW
    sent_bs = []
    for i in range(num_coflow):
        unit = obs[unit_dim*i:unit_dim*(i+1)]
        # print(type(env.high_property), type(env.low_property), type(unit))
        # actual value: id, width/1000, sent_bytes(B), duration_time/1000
        actual_val = (env.high_property-env.low_property)*unit+env.low_property
        sent_b = actual_val[2]
        if actual_val[0] != 0:
            sent_bs.append(sent_b)
    sent_bs = sorted(sent_bs)
    count = np.array([0]*15) # 1B, 10B, 100B
    for sent in sent_bs:
        if sent > 1:
            index = int(math.log10(sent))
        else:
            index = 0
        count[index] += 1

    return count.reshape(5, 3)

def instruction(coflows):
    c_set = set(coflows)
    c_set = sorted(c_set)
    n = len(c_set)
    N = 7
    threshold = []
    if n <= 1:
        return [(10**i)*10*mb for i in range(6)]
    if n <= N:
        for i in range(1, n):
            threshold.append((c_set[i]+c_set[i-1])/2)
        while len(threshold) < N-1:
            threshold.append(threshold[-1]*10)
        return threshold
    else:
        m = int(n/N)
        for i in range(n%N):
            threshold.append((c_set[(i+1)*(m+1)]+c_set[(i+1)*(m+1)-1])/2)
        for i in range(n%N, N-1):
            threshold.append((c_set[(i+1)*m+n%N]+c_set[(i+1)*m+n%N-1])/2)
        return threshold

def run_human(env):
    a_dim = env.action_space.shape[0]
    s_dim = env.observation_space.shape[0]
    a_bound = env.action_space.high

    ###############################################

    for episode in range(1):
        obs = env.reset()
        ep_reward = 0
        acs = []
        ac = []

        for i in range(int(1e10)):
            # action = agent.choose_action(obs)
            # inst = human_inst(env, obs)
            inst = instruction(ac)
            print("active coflows in step %s:"%(i), np.array(ac))
            # print("inst:", np.array(inst))
            action = inst
            # action = [(10**i)*10*mb for i in range(6)]
            # print(action)
            obs_n, reward, done, info = env.step( np.array(action) )
            # print(info)
            ac = sorted([coflow[2] for coflow in eval(info["obs"].split(":")[-1])])
            acs.append(ac)
            # print("active coflows in step",i,":", np.array(acs[-1]))
            obs = obs_n
            ep_reward += reward
            if done:
                print("\nepisode %s: step %s, ep_reward %s"%(episode, i, ep_reward))
                result = env.getResult()
                print("result: ", result, type(result))
                print("coflow set:", acs)
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
    # args = ["dark", "COFLOW-BENCHMARK", testfile] # 326688.0
    # args = ["dark", "COFLOW-BENCHMARK", "./scripts/test_150_250.txt"] # 1.5923608E7
    # args = ["dark", "COFLOW-BENCHMARK", "./scripts/test_150_200.txt"] # 2214624.0
    # args = ["dark", "COFLOW-BENCHMARK", "./scripts/test_200_250.txt"] # 6915640.0
    # args = ["dark", "COFLOW-BENCHMARK", "./scripts/test_175_200.txt"] # 
    # args = ["dark", "COFLOW-BENCHMARK", "./scripts/test_150_250.txt"] # 
    # args = ["dark", "COFLOW-BENCHMARK", "./scripts/test_200_225.txt"] # 3615440.0
    # args = ["dark", "COFLOW-BENCHMARK", "./scripts/custom.txt"] # 
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
    begin = time.time()
    run(env)
    # run_coflowsim(env)
    # run_human(env)
    # sample(6)
    print("Consume Time:", get_h_m_s(time.time()-begin))

    destroy_env()