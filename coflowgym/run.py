import sys, time, os
from jpype import *
import numpy as np
import json, math, pprint
import random

from algo.ddpg import DDPG, OUNoise
from coflow import CoflowSimEnv
from train import makeMLFQVal, action_with_kde
from util import chengji, get_h_m_s, KDE, prepare_pm

kb, mb, gb, tb = 1024**1, 1024**2, 1024**3, 1024**4

args1 = {
    "data": "./scripts/FB2010-1Hr-150-0.txt",
    "models": [220, 230, 240, 250, 260, 270, 280, 290],
    "model_dir": "log/models/2020-5-21-23-42-43", 
    "episode": 1,
    "is_shuffle": True
} ## benchmark

args2 = {
    "data": "./scripts/custom.txt",
    "models": [460, 470],
    "model_dir": "models/2020-5-30-0-6-46", 
    "episode": 3,
    "is_shuffle": False
} ## custom

args3 = {
    "data": "./scripts/FB2010-1Hr-150-0.txt",
    "models": list(range(50, 400, 10)),
    "model_dir": "doc/log/success-2/models/2020-6-4-20-11-22", 
    "episode": 3,
    "is_shuffle": True,
    "detailed": False
} ## benchmark: 对success-2中50-390范围内的model进行测试

args4 = {
    "data": "./scripts/FB2010-1Hr-150-0.txt",
    "models": [110], # 270
    "model_dir": "doc/log/success-2/models/2020-6-4-20-11-22", 
    "episode": 50,
    "is_shuffle": False,
    "detailed": True
} ## benchmark: 对success-2中最好的model测试

args5 = {
    "data": "./scripts/light_tail.txt"
} ## light tail: Test the target of Active Coflows

args6 = {
    "data": "./scripts/light_tail.txt",
    "models": list(range(40, 300, 10)),
    "model_dir": "doc/log/lighttail/models/2020-6-18-16-17-36", 
    "episode": 3,
    "is_shuffle": True,
    "detailed": False
} ## light tail: 对light tail中40-300范围内的model进行测试

args7 = {
    "data": "./scripts/light_tail.txt",
    "models": [70, 120],
    "model_dir": "doc/log/lighttail/models/2020-6-18-16-17-36", 
    "episode": 50,
    "is_shuffle": False,
    "detailed": True
} ## light tail: 对light tail中最好的model进行测试

args8 = {
    "data": "scripts/valid_2.txt"
} ## 测试生成的trace


choice = args8

def run(env, args):
    a_dim = env.action_space.shape[0]
    s_dim = env.observation_space.shape[0]
    a_bound = env.action_space.high

    print("a_dim:", a_dim, "s_dim:", s_dim, "a_bound:", a_bound)
    agent = DDPG(a_dim, s_dim, a_bound)

    print("args:")
    pprint.pprint(args)
    kde = KDE(prepare_pm())
    models = args["models"]
    if args["is_shuffle"]:
        random.shuffle(models)
    print("models:", models)

    for model in models:
        print("Model:", model)
        agent.load("%s/model_%s.ckpt"%(args["model_dir"], model))
        sys.stdout.flush()

        for episode in range(args["episode"]):

            obs = env.reset()
            ep_reward = 0
            sentsize = []
            actions = []
            begin = time.time()
            kde.update()

            for i in range(int(1e10)):
                action = agent.choose_action(obs)
                step_action = action_with_kde(kde, action)
                actions.append(list(step_action))
                obs_n, reward, done, info = env.step(step_action)
                obs = obs_n
                ac = [coflow[2] for coflow in eval(info["obs"].split(":")[-1])]
                sentsize.extend(ac)
                if args["detailed"]:
                    print("step: %s, action: %s, sentsize: %s"%(i,list(step_action),ac))
                ep_reward += reward
                if done:
                    kde.push(np.log10([e for e in sentsize if e != 0]))
                    print("episode %s: step %s, ep_reward %s, consume time: %s"%(episode, i, ep_reward, get_h_m_s(time.time()-begin)))
                    result = env.getResult()
                    print("result: ", result[0], type(result))
                    if args["detailed"]:
                        print("coflows:", result[-1])
                        print("sentsize: ", sentsize)
                        print("Actions:", actions)
                    break
        print()
        sys.stdout.flush()
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
    """
    generate adaptive thresholds by hand, and i-th threshold is the average 
    between i-th coflow and i+1-th coflow in sorted coflow set
    """
    c_set = set(coflows)
    c_set = sorted(c_set)
    n = len(c_set)
    N = 10
    threshold = []
    if n <= 1:
        return [(10**i)*10*mb for i in range(N-1)]
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
    """
    adopt adaptive thresholds to run coflowsim.
    We need to continue improving only if the cct is less than dark.
    """
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
            obs = obs_n
            ep_reward += reward
            if done:
                print("\nepisode %s: step %s, ep_reward %s"%(episode, i, ep_reward))
                result, _ = env.getResult()
                # print("coflow set:", acs)
                print("result: ", result, type(result))
                break
    env.close()
    print("Game is over!")

def config_env():
    # Configure the jpype environment
    jarpath = os.path.join(os.path.abspath("."))
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=%s/target/coflowsim-0.2.0-SNAPSHOT.jar"%(jarpath), convertStrings=False)

    java.lang.System.out.println("Hello World!")
    testfile = "./scripts/test.txt"
    benchmark = "./scripts/FB2010-1Hr-150-0.txt"
    # args = ["dark", "COFLOW-BENCHMARK", benchmark] # 2.4247392E7
    # args = ["dark", "COFLOW-BENCHMARK", "./scripts/light_tail.txt"] # 
    # args = ["dark", "COFLOW-BENCHMARK", testfile] # 326688.0
    # args = ["dark", "COFLOW-BENCHMARK", "./scripts/test_150_250.txt"] # 1.5923608E7
    # args = ["dark", "COFLOW-BENCHMARK", "./scripts/test_150_200.txt"] # 2214624.0
    # args = ["dark", "COFLOW-BENCHMARK", "./scripts/test_200_250.txt"] # 6915640.0
    # args = ["dark", "COFLOW-BENCHMARK", "./scripts/test_175_200.txt"] # 
    # args = ["dark", "COFLOW-BENCHMARK", "./scripts/test_150_250.txt"] # 
    # args = ["dark", "COFLOW-BENCHMARK", "./scripts/test_200_225.txt"] # 3615440.0
    # args = ["dark", "COFLOW-BENCHMARK", "./scripts/custom.txt"] # 
    args = ["dark", "COFLOW-BENCHMARK", choice["data"]]
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
    # run(env, choice)
    # run_coflowsim(env)
    run_human(env)
    # sample(6)
    print("Consume Time:", get_h_m_s(time.time()-begin))

    destroy_env()