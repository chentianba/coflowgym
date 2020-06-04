import sys, time, os, math
from jpype import *
import numpy as np

from algo.ddpg import DDPG, OUNoise
from algo.ddpg_lstm import DDPG_LSTM
from algo.ddpg_prob import DDPGProb
from coflow import CoflowSimEnv
from util import get_h_m_s, get_now_time, KDE, prepare_pm

if not os.path.exists("./models"):
    os.mkdir("./models")
MODEL_DIR = "./models/"+get_now_time()

LOG_FILE = "log/log_10.txt"

def makeMLFQVal(env, thresholds):
    NUM_MLFQ = env.ACTION_DIM
    assert NUM_MLFQ == len(thresholds), "length of thresholds doesnot Match!"
    thresholds = np.clip(thresholds, -1, 10)

    mb = 1024**2
    # kb, gb, tb = mb/1000, mb*1000, mb*1000000
    kb, gb, tb = 1024, 1024**3, 1024**4

    NO = 5
    if NO is 0:
        ## scale to normalization
        for i in range(NUM_MLFQ):
            # if thresholds[i] < 0:
            #     thresholds[i] = thresholds[i]/2+1
            # else:
            #     thresholds[i] = thresholds[i]*4+1
            thresholds[i] = (thresholds[i]+1)*5
        thresholds = np.clip(thresholds, 1.0001, 10)

        ## applied to MLFQ
        # baseline = [300*kb, mb, 3*mb, 10*mb, 30*mb, 100*mb, 1*gb, 10*gb, 100*gb]
        baseline = [10*mb, 100*mb, gb, 10*gb, 100*gb, tb, 10*tb, 100*tb, 1000*tb]
        # baseline = [10*kb, 100*kb, 1*mb, 10*mb, 100*mb, gb, 10*gb, 100*gb, tb]
        for i in range(NUM_MLFQ):
            thresholds[i] = thresholds[i]*baseline[i]
        return np.array(thresholds)
    if NO is 1: ## 7 queues
        initial = 1 # 1B
        for i in range(NUM_MLFQ):
            if thresholds[i] < 0:
                thresholds[i] = thresholds[i]*9+10
            else:
                thresholds[i] = thresholds[i]*90+10
        thresholds = np.clip(thresholds, 1.0001, 100)
        for i in range(NUM_MLFQ):
            initial = initial*thresholds[i]
            thresholds[i] = initial
        return np.array(thresholds)
    if NO is 2: ## 7 queues
        initial = 1*kb # 1K
        for i in range(NUM_MLFQ):
            thresholds[i] = math.pow(10, thresholds[i]+1)
        thresholds = np.clip(thresholds, 1.0001, 100)
        for i in range(NUM_MLFQ):
            initial = initial*thresholds[i]
            thresholds[i] = initial
        return np.array(thresholds)
    if NO is 3: ## 4 queues
        initial = 1*kb # 1K
        for i in range(NUM_MLFQ):
            thresholds[i] = math.pow(10, (thresholds[i]+1)*1.5)
        thresholds = np.clip(thresholds, 1.0001, 1000)
        for i in range(NUM_MLFQ):
            initial = initial*thresholds[i]
            thresholds[i] = initial
        return np.array(thresholds)
    if NO is 4: ## 7 or 10 queues
        initial = 1*mb
        for i in range(NUM_MLFQ):
            thresholds[i] = (thresholds[i]+1)*5
        thresholds = np.clip(thresholds, 0.01, 100)
        for i in range(NUM_MLFQ):
            initial = initial*thresholds[i]
            thresholds[i] = initial
        return np.array(thresholds)
    if NO is 5: ## 7 queues
        initial = 1*kb # 1K
        for i in range(NUM_MLFQ):
            thresholds[i] = math.pow(10, 1.5*(thresholds[i]+1))
        thresholds = np.clip(thresholds, 1.0001, 1000)
        for i in range(NUM_MLFQ):
            initial = initial*thresholds[i]
            thresholds[i] = initial
        return np.array(thresholds)
    if NO is 6: ## 7 or 10 queues
        pass

def action_with_kde(kde, action):
    action = np.clip(action, -1, 1)
    action = sorted(action)
    # sent_s = np.log10([e for e in sentsize if e != 0])
    acts = [kde.get_val((a+1)/2) for a in action]
    return np.power(10, acts)

def action_with_softmax(kde, action):
    action = np.clip(action, 0, 1)
    action = np.array(action)/sum(action)
    act = [0]
    for e in action:
        act.append(act[-1]+e)
    del act[0]

    action = act[:-1]
    acts = [kde.get_val((a+1)/2) for a in action]
    return np.power(10, acts)

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
    oun = OUNoise(a_dim, mu=0)

    ################ hyper parameter ##############
    agent.LR_A = 1e-4
    agent.LR_C = 1e-3
    agent.GAMMA = 0.99
    agent.TAU = 0.001

    epsilon = 1
    EXPLORE = 70 # ep 320 (threshold is 0.01), 480(0.001)
    TH = 20 # threshold MULT default is 10
    PERIOD_SAVE_MODEL = True
    IS_OU = True
    var = 3
    ###############################################

    print("In loop!")
    print("log file:", LOG_FILE)
    print("agent:", agent)
    print("EXPLORE:", EXPLORE)
    print("IS_OU:", IS_OU)
    print("directory of model: ", MODEL_DIR)

    kde = KDE(prepare_pm())    
    kde = KDE(list(range(15)))
    ave_rs = []

    begin_time = time.time()

    for episode in range(1, 1000):
        obs = env.reset()
        oun.reset()
        epsilon -= (epsilon/EXPLORE)

        ep_reward = 0
        mlfqs = []
        kde.print()
        kde.update()
        sentsize = []

        ep_time = time.time()
        for i in range(int(1e10)):
            ## Add exploration noise
            action_original = agent.choose_action(obs)
            # action_original = np.array(thresholds)
            # action_original = (np.random.rand(a_dim))*2-1
            if IS_OU:
                # action = action_original + max(0.01, epsilon)*oun.noise()
                action = action_original + epsilon*oun.noise()
            else:
                action = np.clip(np.random.normal(action_original, var), -1, 1)

            ## because of `tanh` activation which valued in [-1, 1], we need to scale
            step_action = action_with_kde(kde, action)
            # obs_n, reward, done, info = env.step( makeMLFQVal(env, action) )
            obs_n, reward, done, info = env.step(step_action)
            print("episode %s step %s"%(episode, i))
            print("obs_next:", obs_n.reshape(-1, env.UNIT_DIM), "reward:", reward, "done:", done)
            print("action:", action.tolist(), "step action:", step_action, "original:", action_original.tolist())
            # mlfqs.append(info["mlfq"])
            ac = [coflow[2] for coflow in eval(info["obs"].split(":")[-1])]
            sentsize.extend(ac)
            print("active coflow:", np.array(sorted(ac)))

            agent.store_transition(obs, action, reward, obs_n)

            start_learning = agent.pointer > agent.BATCH_SIZE
            if start_learning:
                agent.learn()
                var *= 0.9995
            
            obs = obs_n
            ep_reward += reward
            if done:
                if start_learning:
                    kde.push(np.log10([e for e in sentsize if e != 0]))

                ## print stats
                result, cf_info = env.getResult()
                print("episodic sentsize:", sorted(sentsize))
                print("cf_info:", cf_info)
                print("\nepisode %s: step %s, ep_reward %s"%(episode, i, ep_reward))
                print("result: ", result)
                if IS_OU:
                    print("epsilon:", epsilon)
                else:
                    print("var:", var)
                print("time: total-%s, episode-%s"%(get_h_m_s(time.time()-begin_time), get_h_m_s(time.time()-ep_time)))
                sys.stdout.flush()
                break
        if PERIOD_SAVE_MODEL and episode%10 == 0:
            model_name = "%s/model_%s.ckpt"%(MODEL_DIR, episode)
            agent.save(model_name)

    env.close()
    print("Game is over!")


def train_action_prob(env):
    """Coflow Environment
    """
    # thresholds = [1.0485760E7*(10**i) for i in range(9)]
    thresholds = np.array([10]*9)
    a_dim = env.action_space.shape[0]
    s_dim = env.observation_space.shape[0]
    a_bound = env.action_space.high

    print("a_dim:", a_dim, "s_dim:", s_dim, "a_bound:", a_bound)
    agent = DDPGProb(a_dim+1, s_dim, 1)
    oun = OUNoise(a_dim+1, mu=0)

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
    ###############################################3

    kde = KDE(prepare_pm())
    ave_rs = []

    begin_time = time.time()

    for episode in range(1, 1000):
        obs = env.reset()
        oun.reset()
        epsilon -= (epsilon/EXPLORE)

        ep_reward = 0
        mlfqs = []
        kde.print()
        kde.update()
        sentsize = []

        ep_time = time.time()
        for i in range(int(1e10)):
            ## Add exploration noise
            action_original = agent.choose_action(obs)
            # action_original = np.array(thresholds)
            # action_original = (np.random.rand(a_dim))*2-1
            if IS_OU:
                action = action_original + max(0.01, epsilon)*oun.noise()
            else:
                action = np.clip(np.random.normal(action_original, var), -1, 1)

            ## because of `tanh` activation which valued in [-1, 1], we need to scale
            step_action = action_with_softmax(kde, action)
            # obs_n, reward, done, info = env.step( makeMLFQVal(env, action) )
            obs_n, reward, done, info = env.step(step_action)
            print("episode %s step %s"%(episode, i))
            print("obs_next:", obs_n.reshape(-1, env.UNIT_DIM), "reward:", reward, "done:", done)
            print("action:", action.tolist(), "step action:", step_action, "original:",action_original.tolist())
            # mlfqs.append(info["mlfq"])
            ac = [coflow[2] for coflow in eval(info["obs"].split(":")[-1])]
            sentsize.extend(ac)
            print("active coflow:", np.array(sorted(ac)))

            agent.store_transition(obs, action, reward, obs_n)

            if agent.pointer > agent.BATCH_SIZE:
                agent.learn()
                var *= 0.9995
            
            obs = obs_n
            ep_reward += reward
            if done:
                kde.push(np.log10([e for e in sentsize if e != 0]))
                print("\nepisode %s: step %s, ep_reward %s"%(episode, i, ep_reward))
                print("MLFQs:", mlfqs)
                result = env.getResult()
                print("result: ", result, type(result))
                print("time: total-%s, episode-%s"%(get_h_m_s(time.time()-begin_time), get_h_m_s(time.time()-ep_time)))
                sys.stdout.flush()
                break
        if PERIOD_SAVE_MODEL and episode%10 == 0:
            model_name = "%s/model_%s.ckpt"%(MODEL_DIR, episode)
            agent.save(model_name)

    env.close()
    print("Game is over!")


def train_lstm(env):
    """Coflow Environment
    """
    # thresholds = [1.0485760E7*(10**i) for i in range(9)]
    thresholds = np.array([10]*9)
    a_dim = env.action_space.shape[0]
    s_dim = env.observation_space.shape[0]
    a_bound = env.action_space.high
    time_sequence = 10

    print("a_dim:", a_dim, "s_dim:", s_dim, "a_bound:", a_bound)
    # agent = DDPG(a_dim, s_dim, a_bound)
    agent = DDPG_LSTM(a_dim, s_dim, a_bound, time_sequence)
    oun = OUNoise(a_dim, mu=0)

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
    ###############################################3

    ave_rs = []

    begin_time = time.time()

    for episode in range(1, 1000):
        obs = env.reset()
        ep_reward = 0
        oun.reset()
        epsilon -= (epsilon/EXPLORE)
        ## record state in one episode
        last_s = [[0]*s_dim]*(time_sequence-1)
        last_s.append(obs)

        ep_time = time.time()
        for i in range(int(1e10)):
            ## Add exploration noise
            action_original = agent.choose_action(np.array(last_s))
            # action_original = np.array(thresholds)
            # action_original = (np.random.rand(a_dim))*2-1
            if IS_OU:
                action = action_original + max(0.01, epsilon)*oun.noise()
            else:
                action = np.clip(np.random.normal(action_original, var), -1, 1)

            ## because of `tanh` activation which valued in [-1, 1], we need to scale
            obs_n, reward, done, _ = env.step( makeMLFQVal(env, action) )
            print("episode %s step %s"%(episode, i))
            print("obs_next:", obs_n.reshape(-1, env.UNIT_DIM), "reward:", reward, "done:", done)
            print("action:", action.tolist(), "env_action:", makeMLFQVal(env, action).tolist())

            last_s_ = last_s.copy()
            del last_s_[0]
            last_s_.append(obs_n)
            agent.store_transition(np.array(last_s), action, reward, np.array(last_s_))
            # print("last_s:", last_s)
            # print("last_s_:", last_s_)
            last_s = last_s_
            # agent.store_transition(obs, action, reward, obs_n)

            if agent.pointer > agent.BATCH_SIZE:
                agent.learn()
                var *= 0.9995
            
            obs = obs_n
            ep_reward += reward
            if done:
                print("\nepisode %s: step %s, ep_reward %s"%(episode, i, ep_reward))
                result = env.getResult()
                print("result: ", result, type(result))
                print("time: total-%s, episode-%s"%(get_h_m_s(time.time()-begin_time), get_h_m_s(time.time()-ep_time)))
                sys.stdout.flush()
                break
        if PERIOD_SAVE_MODEL and episode%10 == 0:
            model_name = "%s/model_%s.ckpt"%(MODEL_DIR, episode)
            agent.save(model_name)

    env.close()
    print("Game is over!")


def config_env():
    # Configure the jpype environment
    jarpath = os.path.join(os.path.abspath("."))
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=%s/target/coflowsim-0.2.0-SNAPSHOT.jar"%(jarpath), convertStrings=False)

    java.lang.System.out.println("Hello World!")
    testfile = "./scripts/100coflows.txt"
    benchmark = "./scripts/FB2010-1Hr-150-0.txt"
    # args = ["dark", "COFLOW-BENCHMARK", benchmark] # 2.4247392E7
    # args = ["dark", "COFLOW-BENCHMARK", testfile] # 326688.0
    # args = ["dark", "COFLOW-BENCHMARK", "./scripts/test_150_250.txt"] # 1.5923608E7
    # args = ["dark", "COFLOW-BENCHMARK", "./scripts/test_150_200.txt"] # 2214624.0
    # args = ["dark", "COFLOW-BENCHMARK", "./scripts/test_200_250.txt"] # 6915640.0
    # args = ["dark", "COFLOW-BENCHMARK", "./scripts/test_200_225.txt"] # 3615440.0    
    args = ["dark", "COFLOW-BENCHMARK", "./scripts/custom.txt"] # 
    print("arguments:", args)
    CoflowGym = JClass("coflowsim.CoflowGym")
    gym = CoflowGym(args)
    return CoflowSimEnv(gym)

def destroy_env():
    shutdownJVM()

if __name__ == "__main__":
    if not os.path.exists("./log/"):
        os.mkdir("./log/")
    file = open(LOG_FILE, "w")
    sys.stdout = file

    env = config_env()

    # record
    print("training begins: %s"%(time.asctime(time.localtime(time.time()))))
    sys.stdout.flush()

    # main loop
    loop(env)
    # train_action_prob(env)
    # train_lstm(env)

    destroy_env()