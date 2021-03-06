import numpy as np 
import sys, time, os, math, json
from jpype import *
from coflowgym.coflow import CoflowSimEnv
from tf2rl.algos.ddpg import DDPG
from tf2rl.algos.gail import GAIL
from tf2rl.experiments.irl_trainer import IRLTrainer
from tf2rl.experiments.trainer import Trainer

def get_threshold():
    init = 4.5 * 1024 * 1024
    e = 10
    thresholds = [init]
    for i in range(1, 9):
        thresholds.append(thresholds[i-1]*e)

    return thresholds

def config_env(newInstance=False):
    if not newInstance:
    # Configure the jpype environment
        jarpath = os.path.join(os.path.abspath("."))
        startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=%s/target/coflowsim-0.2.0-SNAPSHOT.jar"%(jarpath), convertStrings=False)

    java.lang.System.out.println("Hello World!")
    benchmark = "./scripts/FB2010-1Hr-150-0.txt"
    args = ["dark", "COFLOW-BENCHMARK", benchmark] # 2.4247392E7
    print("arguments:", args)
    CoflowGym = JClass("coflowsim.CoflowGym")
    gym = CoflowGym(args)
    return CoflowSimEnv(gym)

def get_train_samples():
    env = config_env()
    print("Env:", env)
    thresholds = get_threshold()
    print(thresholds)
    save_in_json = False

    expert_data = {"obs":[], "obs_n":[], "act":[], "rew":[], "done":[]}
    for ep in range(1):
        obs = env.reset()
        ep_reward = 0
        for step in range(int(1e10)):
            obs_n, r, done, _ = env.step(thresholds)
            if save_in_json:
                expert_data["obses"].append(obs)
                expert_data["next_obses"].append(obs_n)
                expert_data["acts"].append(thresholds)
                expert_data["rews"].append([r])
                expert_data["done"].append([1 if done else 0])
            print("step:", step, obs, obs_n, r, done)

            ep_reward += r
            obs = obs_n
            if done:
                result, cf_info = env.getResult()
                print("ep_reward:", ep_reward)
                print("result:", result)
                # print("cf_info:", cf_info)
                break
    if save_in_json:
        for key in expert_data.keys():
            expert_data[key] = np.array(expert_data[key]).tolist()
            print(type(expert_data[key]))
        f = open("./scripts/expert.json", "w", encoding="utf-8")
        json.dump(expert_data, f)


if __name__ == "__main__":
    pass
    get_train_samples()