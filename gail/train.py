import numpy as np 
import sys, time, os, math, json
from jpype import *
from coflowgym.coflow import CoflowSimEnv, CoflowKDEEnv

from lib.tf2rl.tf2rl.algos.ddpg import DDPG
from lib.tf2rl.tf2rl.algos.gail import GAIL
from lib.tf2rl.tf2rl.experiments.irl_trainer import IRLTrainer
from lib.tf2rl.tf2rl.experiments.trainer import Trainer

# from tf2rl.algos.ddpg import DDPG
# from tf2rl.algos.gail import GAIL
# from tf2rl.experiments.irl_trainer import IRLTrainer
# from tf2rl.experiments.trainer import Trainer

def config_env(newInstance=False, test=False):
    if not newInstance:
    # Configure the jpype environment
        jarpath = os.path.join(os.path.abspath("."))
        startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=%s/target/coflowsim-0.2.0-SNAPSHOT.jar"%(jarpath), convertStrings=False)

    java.lang.System.out.println("Hello World!")
    benchmark = "./scripts/FB2010-1Hr-150-0.txt" # 2.4247392E7
    benchmark = "./scripts/valid_1.txt"
    args = ["dark", "COFLOW-BENCHMARK", benchmark] 
    print("arguments:", args)
    CoflowGym = JClass("coflowsim.CoflowGym")
    gym = CoflowGym(args)
    return CoflowKDEEnv(gym, isTest=test)

def gail_train():
    parser = IRLTrainer.get_argument()
    parser = GAIL.get_argument(parser)
    parser.add_argument("--logdir", type=str, default="log/results")
    args = parser.parse_args()

    env = config_env()
    test_env = config_env(True, test=True)
    units = [400, 300]
    policy = DDPG(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        max_action=env.action_space.high[0],
        gpu=-1, # -1 is only cpu
        actor_units=units,
        critic_units=units,
        n_warmup=10, # default is 10000
        batch_size=100)
    irl = GAIL(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        units=units,
        enable_sn=args.enable_sn,
        batch_size=32,
        gpu=-1) # -1 is only cpu
    obs, act, obs_n = read_expert_trajs()
    trainer = IRLTrainer(policy, env, args, irl, obs, obs_n, act, test_env)
    trainer()


def read_expert_trajs():
    f = open("scripts/expert_1.json", "r")
    data = json.load(f)
    return np.array(data["obs"]), np.array(data["act"]), np.array(data["obs_n"])

if __name__ == "__main__":
    pass
    # obses, acts, obs_ns = read_expert_trajs()
    # print(acts.shape, acts[:10])

    gail_train()
