import numpy as np 
import matplotlib.pyplot as plt
import codecs
import sys


def parse_log(file):
    """
    Parse log file to get training information.
    @return:
        result: runtime of benchmark in a episode
        ep_reward: total rewards accumulated in a episode
    """
    with open(file, 'r') as f:
        result = []
        ep_reward = []

        line = f.readline()
        while line:
            if line.startswith("result:"):
                res = eval(line.split()[1])
                # print(res)
                result.append(res)

            line = f.readline()
            if line.find("ep_reward") != -1:
                rs = eval(line.split()[-1])
                # print(rs)
                ep_reward.append(rs)
        return result, ep_reward

def plot_compare(result, ep_reward, newfigure=True):
    if newfigure:
        plt.figure()
    x = list(range(len(result)))
    plt.plot(x, result, 'bo-')
    # plt.scatter(x, result)
    plt.plot(x, [2.4247392E7]*len(x), "red") # DARK
    plt.plot(x, [4.3473352E7]*len(x), "cyan") # FIFO
    plt.plot(x, [1.5005968E7]*len(x), "lawngreen") # SEBF
    plt.legend(["DRL", "DARK", "FIFO", "SEBF"])
    plt.xlabel("episode")
    plt.ylabel("ep_runtime")

    # plt.figure()
    # plt.plot(x, [-r for r in ep_reward])
    # plt.scatter(x, [-r for r in ep_reward])

def validate_reward(result, ep_reward, newfigure=True):
    if newfigure:
        plt.figure()
    plt.scatter(result, ep_reward)
    plt.xlabel("ep_runtime")
    plt.ylabel("ep_total_reward")

def analyse_log():
    exp_no = -1

    if exp_no == 1:
        res1, ep_r1 = parse_log("doc/log/1_log.txt")
        res2, ep_r2 = parse_log("doc/log/2_log.txt")
        print("Number of samples in log_1", len(res1))
        print("Number of samples in log_2", len(res2))
        result, ep_reward = np.array(res1+res2), np.array(ep_r1+ep_r2)
        validate_reward(result[result < 2.5*1e7], ep_reward[result < 2.5*1e7])
    if exp_no is 2:
        result, ep_reward = parse_log(("doc/log/3_log.txt"))
        plt.subplot(211)
        validate_reward(result, ep_reward, False)
        plt.subplot(212)
        plot_compare(result, ep_reward, False)

    ## 
    if exp_no is -1:
        result, ep_reward = parse_log(("log/log.txt"))
        print(result)
        print(ep_reward)

        validate_reward(result, ep_reward)
        plot_compare(result, ep_reward)

if __name__ == "__main__":
    # analyse_log
    analyse_log()
    plt.show()