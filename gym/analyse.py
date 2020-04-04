import numpy as np 
import matplotlib.pyplot as plt
import codecs
import sys

def analyse_log(file):
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

    ## 
    print(result)
    print(ep_reward)
    # validate_reward(result, ep_reward)
    plot_compare(result, ep_reward)

def plot_compare(result, ep_reward):
    plt.figure()
    x = list(range(len(result)))
    plt.plot(x, result, 'bo-')
    # plt.scatter(x, result)
    plt.plot(x, [2.4247392E7]*len(x), "red") # DARK
    plt.plot(x, [4.3473352E7]*len(x), "cyan") # FIFO
    plt.plot(x, [1.5005968E7]*len(x), "lawngreen") # SEBF
    plt.legend(["DRL", "DARK", "FIFO", "SEBF"])

    # plt.figure()
    # plt.plot(x, [-r for r in ep_reward])
    # plt.scatter(x, [-r for r in ep_reward])

def validate_reward(result, ep_reward):
    # com_r = []
    # for i in range(len(result)):
    #     com_r.append([result[i], ep_reward[i]])
    # print(com_r)
    # com_r = sorted(com_r, key=lambda x: (x[0], x[1]))
    # print(com_r)
    plt.figure()
    plt.scatter(result, ep_reward)

if __name__ == "__main__":
    pass
    analyse_log("log_3.txt")
    plt.show()