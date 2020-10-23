import matplotlib.pyplot as plt
import numpy as np
from coflowgym.analyse import parse_log

def question():
    ## success-2 log / Facebook
    plt.figure("CCT-Reward")
    result, ep_reward = parse_log(("doc/log/success-2/log/log_10.txt"))
    result = np.array(result)/526000 # 平均CCT，单位是秒
    ep_reward = np.array(ep_reward)/1000 # 累计奖励，单位是k
    plt.scatter(result, ep_reward)
    f = np.poly1d(np.polyfit(result, ep_reward, 1))
    plt.plot(result, f(result), 'y')
    plt.xlabel("Average CCT(second)")
    plt.ylabel("Total Reward(k)")


def expSSCF():
    data = np.array([
        [1, 1.892824E7], [2, 1.4671248E7], [3, 1.432372E7],
        [4, 1.3928736E7], [4.25, 1.392728E7], 
        [4.5, 1.3926672E7], [4.75, 1.3926672E7],
        [4.8, 1.3926784E7], [4.9, 1.3926784E7],
        [4.95, 1.3926784E7], [4.975,  1.3926784E7], 
        [5, 1.9272504E7], [5.25,  1.9270152E7], 
        [5.5, 1.9270144E7], [7.5, 1.945452E7],
        [10, 1.8928288E7], [30, 1.4324048E7],
        [40, 1.3929064E7], [45, 1.3927E7],
        [49, 1.3927112E7],
        [50, 1.9272832E7]
    ])
    x = data[:, 0]
    y = data[:, 1]
    plt.plot(x, y, "x-")

if __name__ == "__main__":
    pass

    expSSCF()

    question()

    plt.show()
    