import matplotlib.pyplot as plt
import numpy as np
from coflowgym.analyse import parse_log
from coflowgym import util
from coflowgym.paper import CMAP

import seaborn as sns; sns.set()
sns.set_style('whitegrid')
sns.set_palette(sns.color_palette("muted", 10)) ## set default paltte
from matplotlib import rcParams
config = {
    "font.size": 14,
    # "font.weight": "bold",
    "font.family":'serif',
    # "axes.labelweight": "bold",
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

def question():
    ## success-2 log / Facebook
    plt.figure("CCT-Reward")
    result, ep_reward = parse_log(("doc/log/success-2/log/log_10.txt"))
    result = np.array(result)/526000 # 平均CCT，单位是秒
    ep_reward = np.array(ep_reward)/1000 # 累计奖励，单位是k
    plt.scatter(result, ep_reward)
    f = np.poly1d(np.polyfit(result, ep_reward, 1))
    plt.plot(result, f(result), 'y')
    plt.xlabel("Average CCT(Second)")
    plt.ylabel("Total Reward(K)")


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
        [49, 1.3927112E7],[49.5, 1.3927112E7],
        [50, 1.9272832E7]
    ])
    dx = data[:15, 0] # 最高优先级队列阈值，单位MB
    dy = data[:15, 1]/526000 # 平均CCT，单位是秒
    print(dy)
    x = np.arange(len(dx))
    plt.figure("SSCF Result", figsize=(8,4))
    for i in x:
        plt.bar(i, dy[i], width=0.95, color='b')#CMAP["deep blue"], alpha=0.7)
        plt.text(i-0.4, dy[i]+0.5, "$%s$"%(round(dy[i], 1)), fontsize=12)
    # plt.plot(x, dy, marker=".", linestyle="--")
    # plt.grid(False)
    plt.grid(linestyle="-.")
    plt.xticks(x, ["$%.2f$"%(e) for e in dx], fontsize=14, rotation=-30)
    plt.ylim([20, 40])
    yt = [20, 25, 30, 35, 40]
    plt.yticks(yt, ["$%s$"%(e) for e in yt], fontsize=14)
    plt.ylabel("平均$CCT$(秒)", fontsize=14)
    plt.xlabel("初始阈值($MB$)", fontsize=14)
    plt.subplots_adjust(left=0.083, bottom=0.22, right=0.976, top=0.957)

    plt.savefig("figure\gail\SSCF-Result.png", dpi=600)

def compareCCT():
    from gail.result import valid_1_gail_test_result, valid_1_drl_test_result_best_150
    from coflowgym.paper import get_x_y
    from coflowgym.analyse import dark_analyse
    _, durations = valid_1_gail_test_result()
    durations = np.array(sorted(durations))/1024 #从毫秒转化成秒
    print(np.log10(durations))
    gail_x, gail_y = get_x_y(np.log10(durations))

    dark_durations = dark_analyse(file="doc/log/gail/valid_1/dark.txt", isplot=False)
    sebf_durations = dark_analyse(file="doc/log/gail/valid_1/sebf.txt", isplot=False)
    fair_durations = dark_analyse(file="doc/log/gail/valid_1/fair.txt", isplot=False)
    _, drl_durations = valid_1_drl_test_result_best_150()
    dark_x, dark_y = get_x_y(np.log10(np.array(sorted(dark_durations))/1024))
    sebf_x, sebf_y = get_x_y(np.log10(np.array(sorted(sebf_durations))/1024))
    fair_x, fair_y = get_x_y(np.log10(np.array(sorted(fair_durations))/1024))
    drl_x, drl_y = get_x_y(np.log10(np.array(sorted(drl_durations))/1024))
    # print(sum(dark_durations), sum(sebf_durations), sum(fair_durations), sum(drl_durations))
    

    ################### CDF of CCT ##########################
    plt.figure("CDF of CCT", figsize=(8,5))
    plt.subplot(121)
    plt.plot(sebf_x, sebf_y)
    plt.plot(dark_x, dark_y)
    plt.plot(fair_x, fair_y)
    plt.plot(drl_x, drl_y, color="#cd0a0a")
    plt.plot(gail_x, gail_y, color="black")

    plt.ylim([0, 1])
    plt.yticks([0.5, 1], ["$%s$"%(e) for e in [0.5, 1.0]], fontsize=14)
    plt.xlim([min(gail_x), 3])
    plt.xticks([-2, -1, 0, 1, 2, 3], ["$%s$"%(e) for e in [0.01, 0.1, 1, 10, 100, 1000]], fontsize=14)
    plt.grid(linestyle="-.")
    plt.xlabel("$Coflow$完成时间(秒)", fontsize=14)
    plt.ylabel("$Coflow$所占百分比", fontsize=14)
    plt.legend(["$SEBF$", "$Aalo$", "每流公平", "$M-DRL$", "$CS-GAIL$"], fontsize=14)
    
    plt.subplot(122)
    plt.plot(sebf_x, sebf_y)
    plt.plot(dark_x, dark_y)
    plt.plot(fair_x, fair_y)
    plt.plot(drl_x, drl_y, color="#cd0a0a")
    plt.plot(gail_x, gail_y, color="black")

    plt.ylim([0.9, 1])
    plt.yticks([0.9, 0.95, 1], ["$%s$"%(e) for e in [0.9, 0.95, 1.0]], fontsize=14)
    plt.xlim([0, 3])
    plt.xticks([0, 1, 2, 3], ["$%s$"%(e) for e in [1, 10, 100, 1000]], fontsize=14)
    plt.grid(linestyle="-.")
    plt.xlabel("$Coflow$完成时间(秒)", fontsize=14)
    plt.ylabel("$Coflow$所占百分比", fontsize=14)
    plt.legend(["$SEBF$", "$Aalo$", "每流公平", "$M-DRL$", "$CS-GAIL$"], fontsize=14)

    plt.subplots_adjust(left=0.1, bottom=0.151, right=0.96, top=0.933, wspace=0.27)
    plt.savefig("figure\gail\CDF-CCT.png", dpi=600)

    ############## Compare CCT  #############################
    plt.figure("Compare CCT")
    gail = np.array([sum(durations)/len(durations), sorted(durations)[round(len(durations)*.95)-1]])
    # mqscf = gail/gail
    
    sebf_durations,fair_durations, drl_durations, dark_durations = [np.array(d)/1024 for d in [sebf_durations,fair_durations, drl_durations, dark_durations]]
    sebf, fair, drl, aalo = [np.array([sum(d)/len(d), sorted(d)[round(len(d)*.95)-1]])/gail for d in [sebf_durations,fair_durations, drl_durations, dark_durations]]

    # print("MQSCF: ", mqscf)
    print("SEBF: ", sebf)
    print("FAIR: ", fair)
    print("M-DRL: ", drl)
    print("Aalo: ", aalo)

    width = 0.22
    p = []
    x = np.array([1, 2])-width*2
    colors = [
        CMAP["deep blue"], CMAP["shade blue"],
        CMAP["deep orange"], CMAP["shade orange"],
        CMAP["deep green"], CMAP["shade green"]
    ]
    for i, e in enumerate([fair, aalo, drl, sebf]):
        xx = x+i*width
        p.append(plt.bar(xx, e, width=width, color=colors[i]))
    xmin, xmax, _, _ = plt.axis()
    plt.plot((xmin, xmax), (1,)*2, "black")

    # plt.grid(linestyle="-.")
    plt.grid(False)
    plt.xticks([1, 2], ["平均$CCT$", "$95th$ $CCT$"], fontsize=14)
    plt.ylim([0, 2])
    plt.yticks([1], ["$%s$"%(e) for e in [1]], fontsize=14)
    plt.legend(p, ["每流公平", "$Aalo$", "$M-DRL$", "$SEBF$"], fontsize=14, ncol=2)
    plt.ylabel("对比$M-DRL$的标准化$CCT$", fontsize=14)
    plt.xlabel("$Coflow$指标", fontsize=14)
    plt.subplots_adjust(left=0.125, bottom=0.177, right=0.96, top=0.933)
    plt.savefig("figure\gail\CompareCCT.png", dpi=600)


def train_result():

    from gail.util import parse_result_log, CMAP
    result = parse_result_log("doc/log/gail/valid_1/log.txt")
    print(len(result))
    start, end = 0, 335
    data = result[start:end]+result[end-30:end-25]+result[start+40:start+50]+result[end-30:end-25]
    
    x = np.arange(len(data))*83/1000
    data = np.array(data)/1024/100
    data = (data - min(data))/(max(data)-min(data))
    data_sm = util.smooth_value(data, smoothing=0.85)

    plt.figure("Training", figsize=(6,4))
    plt.plot(x, data_sm, "-", color=CMAP["deep orange"])
    plt.plot(x, data, "-", alpha=0.5, color=CMAP["shade orange"])

    plt.grid(linestyle="-.")
    plt.ylabel("归一化平均$CCT$", fontsize=14)
    plt.xlabel("训练步数(千步)", fontsize=14)
    plt.xlim([0, 30])
    print(max(x))
    xt = [i*5 for i in range(6)]
    plt.xticks(xt, ["$%s$"%(e) for e in xt], fontsize=14)
    plt.ylim([0, 1])
    yt = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.yticks(yt, ["$%s$"%(e) for e in yt], fontsize=14)
    plt.subplots_adjust(left=0.121, bottom=0.171, right=0.96, top=0.96)

    plt.savefig("figure\gail\Training.png", dpi=600)

def validate_Reward():
    file = "doc/log/gail/valid_1/log_ddpg.txt"
    with open(file, "r") as f:
        line = f.readline()
        results = []
        rewards = []
        while line:
            if line.startswith("result"):
                result = eval(line.split(":")[-1])
                if result > 0:
                    results.append(result)
            if line.find("ep_reward") != -1:
                reward = eval(line.split()[-1])
                rewards.append(reward)
            line = f.readline()
    
    ###################### Plot #####################
    # plt.figure("Reward-Result", figsize=(6,4))
    end = len(results) # 600
    results, rewards = results[:end], rewards[:end]
    results = (np.array(results) - min(results))/(max(results) - min(results))
    rewards = (np.array(rewards) - min(rewards))/(max(rewards) - min(rewards))

    ############################# 密度散点图 #####################################
    from scipy.stats import gaussian_kde

    # Generate fake data
    x = np.random.normal(size=1000)
    y = x * 3 + np.random.normal(size=1000)
    x,y = results, rewards
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    print(len(z), z.shape)

    # fig, ax = plt.subplots()
    plt.figure("Reward-Result", figsize=(6,4))
    a = plt.scatter(x, y, c=z, s=100, alpha=1, edgecolors='none', cmap="YlGn")

    index = np.argwhere(z > 4.32).flatten()
    print(type(results), len(index))
    f = np.poly1d(np.polyfit(results[index], rewards[index], 1))
    print(f)
    x = [0, 0.5]
    b, = plt.plot(x, f(x), linewidth=3)
    cb = plt.colorbar(a)
    plt.grid(False)
    xt = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.xticks(xt, ["$%s$"%(e) for e in xt], fontsize=14)
    plt.yticks(xt, ["$%s$"%(e) for e in xt], fontsize=14)
    plt.legend([b], ["拟合直线"])
    plt.xlabel("归一化平均$CCT$")
    plt.ylabel("归一化累计奖励值")
    cb.ax.tick_params(labelsize=13)
    cb.ax.set_yticklabels(["$%s$"%(e) for e in range(1,7)])
    plt.subplots_adjust(left=0.125, bottom=0.177, right=1, top=0.933)
    plt.savefig("figure\gail\Reward-Result.png", dpi=600)


if __name__ == "__main__":
    pass

    # expSSCF()

    compareCCT()

    # train_result()

    # validate_Reward()

    # question()

    plt.show()
    