import matplotlib.pyplot as plt 
import numpy as np 
# import benchdata
import analyse
import util

CMAP = {
    "deep blue": "#275d90", 
    "shade blue": "#9ec2e9", 
    "deep orange": "#eb8004",
    "shade orange": "#fccba1",
    "deep green": "#33a02c",# "#438539",
    "shade green": "#b2de8a" # "#7ec87e"
}

debug=False

def get_x_y(data):
    """
    plot CDF and return x, y of CDF
    """
    hist, bin_edges = np.histogram(data, bins=100)
    # print("range of data:", min(data), max(data))
    # print(hist, bin_edges)
    x = [(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
    y = hist.cumsum()/len(data)
    return x, y


def get_heuristic_coflow(method="dark", useBin=True):
    """
    return a tuple about CCT(seconds), 
    for a given heuristic method, e.g. SEBF, Aalo, Per-Flow Fairness, DRL

    @return:
        Ave: [<25, 25-49, 50-74, >=75, all] or [sn, ln, sw, lw]
        95th: [<25, 25-49, 50-74, >=75, all] or [sn, ln, sw, lw]
        coflows: []
    """
    if method != "DRL":
        file = "doc/dark.txt"
        if method == "sebf" or method == "varys":
            file = "doc/sebf.txt"
        if method == "fair":
            file = "doc/fair.txt"
        data = analyse.dark_analyse(file, False)/1024 # transform unit to second
    else:
        data = np.array(util.best_model_log_parse()[2])/1024
    if useBin:
        sn, ln, sw, lw = [data[e] for e in util.classify_analyse()]
        ave = [sum(e)/len(e) for e in [sn, ln, sw, lw]]+[sum(data)/len(data)]
        all_95th = sorted(data)[round(len(data)*.95)-1]
        p_95th = [sorted(e)[round(len(e)*.95)-1] for e in [sn, ln, sw, lw]]+[all_95th]
        return np.array(ave), np.array(p_95th), data
    else:
        s_data = sorted(data)
        start = [0, 131, 262, 394]
        end = [131, 262, 394, 526]
        per_data = [s_data[i:j] for i, j in zip(start, end)]
        per_ave = [sum(e)/len(e) for e in per_data]
        per_95th = [e[round(len(e)*.95)-1] for e in per_data]
        all_95th = s_data[499]
        all_ave = sum(data)/len(data)
        # print(all_ave, all_95th, per_ave, per_95th)
        return np.array(per_ave+[all_ave]), np.array(per_95th+[all_95th]), data

def get_training():
    result, ep_r = benchdata.success_2_data()
    return result, ep_r

### TODO: Benchmark CDF
def plot_CDF():
    _, _, _, shuffle_t = util.parse_benchmark()
    x1, y1 = get_x_y(np.log10(shuffle_t))

    _, _, _, shuffle_t = util.parse_benchmark("scripts/light_tail.txt")
    x2, y2 = get_x_y(np.log10(shuffle_t))

    # from scipy import stats
    import seaborn as sns; sns.set_style("whitegrid")
    sns.set_palette(sns.color_palette("muted", 10)) ## set default paltte

    plt.figure("CDF")
    plt.rc("font", family="Times New Roman")
    plt.plot(x1, y1, alpha=0.9)
    plt.plot(x2, y2, alpha=0.9)

    plt.ylim([0, 1])
    plt.yticks([0, 0.5, 1], [0, 0.5, 1], fontsize=14)
    plt.xlim([0, 7])
    plt.xticks(range(0,8), [r'10$^{%s}$'%(i) for i in range(0, 8)], fontsize=14)
    # plt.text(2, 0, )
    plt.grid(linestyle="-.")

    plt.xlabel("Coflow Size(Megabytes)", fontsize=14)
    plt.ylabel("Fraction of Coflows", fontsize=14)
    plt.legend(["facebook", "light tail"], fontsize=14)

    plt.savefig("doc/paper/CDF.png")

### TODO: CCT指标对比（平均和95th）
def compare_CCT():
    ## [45.0172885 , 80.71187322, 27.85982058, 35.6570669 ] DARK / FIFO / SEBF / TARGET
    sebf = get_heuristic_coflow("sebf")
    dark = get_heuristic_coflow("dark")
    fair = get_heuristic_coflow("fair")
    drl = get_heuristic_coflow("DRL")
    # print(np.array(ave)*1024*526)

    # rate_ave = np.array(ave)/best_ave
    # rate_95th = np.array([sebf_95th, dark_95th, fair_95th])/best_95th

    sebf_ave = sebf[0]/drl[0]
    sebf_95th = sebf[1]/drl[1]
    sebf_ave, dark_ave, fair_ave = [method[0]/drl[0] for method in [sebf, dark, fair]]
    sebf_95th, dark_95th, fair_95th = [method[1]/drl[1] for method in [sebf, dark, fair]]
    print("SEBF:", sebf_ave, sebf_95th)
    print("Aalo:", dark_ave, dark_95th)
    print("Fair:", fair_ave, fair_95th)

    N = 6
    width = 1/(N+1)
    x = np.arange(len(sebf_ave))+width

    import seaborn as sns    
    sns.set_style("whitegrid", {'axes.grid' : False})
    sns.set_palette(sns.color_palette("muted", 10)) ## set default paltte
    plt.figure("Comparision CCT")
    plt.rc("font", family="Times New Roman")

    data = [sebf_ave, sebf_95th, dark_ave, dark_95th, fair_ave, fair_95th]
    colors = [CMAP["deep blue"], CMAP["shade blue"], CMAP["deep orange"],
              CMAP["shade orange"], CMAP["deep green"], CMAP["shade green"]]
    labels = ["SEBF(Avg)", "SEBF(95th)", "Aalo(Avg)", "Aalo(95th)", "Per-Fairness(Avg)", "Per-Fairness(95th)"]
    addition = [width*i for i in range(N)]
    txt = []
    YLIM = 6
    for i in [0, 2, 4]+[1, 3, 5]:
        for ei, e in enumerate(data[i]):
            if e >= YLIM:
                txt.append([(x+addition[i])[ei], e])
                data[i][ei] = YLIM
        plt.bar(x+addition[i], data[i], width=width, color=colors[i], label=labels[i])

    for x, y in txt:
        plt.text(x-0.05, 2, round(y, 2), rotation=90)

    plt.plot(plt.xlim(), [1]*2, "black")

    plt.ylim(0, 8)
    plt.yticks([1, 3, 6], [1, 3 ,6], fontsize=14)
    plt.xlim(0, 5)
    plt.xticks([i+0.5 for i in range(5)], ["Bin1", "Bin2", "Bin3", "Bin4", "All"], fontsize=14)
    plt.legend(fontsize=12, ncol=2)
    plt.ylabel("Normalized Comp.Time\nw.r.t.M-DRL", fontsize=14)
    plt.xlabel("Coflow Metric", fontsize=14)

    # plt.savefig("doc/paper/ComparisionCCT.png")

### TODO：CCT的CDF对比
def compare_CDFofCCT():
    data_sebf = analyse.dark_analyse("doc/sebf.txt", False)/1024 # transform unit to second
    x_sebf, y_sebf = get_x_y(np.log10(data_sebf))

    data_dark = analyse.dark_analyse("doc/dark.txt", False)/1024
    x_dark, y_dark = get_x_y(np.log10(data_dark))

    data_fair = analyse.dark_analyse("doc/fair.txt", False)/1024
    x_fair, y_fair = get_x_y(np.log10(data_fair))

    _, _, data_drl= get_heuristic_coflow("DRL")
    data_drl = np.array(data_drl)
    x_drl, y_drl = get_x_y(np.log10(data_drl))

    import seaborn as sns; sns.set_style("whitegrid")
    sns.set_palette(sns.color_palette("muted", 10)) ## set default paltte
    plt.figure("CDF of CCT")
    plt.rc("font", family="Times New Roman")

    plt.subplot(121)
    plt.plot(x_sebf, y_sebf)
    plt.plot(x_dark, y_dark)
    plt.plot(x_fair, y_fair)
    plt.plot(x_drl, y_drl, "black")

    plt.ylim([0, 1])
    plt.yticks([0.5, 1], [0.5, 1.0], fontsize=14)
    plt.xlim([-2, 3.5])
    plt.xticks([-2, -1, 0, 1, 2, 3], [0.01, 0.1, 1, 10, 100, 1000], fontsize=14)
    plt.grid(linestyle="-.")

    plt.xlabel("Coflow Completion Time(Seconds)", fontsize=14)
    plt.ylabel("Fraction of Coflows", fontsize=14)
    plt.legend(["SEBF", "Aalo", "Per-Flow Fairness", "M-DRL"], fontsize=12)

    plt.subplot(122)
    plt.plot(x_sebf, y_sebf)
    plt.plot(x_dark, y_dark)
    plt.plot(x_fair, y_fair)
    plt.plot(x_drl, y_drl, "black")
    e_x, e_y = util.get_ellipse(2.9, 0.9875, 0.4, .0125, 1)
    # plt.plot([2.5, 3.3], [0.975, 1], ".")
    plt.plot(e_x, e_y, color="r", linestyle="-.")
    print(plt.xlim())

    yts = [round(0.85+0.05*i, 2) for i in range(4)]
    plt.yticks(yts[1:], yts[1:], fontsize=14)
    plt.ylim([min(yts), 1])
    plt.xlim([1, 3.5])
    plt.xticks([1, 2, 3], [10, 100, 1000], fontsize=14)
    plt.grid(linestyle="-.")

    plt.xlabel("Coflow Completion Time(Seconds)", fontsize=14)
    plt.ylabel("Fraction of Coflows", fontsize=14)
    plt.legend(["SEBF", "Aalo", "Per-Flow Fairness", "M-DRL"], fontsize=14)

    # plt.savefig("doc/paper/CDF_of_CCT.png")

### TODO：训练效果
def get_train_result():
    from analyse import parse_log
    result, ep_r = parse_log(("doc/log/success-2/log/log_10.txt"))
    start, end = 1, 380
    
    x = range(start, end+1)
    data = np.array(result[:end])/1024/526
    data_sm = util.smooth_value(data, smoothing=0.9)

    import seaborn as sns; sns.set()
    sns.set_style('whitegrid')
    plt.figure("Training")
    plt.rc("font", family="Times New Roman")
    plt.plot(x, data_sm, "-", color="b")
    plt.plot(x, data, "-", alpha=0.35, color="b")

    plt.legend(["Facebook"], fontsize=14)
    plt.grid(linestyle="-.")
    plt.ylabel("Average CCT(seconds)", fontsize=14)
    plt.xlabel("Training Episodes", fontsize=14)
    plt.ylim([30, 90])
    plt.xlim([0, end])
    plt.xticks(list(range(50, 400, 50)), list(range(50, 400, 50)), fontsize=14)
    yts = range(30, 90, 10)
    plt.yticks(yts, yts, fontsize=14)

    # plt.savefig("doc/paper/Training.png")
    # ******************************************************************* #
    rs = np.array(ep_r[:end])
    normalize_rs = (rs - min(rs))/(max(rs) - min(rs))
    rs_sm = util.smooth_value(normalize_rs, smoothing=0.9)

    plt.figure("Episodic Reward")
    plt.rc("font", family="Times New Roman")
    plt.plot(x, rs_sm, "-", color="b")
    plt.plot(x, normalize_rs, "-", alpha=0.35, color="b")

    plt.legend(["Facebook"], fontsize=14)
    plt.grid(linestyle="-.")
    plt.ylabel("Normalized Reward", fontsize=14)
    plt.xlabel("Training Episodes", fontsize=14)
    # plt.ylim([30, 90])
    plt.xlim([0, end])
    plt.xticks(list(range(50, 400, 50)), list(range(50, 400, 50)), fontsize=14)
    plt.yticks(fontsize=14)

    # plt.savefig("doc/paper/ep_reward.png")

### TODO: 问题提出的CDF图
def raise_question():
    data = util.prepare_pm()
    queues = [0 for _ in range(10)]
    for size in data:
        if size < 7:
            q = 0
        elif size >= 15:
            q = 9
        else:
            q = int(size) - 6
        queues[q] += 1
    
    x = range(10)
    q_count = np.array(queues)/sum(queues)
    print("Percentile in Aalo:", q_count)
    xlabels = ["Q%s"%(i) for i in range(10)]

    import seaborn as sns    
    sns.set_style("whitegrid", {'axes.grid' : False})
    sns.set_palette(sns.color_palette("muted", 10)) ## set default paltte
    plt.figure("Question Raising")
    plt.rc("font", family="Times New Roman")
    plt.bar(x, q_count, width=0.9, color=CMAP["deep blue"])
    for xi, yi in zip(x, q_count):
        plt.text(xi, yi, round(yi, 2), horizontalalignment='center', verticalalignment='bottom', fontdict={'size':14}, rotation=0)
    plt.xticks(x, xlabels, fontsize=14)
    plt.yticks([0, 0.2, 0.4, 0.6], [0, 0.2, 0.4, 0.6], fontsize=14)
    plt.ylabel("Fraction of Active Coflows", fontsize=14)
    plt.xlabel("Multi-Level Feedback Queue", fontsize=14)

### TODO: 问题验证图
def validate_question():
    bmk_sentsize = util.prepare_pm()
    # print(sorted(bmk_sentsize))
    bmk_thresholds = np.arange(7, 16)
    bmk_queue = [[] for _ in range(10)]
    for size in bmk_sentsize:
        q = int(size)
        if q >= 7 and q < 15:
            q = q - 6
        elif q < 7:
            q = 0
        else:
            q = 9
        # print(q)
        bmk_queue[q].append(size)
    bmk_count = [len(e)/len(bmk_sentsize) for e in bmk_queue]

    actions, sentsize, _, _ = util.best_model_log_parse("doc/log/success-2/best_run_log.txt")
    count = [0 for _ in range(10)]
    for action, sent in zip(actions, sentsize):
        for size in sent:
            i = 0
            while i < 9 and size >= action[i]:
                i += 1
            count[i] += 1
    total = sum([len(e) for e in sentsize])
    percetile = np.array(count)/total
    print("Percetile of DRL:", percetile)

    import seaborn as sns    
    sns.set_style("whitegrid", {'axes.grid' : False})
    sns.set_palette(sns.color_palette("muted", 10)) ## set default paltte
    plt.figure("Question Validation")
    plt.rc("font", family="Times New Roman")
    
    plt.plot(range(10), bmk_count, color=CMAP["deep blue"], marker="o", linestyle="--")
    plt.plot(range(10), percetile, color=CMAP["deep orange"], marker="s", linestyle="--")

    plt.xticks(list(range(10)), ["Q%s"%(i) for i in range(1, 11)], fontsize=14)
    yts = [round(0.1*i,2) for i in range(7)]
    plt.yticks(yts, yts, fontsize=14)
    plt.legend(["Aalo", "M-DRL"], fontsize=14)
    plt.ylabel("Fraction of Active Coflows", fontsize=14)
    plt.xlabel("Multi-Level Feedback Queue", fontsize=14)

def test():
    pass
    x, y = util.get_ellipse(0, 0, 2, 1, 45)
    plt.plot(x, y)
    plt.grid()
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xticks(range(-5, 5), range(-5, 5))
    plt.yticks(range(-5, 5), range(-5, 5))

if __name__ == "__main__":

    # test()
    pass

    plot_CDF()

    # compare_CDFofCCT()

    # compare_CCT()

    # get_train_result()

    # raise_question()

    # validate_question()

    plt.show()