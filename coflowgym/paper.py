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

### TODO: 制作coflow数据的表格：SN、LN、SW、LW
def makeTableData():
    def oneData(file="scripts/FB2010-1Hr-150-0.txt"):
        print("using", file)
        bin_index = util.classify_analyse(file)
        _, _, _, shuffle= util.parse_benchmark(file)
        bin_c_num = [len(e) for e in bin_index]
        bin_c_percentile = np.array(bin_c_num)/sum(bin_c_num)
        print("Percentile of Coflows:", ["%s%%"%(round(e, 2)*100) for e in bin_c_percentile])
        data = np.array(shuffle)
        bin_c_megabyte = [sum(data[e]) for e in bin_index]
        bin_byte_per = np.array(bin_c_megabyte)/sum(bin_c_megabyte)
        print("Percentile of Bytes:", ["%s%%"%(round(e*100, 2)) for e in bin_byte_per])
    
    oneData()
    oneData("scripts/light_tail.txt")

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
    plt.legend(["Facebook", "LightTail"], fontsize=14)

    # plt.savefig("doc/paper/CDF.png")

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

    max_x_val = [max(np.log10(data)) for data in [data_sebf, data_dark, data_fair, data_drl]]
    print("Max-X-axis:", max_x_val)

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
    plt.plot(e_x, e_y, color="red", linestyle="-.")

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
    
    x = np.arange(start, end+1)*450
    data = np.array(result[:end])/1024/526
    data_sm = util.smooth_value(data, smoothing=0.9)

    import seaborn as sns; sns.set()
    sns.set_style('whitegrid')
    plt.figure("Training")
    plt.rc("font", family="Times New Roman")
    plt.plot(x, data_sm, "-", color=CMAP["deep orange"])
    plt.plot(x, data, "-", alpha=0.5, color=CMAP["shade orange"])

    # plt.legend(["Facebook"], fontsize=14)
    plt.grid(linestyle="-.")
    plt.ylabel("Average CCT(Seconds)", fontsize=14)
    plt.xlabel("Training Steps(Thousands)", fontsize=14)
    plt.ylim([30, 85])
    plt.xlim([0, end*450])
    # plt.xticks(list(range(50, 400, 50)), list(range(50, 400, 50)), fontsize=14)
    xts = np.arange(25, 175, 25)
    plt.xticks(xts*1000, xts, fontsize=14)
    yts = range(30, 90, 10)
    plt.yticks(yts, yts, fontsize=14)

    # plt.savefig("doc/paper/Training.png")
    # ******************************************************************* #
    rs = np.array(ep_r[:end])
    normalize_rs = (rs - min(rs))/(max(rs) - min(rs))
    rs_sm = util.smooth_value(normalize_rs, smoothing=0.9)

    plt.figure("Episodic Reward")
    plt.rc("font", family="Times New Roman")
    plt.plot(x, rs_sm, "-", color=CMAP["deep orange"])
    plt.plot(x, normalize_rs, "-", alpha=0.5, color=CMAP["shade orange"])

    # plt.legend(["Facebook"], fontsize=14)
    plt.grid(linestyle="-.")
    plt.ylabel("Normalized Reward", fontsize=14)
    plt.xlabel("Training Steps(Thousands)", fontsize=14)
    plt.xlim([0, end*450])
    plt.xticks(xts*1000, xts, fontsize=14)
    plt.ylim([0, 1])
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
    xlabels = ["Q%s"%(i+1) for i in range(10)]

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
    
    # plt.plot(range(10), bmk_count, color=CMAP["deep blue"], marker="o", linestyle="--")
    # plt.plot(range(10), percetile, color=CMAP["deep orange"], marker="s", linestyle="--")
    width = 0.45
    x = np.arange(1-width/2, 10, 1)
    plt.bar(x, bmk_count, width=width, color=CMAP["deep blue"])
    plt.bar(x+width, percetile, width=width, color=CMAP["deep orange"])
    for i, xi in enumerate(x):
        plt.text(xi, bmk_count[i], round(bmk_count[i], 2), horizontalalignment='center', verticalalignment='bottom', fontdict={'size':12}, rotation=0)
        plt.text(xi+width, percetile[i], round(percetile[i], 2), horizontalalignment='center', verticalalignment='bottom', fontdict={'size':12}, rotation=0)

    plt.xticks(list(range(1, 1+10)), ["Q%s"%(i) for i in range(1, 11)], fontsize=14)
    yts = [round(0.1*i,2) for i in range(7)]
    plt.yticks(yts, yts, fontsize=14)
    plt.legend(["Aalo", "M-DRL"], fontsize=14)
    plt.ylabel("Fraction of Active Coflows", fontsize=14)
    plt.xlabel("Multi-Level Feedback Queue", fontsize=14)

def light_tail():
    def get_ave_and_95th(data):
        data = sorted(data)
        ave = sum(data)/len(data)
        position = int(len(data)*.95)-1
        return data[position], ave

    data_sebf = analyse.dark_analyse("doc/log/lighttail/sebf.txt", False)/1024 # transform unit to second
    x_sebf, y_sebf = get_x_y(np.log10(data_sebf))

    data_dark = analyse.dark_analyse("doc/log/lighttail/dark.txt", False)/1024
    x_dark, y_dark = get_x_y(np.log10(data_dark))

    data_fair = analyse.dark_analyse("doc/log/lighttail/fair.txt", False)/1024
    x_fair, y_fair = get_x_y(np.log10(data_fair))

    _, _, data_drl, _ = util.best_model_log_parse("doc/log/lighttail/best_run_log.txt")
    data_drl = np.array(data_drl)/1024
    x_drl, y_drl = get_x_y(np.log10(data_drl))

    max_x_val = [max(np.log10(data)) for data in [data_sebf, data_dark, data_fair, data_drl]]
    min_x_val = [min(np.log10(data)) for data in [data_sebf, data_dark, data_fair, data_drl]]
    print("Min-X-axis:", min(min_x_val), " Max-X-axis:", max(max_x_val))

    import seaborn as sns; sns.set_style("whitegrid")
    sns.set_palette(sns.color_palette("muted", 10)) ## set default paltte
    plt.figure("CDF of CCT in LightTail")
    plt.rc("font", family="Times New Roman")

    plt.plot(x_sebf, y_sebf)
    plt.plot(x_dark, y_dark)
    plt.plot(x_fair, y_fair)
    plt.plot(x_drl, y_drl, "black")

    plt.ylim([0, 1])
    plt.yticks([0.5, 1], [0.5, 1.0], fontsize=14)
    plt.xlim([0, 4.5])
    plt.xticks([0, 1, 2, 3, 4], [1, 10, 100, 1000, 10000], fontsize=14)
    plt.grid(linestyle="-.")

    plt.xlabel("Coflow Completion Time(Seconds)", fontsize=14)
    plt.ylabel("Fraction of Coflows", fontsize=14)
    plt.legend(["SEBF", "Aalo", "Per-Flow Fairness", "M-DRL"], fontsize=12)

    ## ************************ Compare CCT  **************************** ##
    sebf = np.array(get_ave_and_95th(data_sebf))
    dark = np.array(get_ave_and_95th(data_dark))
    fair = np.array(get_ave_and_95th(data_fair))
    drl = np.array(get_ave_and_95th(data_drl))
    
    print("Improvement:", sebf/drl, dark/drl, fair/drl)
    plt.figure("Compare CCT in LightTail")
    plt.grid(False)
    width = 0.3
    p = []
    x = np.array([1, 2])-width
    colors = [
        [CMAP["deep blue"], CMAP["shade blue"]],
        [CMAP["deep orange"], CMAP["shade orange"]],
        [CMAP["deep green"], CMAP["shade green"]]
    ]
    for i, e in enumerate([sebf/drl, dark/drl, fair/drl]):
        xx = x+i*width
        p.append(plt.bar(xx, e, width=width, color=colors[i][0]))
        # for xi, val in zip(xx, e):
        #     plt.text(xi, val, round(val, 2), horizontalalignment='center', verticalalignment='bottom', fontdict={'size':12}, rotation=0)
    plt.plot(plt.xlim(), [1,1], "black")

    plt.xticks([1, 2], ["Average", "95th"], fontsize=14)
    plt.ylim([0, 1.8])
    plt.yticks([1], [1], fontsize=14)
    plt.legend(p, ["SEBF", "Aalo", "Per-Flow Fairness"], fontsize=13)#, ncol=3)
    plt.ylabel("Normalized Comp.Time\nw.r.t.M-DRL", fontsize=14)
    plt.xlabel("Coflow Metric", fontsize=14)

    ## **************** get training result ************************* ##
    import benchdata
    result, ep_r = benchdata.light_tail_data()
    start, end = 1, 220
    
    x = np.arange(start, end+1)*2700
    data = np.array(result[:end])/1024/100
    data_sm = util.smooth_value(data, smoothing=0.9)

    plt.figure("Training in LightTail")
    plt.rc("font", family="Times New Roman")
    plt.plot(x, data_sm, "-", color=CMAP["deep orange"])
    plt.plot(x, data, "-", alpha=0.5, color=CMAP["shade orange"])

    plt.grid(linestyle="-.")
    plt.ylabel("Average CCT(Seconds)", fontsize=14)
    plt.xlabel("Training Steps(Thousands)", fontsize=14)
    plt.xlim([0, end*2700])
    xts = np.array(range(100, 600, 100))
    plt.xticks(xts*1000, xts, fontsize=14)
    # plt.ylim([30, 85])
    yts = np.array(range(3000, 5500, 500))
    plt.yticks(yts, ["%sk"%(round(e/1000, 1)) for e in yts], fontsize=14)

    # ******************************************************************* #
    rs = np.array(ep_r[:end])
    normalize_rs = (rs - min(rs))/(max(rs) - min(rs))
    rs_sm = util.smooth_value(normalize_rs, smoothing=0.9)

    plt.figure("Episodic Reward in LightTail")
    plt.rc("font", family="Times New Roman")
    plt.plot(x, rs_sm, "-", color=CMAP["deep orange"])
    plt.plot(x, normalize_rs, "-", alpha=0.5, color=CMAP["shade orange"])

    plt.grid(linestyle="-.")
    plt.ylabel("Normalized Reward", fontsize=14)
    plt.xlabel("Training Steps(Thousands)", fontsize=14)
    plt.xlim([0, end*2700])
    xts = np.array(range(100, 600, 100))
    plt.xticks(xts*1000, xts, fontsize=14)
    plt.ylim([0, 1])
    plt.yticks(fontsize=14)

def test():
    pass
    plt.bar([1, 2], [1,2])

if __name__ == "__main__":

    # test()
    pass

    # makeTableData()
    # print()

    # plot_CDF()
    # print()

    # compare_CDFofCCT()
    # print()

    # compare_CCT()
    # print()

    # get_train_result()
    # print()

    # raise_question()
    # print()

    # validate_question()
    # print()

    light_tail()

    plt.show()