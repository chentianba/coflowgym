import matplotlib.pyplot as plt 
import numpy as np 
import benchdata, analyse
import util

def get_x_y(data):
    hist, bin_edges = np.histogram(data, bins=100)
    # print("range of data:", min(data), max(data))
    # print(hist, bin_edges)
    x = [(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
    y = hist.cumsum()/len(data)
    # print(y)
    return x, y

def calculate_95th():
    """
    return 95th percentile in SEBF, Aalo, Per-Flow Fairness
    """
    data_sebf = analyse.dark_analyse("doc/sebf.txt", False)/1024 # transform unit to second
    sebf = sorted(data_sebf)[499]
    # print(sorted(data_sebf))
    dark = sorted(analyse.dark_analyse("doc/dark.txt", False)/1024)[499]
    fair = sorted(analyse.dark_analyse("doc/fair.txt", False)/1024)[499]
    print(sebf, dark, fair)
    return {"sebf": sebf, "aalo":dark, "fairness":fair}

### TODO: Benchmark CDF
def plot_CDF():
    _, _, _, shuffle_t = util.parse_benchmark()
    x1, y1 = get_x_y(np.log10(shuffle_t))

    _, _, _, shuffle_t = util.parse_benchmark("scripts/light_tail.txt")
    x2, y2 = get_x_y(np.log10(shuffle_t))

    # from scipy import stats
    import seaborn as sns; sns.set_style("whitegrid")

    plt.figure("CDF")
    plt.rc("font", family="Times New Roman")
    plt.plot(x1, y1, alpha=0.9)
    plt.plot(x2, y2, alpha=0.9)

    plt.ylim([0, 1])
    plt.yticks([0, 0.5, 1], [0, 0.5, 1])
    plt.xlim([0, 7])
    plt.xticks(range(1,8), [r'10$^{%s}$'%(i) for i in range(1, 8)])
    # plt.text(2, 0, )
    plt.grid(linestyle="-.")

    plt.xlabel("Coflow Size(Megabytes)", fontsize=14)
    plt.ylabel("Fraction of Coflows", fontsize=14)
    plt.legend(["facebook", "light tail"], fontsize=12)

    plt.savefig("doc/paper/CDF.png")

### TODO: CCT指标对比（平均和95th）
def compare_CCT():
    bmk_ave = {
        "varys": 1.5005968E7,
        "fairness": 3.7131824E7,
        "aalo": 2.4247392E7,
        "fifo": 4.3473352E7
    }
    best_ave = 1.8377914E7
    bmk_95th = None
    best_95th = None

    width = 0.2
    x = np.array([width*3/2, width*3/2+1])
    varys = [bmk_ave["varys"]/best_ave, bmk_ave["varys"]/best_ave]
    fair = [bmk_ave["fairness"]/best_ave, bmk_ave["fairness"]/best_ave]
    aalo = [bmk_ave["aalo"]/best_ave, bmk_ave["aalo"]/best_ave]

    import seaborn as sns    
    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.figure("Comparision CCT")
    plt.rc("font", family="Times New Roman")
    plt.bar(x, varys, label="Varys", width=width)
    plt.bar(x+width, fair, label="Per-Flow Fairness", width=width)
    plt.bar(x+width*2, aalo, label="Aalo", width=width)

    plt.plot(plt.xlim(), [1]*2, "black")

    plt.ylim(0, 2.5)
    plt.xticks([0.5, 1.5], ["Avg", "95th"], fontsize=14)
    plt.yticks([0.5, 1, 1.5,2, 2.5], None, fontsize=14)
    plt.legend(loc="upper center", fontsize=12)
    plt.ylabel("Normalized Comp.Time\nw.r.t.M-DRL", fontsize=14)
    plt.xlabel("Coflow Metric", fontsize=14)

    plt.savefig("doc/paper/ComparisionCCT.png")

### TODO：CCT的CDF对比
def compare_CDFofCCT():
    data_sebf = analyse.dark_analyse("doc/sebf.txt", False)/1024 # transform unit to second
    x_sebf, y_sebf = get_x_y(np.log10(data_sebf))

    data_dark = analyse.dark_analyse("doc/dark.txt", False)/1024
    x_dark, y_dark = get_x_y(np.log10(data_dark))

    data_fair = analyse.dark_analyse("doc/fair.txt", False)/1024
    x_fair, y_fair = get_x_y(np.log10(data_fair))

    import seaborn as sns; sns.set_style("whitegrid")
    plt.figure("CDF of CCT")
    plt.rc("font", family="Times New Roman")

    plt.plot(x_sebf, y_sebf)
    plt.plot(x_dark, y_dark)
    plt.plot(x_fair, y_fair)

    plt.ylim([0, 1])
    plt.yticks([0.5, 1], [0.5, 1])
    plt.xlim([-2, 3.5])
    plt.xticks([-2, -1, 0, 1, 2, 3], [0.01, 0.1, 1, 10, 100, 1000])
    plt.grid(linestyle="-.")

    plt.xlabel("Coflow Completion Time(Seconds)", fontsize=14)
    plt.ylabel("Fraction of Coflows", fontsize=14)
    plt.legend(["SEBF", "Aalo", "Per-Flow Fairness"], fontsize=12)

    plt.savefig("doc/paper/CDF_of_CCT.png")

### TODO：训练效果
def get_train_result():
    from analyse import parse_log
    result, _ = parse_log(("doc/log/success-2/log/log_10.txt"))
    start, end = 1, 380
    
    x = range(start, end+1)
    data = np.array(result[:end])/1024/526
    import seaborn as sns; sns.set()
    sns.set_style('whitegrid')
    plt.figure("Training")
    plt.rc("font", family="Times New Roman")
    data_sm = util.smooth_value(data, smoothing=0.9)
    plt.plot(x, data_sm, "-", color="b")
    plt.plot(x, data, "-", alpha=0.35, color="b")

    plt.legend(["DRL"])
    plt.grid(linestyle="-.")
    plt.ylabel("Average CCT(seconds)", fontsize=12)
    plt.xlabel("training episodes", fontsize=12)
    plt.ylim([30, 90])
    plt.xlim([0, end])
    plt.xticks(list(range(50, 400, 50)), list(range(50, 400, 50)))

    plt.savefig("doc/paper/Training.png")


if __name__ == "__main__":
    pass

    plot_CDF()

    # calculate_95th()

    compare_CDFofCCT()

    compare_CCT()

    get_train_result()

    plt.show()