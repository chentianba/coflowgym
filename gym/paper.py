import matplotlib.pyplot as plt 
import numpy as np 
import benchdata
import util

### TODO: Benchmark CDF
def plot_CDF():
    pass


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
    sns.set_style("darkgrid", {'axes.grid' : False})
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
    plt.ylabel("Normalized Comp.Time\nw.r.t.DRL", fontsize=14)
    plt.xlabel("Coflow Metric", fontsize=14)


### TODO：CCT的CDF对比


### TODO：训练效果
def get_train_result():
    from analyse import parse_log
    result, _ = parse_log(("log/log_10.txt"))
    start, end = 1, 450
    
    x = range(start, end+1)
    data = np.array(result[:end])/1024/100
    import seaborn as sns; sns.set()
    sns.set_style('whitegrid')
    plt.figure("Training")
    plt.rc("font", family="Times New Roman")
    data_sm = util.smooth_value(data, smoothing=0.9)
    plt.plot(x, data_sm, "-")
    plt.plot(x, data, "-", alpha=0.35, color="g")

    plt.legend(["DRL"])
    plt.ylabel("Average CCT(seconds)", fontsize=12)
    plt.xlabel("training episodes", fontsize=12)


if __name__ == "__main__":
    pass

    compare_CCT()

    get_train_result()

    plt.show()