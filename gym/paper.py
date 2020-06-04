import matplotlib.pyplot as plt 
import numpy as np 
import benchdata

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

    plt.figure("Comparision CCT")
    plt.rc("font", family="Times New Roman")
    plt.bar(x, varys, label="Varys", width=width, color="red")
    plt.bar(x+width, fair, label="Per-Flow Fairness", width=width, color="blue")
    plt.bar(x+width*2, aalo, label="Aalo", width=width, color="green")

    plt.plot(plt.xlim(), [1]*2, "black")

    plt.ylim(0, 2.5)
    plt.xticks([0.5, 1.5], ["Avg", "95th"], fontsize=14)
    plt.yticks([0.5, 1, 1.5,2, 2.5], None, fontsize=14)
    plt.legend(loc="upper center", fontsize=12)
    plt.ylabel("Normalized Comp.Time\nw.r.t.DRL", fontsize=14)
    plt.xlabel("Coflow Metric", fontsize=14)


### TODO：CCT的CDF对比


### TODO：训练效果


if __name__ == "__main__":
    pass

    compare_CCT()

    plt.show()