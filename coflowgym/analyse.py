import numpy as np 
import codecs
import sys, math
import pprint
import matplotlib.pyplot as plt
from coflowgym.benchdata import exp4_benchmark_data
from coflowgym.util import toFactor, prepare_pm, parse_benchmark
import coflowgym.util

## DARK / FIFO / SEBF
configuration = [
    [2.4247392E7, 4.3473352E7, 1.5005968E7, 1.9205752E7], # benchmark
    [326688.0, 612776.0, 281880.0, 0],   # 100coflows
    [6915640.0, 4483152.0, 3461920.0, 0], # test_200_250 ## 2
    [2214624.0, 4509552.0, 1952872.0, 0], # test_150_200  ## 3
    [1.5923608E7, 1.3596456E7, 7.337872E6, 0], # test_150_250  ## 4
    [3615440.0, 2906384.0, 2474200.0, 0], # test_200_225 ## 5
    [3.73975776E8, 5.20843856E8, 2.44592544E8, 343305064.0], # light tail
    [1379944.0, 1785416.0, 1222736.0,1233768.0], # custom
]
choice = -2

def stats_action(file):
    with open(file, 'r') as f:
        actions = []

        line = f.readline()
        while line:
            pos = line.find("env_action")
            if pos != -1:
                action = eval(line[:pos].split(":")[-1])
                actions.append(action)

            line = f.readline()
        # print(actions[:10])
    actions = np.array(actions)
    N = len(actions[0])
    plt.figure()
    n = math.ceil(math.sqrt(N))
    for i in range(N):
        plt.subplot("%s%s%s"%(n, n, i+1))
        cdf = toCDF(actions[:, i])
        plt.plot(cdf[:, 0], cdf[:, 1])
        plt.xlabel("action[%s]"%i)

def toCDF(data_l):
    """ 
    data is discrete.
    example:
    [1,1,2,3] -> [[1, 0.5], [2, 0.5], [2, 0.75], [3, 0.75], [3, 1.0]]
    """
    l_dict = {}
    for e in sorted(data_l):
        if e in l_dict:
            l_dict[e] += 1
        else:
            l_dict[e] = 1 
    l_sum = 0 
    res = [[0, 0]] 
    for e in sorted(l_dict):
        l_sum += l_dict[e]
        res.append([e, res[-1][1]])
        res.append([e, l_sum/len(data_l)])
    del res[:2]
    return np.array(res)

def toPercentile(cdf, num=10):
    count = 1
    percentile = []
    for e in cdf:
        while e[1] > count/num:
            percentile.append(e[0])
            count += 1
    return percentile

def benchmark_analyse(benchmark_file):
    with open(benchmark_file, "r") as f:
        time = []
        mappers = []
        reducers = []
        shuffle_t = []

        line = f.readline()
        num_machines, num_jobs = (eval(word) for word in line.split())
        print("Number of machines:", num_machines)
        print("Number of jobs:", num_jobs)
        for i in range(num_jobs):
            line = f.readline()
            words = line.split()
            time.append(eval(words[1]))

            m = eval(words[2])
            r = eval(words[3+m])
            total = 0
            for reduce in words[4+m:]:
                total += eval(reduce.split(":")[-1])
            mappers.append(m)
            reducers.append(r)
            shuffle_t.append(total)
        
        ##
        # print(time)
        # print(mappers)
        # print(reducers)
        shuffle_t = np.array(shuffle_t)
        shuffle_log = np.log10(shuffle_t)
        plt.figure("benchmark_analyse")

        plt.subplot(221)
        plt.scatter(range(1, 1+len(shuffle_t)), shuffle_log, marker=".")
        plt.xlabel("No")
        plt.ylabel("log of shuffle size/MB")
        plt.grid()
        
        plt.subplot(222)
        cdf = toCDF(shuffle_log)
        plt.plot(cdf[:, 0], cdf[:, 1])
        plt.xlabel("log of shuffle size/MB")
        plt.ylabel("CDF")
        plt.grid()

        plt.subplot(223)
        plt.plot(time, shuffle_t)
        plt.xlabel("arrive time/ms")
        plt.ylabel("shuffle size/MB")

        plt.subplot(224)
        plt.plot([m*r for m,r in zip(mappers, reducers)])
        plt.xlabel("No")
        plt.ylabel("width")

        plt.figure("sentsize analyse")
        import scipy.stats as st
        import seaborn as sns 
        sns.set()
        sent_s = prepare_pm()
        sns.distplot(sent_s, fit=st.norm, fit_kws={"color":"r", "label":"norm"}, kde_kws={"label":"KDE"})
        sns.utils.axlabel("sentsize(log10)", "")
        sns.utils.plt.legend()

        # sns.set_style("whitegrid")

def dark_analyse(file="doc/dark.txt", isplot=True):
    with open(file, "r") as f:
        durations = []
        shuffle_t = []

        line = f.readline()
        while line:
            if line.startswith("JOB"):
                words = line.split()
                start_time = eval(words[1])
                finish_time = eval(words[2])
                total_shuffle = eval(words[5])
                duration = eval(words[-3])
                
                durations.append(duration)
                shuffle_t.append(total_shuffle)
                if finish_time - start_time != duration:
                    print(words[0], "time doesn't match!")

            line = f.readline()
        ##
        if isplot:
            plt.figure()
            plt.plot(shuffle_t)
            plt.xlabel("JOB")
            plt.ylabel("Shuffle/B")
        return np.array(durations)

def analyse_shuffle():
    _, _, _, shuffle = parse_benchmark()
    _, _, _, shuffle = parse_benchmark("scripts/custom.txt")
    import seaborn as sns 
    sns.set()
    plt.figure("Analyse of Coflow Size")
    sns.scatterplot(range(len(shuffle)), np.log10(shuffle))
    # for i, size in enumerate(np.log10(shuffle)):
    #     print(i, "--", size)
    fig = plt.figure()
    fig.add_subplot(121)
    sns.distplot(np.log10(shuffle))
    fig.add_subplot(122)
    sns.kdeplot(np.log10(shuffle), cumulative=True)

def analyse_sentsize():
    with open("log/log.txt") as f:
        ss_l = []
        sent = []
        
        line = f.readline()
        while line:
            if line.startswith("episodic"):
                sentsize = eval(line.split(":")[-1])
                ss_l.append([e for e in sentsize if e != 0])
                sent.extend(ss_l[-1])
            line = f.readline()
        plt.figure()
        import seaborn as sns 
        sns.distplot(np.log10(sent[-1000:]))

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

def parse_complete_log(file="log/log.txt"):
    """
    return a dict, including {"result":[], "ep_reward":[], "coflows":[[],[],...]}
    """
    with open(file, 'r') as f:
        result = []
        ep_reward = []
        coflows = []

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
            if line.startswith("cf_info:"):
                jobs = eval(line.split(":")[-1])
                coflows_t = []
                for job in jobs:
                    words = job.split()
                    duration = eval(words[-3])
                    coflows_t.append(duration)
                coflows.append(coflows_t)
        return {"result":result, "ep_reward":ep_reward, "coflows":coflows}

def plot_compare(result, ep_reward, newfigure=True, is_benchmark=True):
    if newfigure:
        plt.figure()
    x = list(range(len(result)))

    if is_benchmark:
        # comp = [2.4247392E7, 4.3473352E7, 1.5005968E7]
        comp = configuration[0]
    else:
        # comp = [326688.0, 612776.0, 281880.0]
        comp = configuration[choice]
    plt.plot(x, result, 'b.-')
    plt.plot(x, [comp[0]]*len(x), "red") # DARK
    plt.plot(x, [comp[1]]*len(x), "cyan") # FIFO
    plt.plot(x, [comp[2]]*len(x), "lawngreen") # SEBF
    plt.plot(x, [comp[3]]*len(x), "sandybrown") # Target
    plt.legend(["DRL", "DARK", "FIFO", "SEBF", "Target"])
    plt.xlabel("episode")
    plt.ylabel("ep_runtime")

    # plt.figure()
    # plt.plot(x, [-r for r in ep_reward])
    # plt.scatter(x, [-r for r in ep_reward])

def validate_reward(result, ep_reward, newfigure=True):
    if newfigure:
        plt.figure()
    plt.scatter(result, ep_reward)
    f = np.poly1d(np.polyfit(result, ep_reward, 1))
    print(f)
    plt.plot(result, f(result), 'y')
    plt.xlabel("ep_runtime")
    plt.ylabel("ep_total_reward")

def analyse_reward(file):
    with open(file, "r") as f:
        rs = []

        line = f.readline()
        while line:
            start = line.find("reward:")
            if start != -1:
                words = line[start:].split()
                r = eval(words[1])
                rs.append(r)
            line = f.readline()
        plt.figure()
        p = rs[:500]
        plt.scatter(range(len(p)), p)

def analyse_mlfq():
    with open("log/mlfq.txt", "r") as f:
        mlfqs = []
        line = f.readline()
        while line:
            if line.startswith("MLFQ"):
                mq = eval(line.split(":")[-1])
                mlfqs.append(mq)
            line = f.readline()
        coflow_mlfq = [sum(mlfq) for mlfq in mlfqs]
        plt.figure("Number in MLFQ")
        plt.subplot(211)
        plt.scatter(range(len(coflow_mlfq)), coflow_mlfq, marker='.')
        plt.ylabel("number of coflow")
        plt.xlabel("step")
        cdf = toCDF(coflow_mlfq)
        plt.subplot(212)
        plt.plot(cdf[:, 0], cdf[:, 1])
        plt.xlabel("number of coflow")
        plt.ylabel("cdf")
    if True:
        # analyse the rule of sent size in benchmark
        # config: 7 queues, [10M, 100M, 1G, 10G, 100G, 1T]
        with open("doc/benchmark_sentsize.txt") as f:
            line = f.readline()
            mlfqs = []
            while line:
                if line.startswith("coflow"):
                    mlfqs = eval(line.split(":")[-1])
                line = f.readline()
            print("steps of benchmark: ", len(mlfqs))
            sent_s = []
            for coflow in mlfqs:
                sent_s.extend(coflow)
            N = 13
            count = [0]*N
            powers = []
            for size in sent_s:
                index = int(math.log10(size)) if size != 0 else 0
                powers.append(index)
                count[index] += 1
            print("count:", count)
            data = np.array(count)/sum(count)
            print("data", data)
            print("powers:", powers)
            plt.figure("Benchmark Sent Size")
            plt.title("Distribution of sent size in Benchmark")
            # plt.plot(range(N), data)
            res = plt.hist(powers,bins=N*2,density=True)
            vals, pos = res[0], res[1][:-1]
            # plt.plot(pos, vals, 'r')
            # for v, p in zip(vals, pos):
            #     plt.text(p+0.5, v, "{:2f}".format(v))
    if False:
        with open("log/result.txt") as f:
            mlfqs = []
            line = f.readline()
            while line:
                if line.startswith("MLFQ"):
                    mq = eval(line.split(":")[-1])
                    mlfqs.append(mq)
                line = f.readline()
            count = 0
            for mq in mlfqs:
                if np.sum(np.array(mq) > 1) > 0:
                    count += 1
            print("Length of samples:", len(mlfqs))
            print("count of Coflow in MLFQ is more than 1:", count/len(mlfqs))

def analyse_samples(file="log/sample_1.txt"):
    with open(file) as f:
        actions = []
        results = []
        
        line = f.readline()
        while line:
            if line.startswith("Action"):
                pos = line.find("result")
                res = eval(line[pos:].split(":")[-1])
                if res < 326688:
                    act = eval(line[:pos].split(":")[-1])
                    actions.append(act)
                    results.append(res)
            line = f.readline()
        # for act, res in zip(actions, results):
        #     print(toFactor(act, 1024), res)
        print("Minimum Result:", min(results))

def plot_smoothing(x, y, fig):
    from util import smooth_value
    import pandas as pd
    y_s = smooth_value(y, 0.9)
    data = pd.DataFrame({
        "x": list(range(len(y))),
        "y": y,
        "y_s": y_s
    })
    # print(data)

    # import seaborn as sns; sns.set()
    # sns.lineplot(x="x", y="y", data=data, ci=0.5)
    # sns.lineplot(x="x", y="y_s", data=data)

    plt.figure(fig)
    color = "royalblue"
    plt.plot(x, y, linestyle="-", color=color, alpha=0.35)
    plt.plot(x, y_s, linestyle="-", color=color)
    # plt.show()

def add_compare(is_benchmark=True):
    x = plt.xlim()
    comp = configuration[0] if is_benchmark else configuration[choice]
    plt.plot(x, [comp[0]]*len(x), "red") # DARK
    plt.plot(x, [comp[1]]*len(x), "cyan") # FIFO
    plt.plot(x, [comp[2]]*len(x), "lawngreen") # SEBF
    plt.plot(x, [comp[3]]*len(x), "sandybrown") # Target

def parse_run_log(file="log/run.txt"):
    with open(file, "r") as f:
        runtime = {}
        model = -1
        result = []

        line = f.readline()
        while line:
            if line.startswith("Model:"):
                runtime[model] = result
                model = eval(line.split(":")[-1])
                result = []
            if line.startswith("result"):
                result.append(eval(line.split(":")[-1].split()[0]))

            line = f.readline()
        runtime[model] = result
    del runtime[-1]
    # print(runtime)
    pprint.pprint(runtime)

    x, y = [], []
    for key in runtime:
        x.extend([key]*len(runtime[key]))
        y.extend(runtime[key])
    px = sorted(runtime.keys())
    py = [[sum(runtime[key])/len(runtime[key]), (np.argmin(runtime[key]), min(runtime[key]))] for key in px]
    py = []
    for key in px:
        if len(runtime[key]) < 10:
            py.append([sum(runtime[key])/len(runtime[key]), (np.argmin(runtime[key]), min(runtime[key]))])
        else:
            t = sorted(runtime[key])
            l = len(t)
            t = t[l//5: l*4//5]

            py.append([sum(t)/len(t), (np.argmin(runtime[key]), min(runtime[key]))])

    plt.figure("run:%s"%(file))
    plt.scatter(x, y, marker=".")
    print("debug:", px, py)
    plt.plot(px, [e[0] for e in py], "r")

def analyse_log(exp_no):

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
        validate_reward(result, ep_reward, newfigure=False)
        plt.subplot(212)
        plot_compare(result, ep_reward, newfigure=False)
    if exp_no is 3:
        result, ep_reward = parse_log("doc/log/4_log.txt")
        result_2, ep_reward_2 = parse_log("doc/log/5_log.txt")
        plt.subplot(221)
        validate_reward(result, ep_reward, newfigure=False)
        plt.title("alpha=0.6")
        plt.subplot(222)
        validate_reward(result_2, ep_reward_2, newfigure=False)
        plt.title("alpha=1")
        plt.subplot(223)
        plot_compare(result, ep_reward, newfigure=False)
        plt.subplot(224)
        plot_compare(result_2, ep_reward_2, newfigure=False)

        plt.figure()
        res100, ep_r_100 = parse_log("doc/log/1_log_100.txt")
        plt.subplot(221)
        validate_reward(res100, ep_r_100, newfigure=False)
        plt.title("100coflows: alpha=0")
        plt.subplot(223)
        plot_compare(res100, ep_r_100, newfigure=False, is_benchmark=False)

        result_6, ep_reward_6 = parse_log("doc/log/6_log.txt")
        plt.subplot(222)
        validate_reward(result_6, ep_reward_6, newfigure=False)
        plt.title("alpha=0")
        plt.subplot(224)
        plot_compare(result_6, ep_reward_6, newfigure=False)
    if exp_no is 4:
        result100, ep_reward_100 = parse_log("doc/log/2_log_100.txt")
        plt.subplot(221)
        validate_reward(result100, ep_reward_100, newfigure=False)
        plt.title("100coflows")
        plt.subplot(223)
        plot_compare(result100, ep_reward_100, is_benchmark=False, newfigure=False)
        # result, ep_reward = parse_log("doc/log/7_log.txt")
        result, ep_reward = exp4_benchmark_data()
        plt.subplot(222)
        validate_reward(result, ep_reward, newfigure=False)
        plt.subplot(224)
        plot_compare(result, ep_reward, newfigure=False)
    if exp_no is 5:
        res100, ep_r_100 = parse_log("doc/log/3_log_100.txt")
        plt.subplot(221)
        validate_reward(res100, ep_r_100, newfigure=False)
        plt.title("100coflows")
        plt.subplot(223)
        plot_compare(res100, ep_r_100, is_benchmark=False, newfigure=False)
        res, ep_r = parse_log("doc/log/8_log.txt")
        plt.subplot(222)
        validate_reward(res, ep_r, newfigure=False)
        plt.subplot(224)
        plot_compare(res, ep_r, newfigure=False)
    if exp_no is 6:
        res100, ep_r_100 = parse_log("doc/log/4_log_100.txt")
        plt.subplot(211)
        plot_compare(res100, ep_r_100, is_benchmark=False, newfigure=False)
        plt.title("100coflows")
        plt.subplot(212)
        plt.plot(ep_r_100)
        plt.xlabel("episode")
        plt.ylabel("ep_reward")

    ## 
    if exp_no is 102:
        ## success-2 log / Facebook
        result, ep_reward = parse_log(("doc/log/success-2/log/log_10.txt"))
        print("Number of samples:", len(result))
        print(len(result))
        # print(result, ",", ep_reward)
        result, ep_reward = np.array(result), np.array(ep_reward)
        is_benchmark = True

        # validate_reward(result[result < 350000], ep_reward[result < 350000])
        plot_compare(result, ep_reward, is_benchmark=is_benchmark, newfigure=False)

        plot_smoothing(range(len(result)), result, "result")
        add_compare(is_benchmark=is_benchmark)

        plot_smoothing(range(len(ep_reward)), ep_reward, "ep_reward")

        # plt.figure("Exp")
        # plt.subplot(221)
        # plt.subplot(222)
        validate_reward(result, ep_reward, newfigure=True)
        # plt.subplot(223)
        # plt.plot(ep_reward)
        # plt.ylabel("ep_reward")
        # plt.xlabel("episode")
        # plt.subplot(224)
        # cdf = toCDF(result)
        # plt.plot(cdf[:, 0], cdf[:, 1])
        # plt.xlabel("runtime")
        # plt.ylabel("CDF")

    if exp_no < 0:
        # result, ep_reward = parse_log(("doc/log/success-2/log/log_10.txt"))
        # result, ep_reward = parse_log("E:/@Data/programming/coflowgym/doc/log/lighttail/best_run_log.txt")
        result, ep_reward = parse_log("E:/@Data/programming/coflowgym/doc/log/lighttail/log/log_10.txt")
        print("Number of samples:", len(result))
        print(len(result))
        # print(result, ",", ep_reward)
        result, ep_reward = np.array(result), np.array(ep_reward)
        is_benchmark = False

        # validate_reward(result[result < 350000], ep_reward[result < 350000])
        plot_compare(result, ep_reward, is_benchmark=is_benchmark, newfigure=False)

        plot_smoothing(range(len(result)), result, "result")
        add_compare(is_benchmark=is_benchmark)

        plot_smoothing(range(len(ep_reward)), ep_reward, "ep_reward")

        # plt.figure("Exp")
        # plt.subplot(221)
        # plt.subplot(222)
        validate_reward(result, ep_reward, newfigure=True)

def queue_validate():
    bmk_sentsize = prepare_pm()
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
    plt.figure("Queue Validation")
    plt.plot(range(10), bmk_count, "o-")
    # plt.bar([i for i in range(6, 16)], bmk_count)
    plt.xticks(list(range(10)), ["Q%s"%(i) for i in range(1, 11)])

    actions, sentsize, _, _ = util.best_model_log_parse("doc/log/success-2/best_run_log.txt")
    print(len(actions), len(sentsize))
    # for a, b in zip(actions[:10], sentsize[:10]):
    #     print(a, b)
    count = [0 for _ in range(10)]
    for action, sent in zip(actions, sentsize):
        for size in sent:
            i = 0
            while i < 9 and size >= action[i]:
                i += 1
            count[i] += 1
    print(count)
    total = sum([len(e) for e in sentsize])
    plt.plot(range(10), np.array(count)/total, "-")
    plt.legend(["Aalo", "DRL"])
    

if __name__ == "__main__":
    
    analyse_log(-102)

    # stats_action("log/log.txt")
    
    # analyse_mlfq()

    # benchmark_analyse("scripts/FB2010-1Hr-150-0.txt")

    # analyse_shuffle()

    # analyse_sentsize()

    # dark_analyse()

    # analyse_samples("log/run_8000.txt")

    # parse_run_log()
    # parse_run_log("log/6_run.txt")

    # queue_validate()

    plt.show()