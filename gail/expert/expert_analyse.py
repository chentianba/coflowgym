import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.gridspec as gridspec
import math

sscf_dark_data = [
    [1,    1.892824E7],
    [2,    1.4671248E7],
    [3,    1.432372E7],
    [4,    1.3928736E7],
    [4.25, 1.392728E7],
    [4.5,  1.3926672E7],
    [4.75, 1.3926672E7],
    [4.8,  1.3926784E7],
    [4.9,  1.3926784E7],
    [5,    1.9272504E7]
]

def get_sebf_log(filename="gail/sebf-expert.txt"):
    with open(filename, "r") as f:
        line = f.readline()
        count = 0
        jobs_ep = [] # jobs in an episode
        while line:
            if line.startswith("sortedJobs"):
                jobs_str = line.split(":")[-1].strip()[1:-1].split(",")
                # print(jobs_str)
                count += 1
                jobs = []
                for job_s in jobs_str:
                    job_name, job_sent = job_s.strip().split("(")
                    name = eval(job_name.split("-")[-1])
                    sent = eval(job_sent.split("|")[-1].split("/")[0])
                    jobs.append([name, sent])
                jobs_ep.append(jobs)
                # print(jobs)
            line = f.readline()
        return jobs_ep

def analyse_sebf_log():
    jobs_ep = get_sebf_log()
    ## proceed jobs
    # print(jobs_ep)
    prior_jobs = {}
    sent_list = []
    for jobs in jobs_ep:
        for pr, job in enumerate(jobs):
            size = math.log10(job[1]) if job[1] != 0 else 0
            if size == 0:
                continue
            sent_list.append([pr, size])
            if pr in prior_jobs:
                prior_jobs[pr].append(size)
            else:
                prior_jobs[pr]= [size]
    print(prior_jobs.keys())
    sent_list = np.array(sent_list)

    sns.set_style('white')
    data_sent = pd.DataFrame({'prior':sent_list[:,0], 'sent':sent_list[:,1]})
    sns.scatterplot(data=data_sent, x='prior', y='sent')
    plt.xlabel("Priority")
    plt.ylabel("Sent Size(B)")

    # plt.subplot(2, 2, 2)
    # num_jobs = np.array([[key, len(prior_jobs[key])] for key in prior_jobs])
    # plt.scatter(num_jobs[:, 0], num_jobs[:, 1])
    # plt.xlabel("Priority")
    # plt.ylabel("Number of Coflows")

    avg_jobs = np.array([[key, sum(prior_jobs[key])/len(prior_jobs[key])] for key in prior_jobs])
    data = pd.DataFrame({'x':avg_jobs[:,0], 'y':avg_jobs[:,1]})
    sns.lmplot(data=data, x='x', y='y')
    plt.xlabel('Priority')
    plt.ylabel('Average Sent Size(B)')

    plt.show()

def rebuild_mlfq_based_on_sebf():
    jobs_ep = get_sebf_log()
    jobs_log = []
    for jobs in jobs_ep:
        jobs_log.append([(math.log10(job[1]) if job[1] != 0 else 0) for job in jobs])
    MLFQ = [[],[],[],[],[],[],[],[],[],[]]
    N = 10
    ## 尾链表重叠
    count = 0 # 统计优先级错位的数量
    # for jobs in jobs_log:
    #     if len(jobs) <= N:
    #         for i,job in enumerate(jobs):
    #             MLFQ[i].append(job)
    #     else:
    #         rem = len(jobs) - N
    #         count += (rem*2)
    #         for i in range(N-rem):
    #             MLFQ[i].append(jobs[i])
    #         j = N-rem
    #         for i in range(N-rem, N):
    #             MLFQ[i].append(jobs[j])
    #             MLFQ[i].append(jobs[j+1])
    #             j += 2

    ## 末队列添加
    for jobs in jobs_log:
        if len(jobs) <= N:
            for i,job in enumerate(jobs):
                MLFQ[i].append(job)
        else:
            rem = len(jobs) - N
            count += rem
            for i in range(N):
                MLFQ[i].append(jobs[i])
            for i in range(N, N+rem):
                MLFQ[N-1].append(jobs[i])
    sum_active_c = sum([len(jobs) for jobs in jobs_ep])
    print("total: ", sum_active_c, "mismatch:", count)
    print("Rate of mismatch:", round(count/sum_active_c, 4)*100, "%")
    
    ## 画图
    nozero_MLFQ = []
    for queue in MLFQ:
        q = []
        for size in queue:
            if size > 0:
                q.append(size)
        nozero_MLFQ.append(q)
    queue_size = np.array([[min(queue), max(queue), sum(queue)/len(queue)] for queue in MLFQ])
    print(queue_size)
    queue_nozero = np.array([[min(queue), max(queue), sum(queue)/len(queue)] for queue in nozero_MLFQ])
    print(queue_nozero)
    # plt.plot(range(queue_size.shape[0]), queue_size[:, 2])
    plt.plot(range(queue_nozero.shape[0]), queue_nozero[:, 0], 'ro-')
    plt.plot(range(queue_nozero.shape[0]), queue_nozero[:, 1], 'b+-')
    plt.plot(range(queue_nozero.shape[0]), queue_nozero[:, 2])
    plt.legend(['min', 'max', 'average'])
    plt.show()

def plot_sscf_dark():
    sns.set_style('white')

    data = np.array(sorted(sscf_dark_data, key=lambda x : x[0]))
    print(data)
    plt.plot(data[:, 0], data[:, 1], 'o-')
    plt.show()

if __name__ == "__main__":
    pass
    # analyse_sebf_log()

    # rebuild_mlfq_based_on_sebf()

    plot_sscf_dark()