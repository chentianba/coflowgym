import numpy as np
from datetime import datetime
import os, sys, math
from scipy.stats import gaussian_kde

def chengji(x):
    """
    calculate the product of x, e.g. [2,3,5]->30
    """
    res = 1
    for e in x:
        res = res*e
    return res

def toFactor(x, init_limit=1024):
    res = [x[0]/init_limit]
    for i in range(1, len(x)):
        res.append(x[i]/x[i-1])
    return res

def get_h_m_s(second):
    """
    transform from second to hour-minite-second
    return a string
    """
    if second <= 0:
        return 0, 0, 0
    m, s = divmod(round(second), 60)
    h, m = divmod(m, 60)
    return "%sH %sM %sS"%(h, m, s)

def get_now_time():
    """
    return a string of Y-M-D-H-M-S
    """
    now = datetime.now()
    return "%s-%s-%s-%s-%s-%s"%(now.year, now.month, now.day, now.hour, now.minute, now.second)

def cal_limit(file):
    """
    @return
      width: unit is megabyte
      size:
    """
    with open(file) as f:
        width = [1e100, 0]
        size = [1e100, 0]

        line = f.readline()
        _, coflows = (eval(word) for word in line.split())
        for _ in range(coflows):
            line = f.readline()
            words = line.split()
            num_mapper = eval(words[2])
            num_reduer = eval(words[3+num_mapper])
            t_size = 0
            c_width = num_mapper*num_reduer
            for reducer in words[4+num_mapper:]:
                _, rs = reducer.split(":")
                t_size += eval(rs)
            width[0] = min(width[0], c_width)
            width[1] = max(width[1], c_width)
            size[0] = min(size[0], t_size)
            size[1] = max(size[1], t_size)
            # print(c_width, t_size)
        return width, size

def parse_benchmark(benchmark_file="scripts/FB2010-1Hr-150-0.txt"):
    """
    Parse coflows in Benchmark `scripts/FB2010-1Hr-150-0.txt`
    return
    -------
        time:
        mappers:
        reducers:
        shuffle_t:
    """
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
        return time, mappers, reducers, shuffle_t

def make100coflows(benchmark_file="scripts/FB2010-1Hr-150-0.txt"):
    time = []
    shuffle_t = []
    lines = []
    target = []
    with open(benchmark_file, "r") as f:

        line = f.readline()
        num_machines, num_jobs = (eval(word) for word in line.split())
        print("Number of machines:", num_machines)
        print("Number of jobs:", num_jobs)
        for _ in range(num_jobs):
            line = f.readline()
            lines.append(line)

            words = line.split()
            time.append(eval(words[1]))

            m = eval(words[2])
            total = 0 ## unit is MB
            for reduce in words[4+m:]:
                total += eval(reduce.split(":")[-1])
            shuffle_t.append(total)
        # import matplotlib.pyplot as plt
        # plt.figure("Coflow Size")
        # plt.plot(shuffle_t)
        # plt.show()
        start, end = 0, 195
        start_time = time[start]
        for index in range(start, end):
            line = lines[index]
            words = line.split()
            t_line = "%s %s %s"%(index-start+1, time[index]-start_time, " ".join(words[2:]))
            target.append(t_line)
    with open("scripts/test_%s_%s.txt"%(start, end), "w") as f:
        f.write("%s %s\n"%(num_machines, end-start))
        for line in target:
            f.write(line)
            f.write("\n")

def prepare_pm():
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
    sent_s = [e for e in sent_s if e != 0]
    sent_s = np.log10(sent_s)
    return sent_s

class Logger:
    def __init__(self, file):
        assert type(file) == str, "please given a right file of 'str'."
        if os.path.exists(file):
            os.remove(file)
        self.filename = file
    
    def print(self, info):
        assert type(info) == str, "Info into logger should be a string."
        log = open(self.filename, 'a')
        log.write(info+"\n")
        log.close()

class KDE():
    def __init__(self, init, size=10000):
        self.pool = list(init)
        self.size = size
        self.pointer = len(self.pool)

        # self.max_val = 0
        self.GAP = 20
        self.kde = gaussian_kde(self.pool)
    
    def push(self, data):
        self.pool.extend(data)
        self.pointer += len(data)
        if self.pointer > self.size:
            self.pool = self.pool[-self.size:]
            self.pointer = self.size
    
    def update(self):
        if self.pool == []:
            self.kde = gaussian_kde([0, 1])
        else:
            self.kde = gaussian_kde(self.pool)
    
    def get_val(self, prob):
        epsilon = 0.001
        v_min = 0
        v_max = self.GAP
        if prob < self.kde.integrate_box(-math.inf, v_min):
            return v_min
        if prob > self.kde.integrate_box(-math.inf, v_max):
            return v_max
        v = (v_min+v_max)/2
        p = self.kde.integrate_box(-math.inf, v)
        while abs(p - prob) > epsilon:
            if p > prob:
                v_max = v
            else:
                v_min = v
            v = (v_max+v_min)/2
            p = self.kde.integrate_box(-math.inf, v)
        return v
    
    def print(self):
        x = np.arange(0, 15)
        print("In KDE:", self.kde.pdf(x))

def test():
    action = [0.1, 0.2, 0.3, 0.4]
    a = [0]
    for e in action:
        a.append(a[-1]+e)
    del a[0]
    print(a)

if __name__ == "__main__":
    # print(cal_limit("scripts/FB2010-1Hr-150-0.txt")) # result is ([1, 21170], [1.0, 8501205.0]) MB
    pass
    # print(toFactor([2,4,12,36], 2))
    # make100coflows()
    # makePM()
    test()
