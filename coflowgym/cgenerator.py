import random, math
import matplotlib.pyplot as plt

def getStructure(benchmark_file="scripts/FB2010-1Hr-150-0.txt"):
    """
    生成的Coflow结构为:[
        {
            "time": t, 
            "mappers": [m1,m2,m3], 
            "reducers": [
                [r1_id, r1_size],
                [r2_id, r2_size]
            ],
            "total": t_size
        },
        , ...
    ]
    """
    structures = []
    with open(benchmark_file, "r") as f:

        line = f.readline()
        num_machines, num_jobs = (eval(word) for word in line.split())
        for _ in range(num_jobs):
            line = f.readline()
            words = line.split()
            
            line_dict = {}
            t_c = eval(words[1])
            line_dict["time"] = t_c

            m = eval(words[2])
            total = 0 ## unit is MB
            mappers = []
            for i in range(3, 3+m):
                mappers.append(eval(words[i]))
            line_dict["mappers"] = mappers

            reducers = []
            for reduce in words[4+m:]:
                reducers.append([eval(reduce.split(":")[0]), eval(reduce.split(":")[-1])])
            total = [r[1] for r in reducers]
            line_dict["reducers"] = reducers
            line_dict["total"] = sum(total)
            structures.append(line_dict)
    # print(structures[:4])
    return structures

def makeHeavyTail(N = 100):
    """
    根据二八分布，生成N条重尾分布的Coflow
    """
    structure = getStructure()
    sizes = sorted([line["total"] for line in structure])
    size_limit = int(len(sizes)*0.9)
    long_s = sizes[size_limit:]
    short_s = sizes[:size_limit]
    # print(short_s, long_s)
    rate = 0.88 # short coflows
    short_len = int(N*rate)
    long_len = N-short_len
    all_s = []
    random.seed(47)
    for _ in range(short_len):
        all_s.append(random.choice(short_s))
    for _ in range(long_len):
        all_s.append(random.choice(long_s))
    all_s = sorted(all_s)
    random.shuffle(all_s)
    return all_s

def policy1SizeToCoflow(begin=0, end=100):
    """
    将Coflow大小按比例映射到Coflow结构上
    """
    sizes = makeHeavyTail(end-begin)
    print(sizes)
    assert (end-begin) == len(sizes), "Length of file does not match the given sizes"
    structure = getStructure()
    structure = structure[begin:end]
    begintime = structure[0]["time"]
    for i in range(begin, end):
        coflow = structure[i-begin]
        rate = sizes[i-begin]/coflow["total"]
        for r in range(len(coflow["reducers"])):
            coflow["reducers"][r][1] = float(math.ceil(coflow["reducers"][r][1]*rate))
        coflow["time"] -= begintime
        coflow["total"] = sum([r[1] for r in coflow["reducers"]])
    generateFile(structure)

def policy2SizeToCoflow(begin=0, end=100):
    """
    将Coflow大小按照宽度顺序映射到Coflow结构上
    """
    sizes = makeHeavyTail(end-begin)
    assert (end-begin) == len(sizes), "Length of file does not match the given sizes"
    structure = getStructure()
    structure = structure[begin:end]
    begintime = structure[0]["time"]
    ## 按照宽度排序
    structure = sorted(structure, key=lambda coflow : len(coflow["mappers"])*len(coflow["reducers"]))
    sizes = sorted(sizes)
    print(sizes)
    print([(coflow["time"], coflow["total"], len(coflow["mappers"])*len(coflow["reducers"])) for coflow in structure])

    for i in range(end-begin):
        coflow = structure[i]
        rate = sizes[i]/coflow["total"]
        for r in range(len(coflow["reducers"])):
            coflow["reducers"][r][1] = float(math.ceil(coflow["reducers"][r][1]*rate))
        coflow["time"] -= begintime
        coflow["total"] = sum([r[1] for r in coflow["reducers"]])
    structure = sorted(structure, key=lambda coflow : coflow["time"])
    generateFile(structure)

def generateFile(structure):
    """
    将指定Coflow大小的列表转化为Trace文件
    """
    ## 将修改后的structure写入文件
    with open("scripts/test.txt", "w") as f:
        f.write("%s %s\n"%(150, len(structure)))
        for i, coflow in enumerate(structure):
            # coflow = structure[i-begin]
            line = "%s %s %s "%(i+1, coflow["time"], len(coflow["mappers"]))
            for mapper in coflow["mappers"]:
                line += "%s "%(mapper)
            line += "%s"%(len(coflow["reducers"]))
            for reducer in coflow["reducers"]:
                line += " %s:%s"%(reducer[0], reducer[1])
            line += "\n"
            f.write(line)

def plotWidth():
    structure = getStructure()
    times = [coflow["time"] for coflow in structure]
    plt.subplot(211)
    widths = []
    for coflow in structure:
        widths.append(len(coflow["mappers"])*len(coflow["reducers"]))
    plt.scatter(times, widths, marker='.')

    plt.subplot(212)
    total_size = [math.log10(coflow["total"]) for coflow in structure]
    plt.scatter(times, total_size, marker='.')

    plt.figure()
    upflows = [0]*150
    downflows = [0]*150
    for coflow in structure[:]:
        mappers = coflow["mappers"]
        reducers = coflow["reducers"]
        for r in reducers:
            downflows[r[0]] += r[1]
            for m in mappers:
                upflows[m] += r[1]/len(mappers)
    plt.subplot(211)
    plt.scatter(range(150), upflows, marker='.')
    plt.subplot(212)
    plt.scatter(range(150), downflows, marker='.')

    plt.show()

if __name__ == "__main__":
    pass
    # getStructure()
    # makeHeavyTail()
    policy2SizeToCoflow(begin=150, end=250)
    # plotWidth()