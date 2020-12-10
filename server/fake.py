import random, time, math, wmi, psutil
import numpy as np

qs = COFLOW_DATA = CoflowRequest = GCCT = STEP = None

SYSTEM = None

def reset():
    global qs, STEP, GCCT, COFLOW_DATA, CoflowRequest
    qs = [[],[],[],[],[],[],[],[],[],[]]
    STEP = CoflowRequest = 1
    COFLOW_DATA = []
    GCCT = {
        "table": [
            # {
            #     "sid": 1,
            #     "acct": 100,
            #     "num": 10,
            #     "scheduling": "[1,2,3]",
            # }
        ],
        "chart": {
            "xdata": [],
            "ydata": []
        }
    }
    global SYSTEM
    SYSTEM = {
        "time": [],
        "bw": [],
        "cpu": []
    }

reset()

## MLFQ信息
def genMLFQ():
    global qs
    q = generateRandomMLFQ()
    for i in range(10):
        qs[i].append(q[i])
    return qs

def generateRandomMLFQ():
    q = [0]*10
    for _ in range(100):
        index = random.choice(list(range(10)))
        q[index] += 1
    assert sum(q) == 100, "MLFQ Error: Sum is equal to 100!"
    return q

## Coflow表信息
def genCoflows():
    global CoflowRequest
    global COFLOW_DATA
    COFLOW_DATA.append({
            "cid": CoflowRequest,
            "arrive": 100,
            "flows": '-',
            "mnum": 50,
            "rnum": 100,
            "length": '1TB',
            "sentsize": '1MB',
            "totalsize": '10MB',
            "completed": random.choice([100, 70, 50, 30])
        })
    CoflowRequest += 1
    return COFLOW_DATA

## 实时CCT图和表
def genCCT():
    global GCCT
    oneCCT = generateRandomCCT()
    GCCT["table"].append(oneCCT["table"])
    GCCT["chart"]["xdata"].append(oneCCT["chart"]["xdata"])
    GCCT["chart"]["ydata"].append(oneCCT["chart"]["ydata"])
    return GCCT

def generateRandomCCT():
    global STEP
    acct = random.randint(1, 100)
    one = {
        "table": {
            "sid": STEP,
            "acct": acct,
            "num": random.randint(1, 100),
            "scheduling": "[1,2,3]"
        },
        "chart": {
            "xdata": STEP,
            "ydata": acct
        }
    }
    STEP += 1
    return one

## 系统资源
def generateSystem():
    now = time.strftime("%H:%M:%S")
    cpu = psutil.cpu_percent(interval=1)
    bw = random.randint(80, 90)
    global SYSTEM
    SYSTEM["time"].append(now)
    SYSTEM["cpu"].append(cpu)
    SYSTEM["bw"].append(bw)
    maxLen=10
    data = SYSTEM["time"]
    interval = (1 if len(data) <= maxLen else math.ceil(len(data)/maxLen))
    return {
        "xdata": data,
        "bandwidth": SYSTEM["bw"],
        "cpu": SYSTEM["cpu"],
        "interval": interval
    }

def generateTraces():
    return [{
            "tid": 1,
            "desc": 'Facebook公开数据集',
            "distname": '分布',
            "dist": [{
                "metric": 'Coflow数量',
                "sn": '60%',
                "ln": '16%',
                "sw": '12%',
                "lw": '12%',
            }, {
                "metric": 'Coflow总字节数',
                "sn": '0.01%',
                "ln": '0.11%',
                "sw": '0.88%',
                "lw": '99%',
            }],
            "cdf": {
                "data": [
                    [1,1], [2, 0.7], [3, 0.5], [4, 0.3], [5, 0.1]
                ]
            },
            "cnum": 526,
            "fnum": 1000,
            "length": "8TB",
            "mnum": 150,
        }, {
            "tid": 2,
            "desc": '自定义数据集',
            "distname": '分布',
            "dist": [{
                "metric": 'Coflow数量',
                "sn": '0%',
                "ln": '76%',
                "sw": '5%',
                "lw": '19%',
                },{
                "metric": 'Coflow总字节数',
                "sn": '0%',
                "ln": '44.21%',
                "sw": '0.07%',
                "lw": '55.72%',
                }],
            "cdf": {
                "data": [
                    [1,1], [2, 0.7], [3, 0.5], [4, 0.3], [5, 0.1]
                ]
            },
            "cnum": 100,
            "fnum": 500,
            "length": "1TB",
            "mnum": 150,
        }]


def test():
    pass
    from test import Monitor
    monitor = Monitor(22)
    monitor.start()
    for _ in range(4):
        print(monitor.num)
        time.sleep(2)
    monitor.join()


if __name__ == "__main__":
    # genMLFQ()
    # generateSystem()
    test()