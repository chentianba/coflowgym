import random, time, math, psutil
import numpy as np

qs = COFLOW_DATA = CoflowRequest = GCCT = STEP = None

SYSTEM = None

from server.monitor import Monitor
from server.collector import Collector
monitor = Monitor(23)
monitor.start()
collector = Collector(47)
collector.start()


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
            # "xdata": [],
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
    qs = [[],[],[],[],[],[],[],[],[],[]]
    mlfqs = collector.getMLFQ()
    for e in mlfqs:
        q = np.array(e)/sum(e)
        for i, y in enumerate([round(x*100, 2) for x in q]):
            qs[i].append(y)
    return qs

## Coflow表信息
def genCoflows():
    global COFLOW_DATA

    coflows = collector.getCoflows()
    from server.monitor import Utils
    COFLOW_DATA = [{
        "cid": e["id"],
        "arrive": e["arriveTime"],
        "flows": '-',
        "mnum": 1,
        "rnum": 1,
        "length": Utils.formatFromByte(e["length"]),
        "sentsize": Utils.formatFromByte(e["sentsize"]),
        "totalsize": Utils.formatFromByte(e["totalsize"]),
        "completed": round(e["sentsize"]*100/e["totalsize"])
    } for e in coflows]
    return COFLOW_DATA

## 实时CCT图和表
def genCCT():
    global GCCT

    ccts = collector.getCCT()
    GCCT["table"] = [{
        "sid": i+1,
        "acct": round(e["acct"], 2),
        "num": e["completedNum"],
        "scheduling": e["activeList"],
    } for i, e in enumerate(ccts)]
    GCCT["chart"]["ydata"] = [[i+1, round(e["acct"], 2)] for i, e in enumerate(ccts)]
    return GCCT

## 系统资源
def generateSystem():
    global monitor
    system = monitor.getSystem()
    time = system["time"]
    return {
        "bandwidth": [[t, e] for t, e in zip(time, system["bw"])],
        "cpu": [[t, e] for t, e in zip(time, system["cpu"])],
    }

## Trace展示
def generateTraces():
    return {
        "table": collector.getTraces(),
        "form": {
            "algo": collector.algo,
            "num": collector.traceID,
        }
    }

## 获取后台所有配置
def getConfig():
    return {
        "algo": collector.algo,
        "traceID": collector.traceID,
        "selectedModel": collector.selectedModel,
        "models": collector.models,
        "ending": collector.ending,
    }

############# Setter ###############
def configSource(config):
    print("collector status:", collector.ending)
    if not collector.ending:
        return False
    collector.setTraceID(config["traceID"])
    collector.setAlgo(config["algo"])
    collector.reset()
    return True

def transfer(info):
    collector.setModel(info["model"])
    collector.transfering = True

def cancel():
    collector.ending = True

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