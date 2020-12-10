import numpy as np
import threading, os, pdb
from jpype import *

from constant import INNER_TRACES, ALGOS, MDRL_MODELS, GAIL_MODELS
from coflowgym.coflow import CoflowSimEnv, CoflowKDEEnv
from coflowgym import util
from server.monitor import Utils
from server.model import DDPGModel

class Collector(threading.Thread):

    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.setDaemon(True)
        self.threadID = threadID

        self._configJava()
        
        ## setting
        self.traceID = 1
        self.algo = "Aalo"
        self.models = []
        self.selectedModel = None

        ## global
        self.env = None
        self.ending = True
        self.policy = None 
        self.transfering = False
        self.cacheTrace = None

        ## data
        self.steps = [
            # {
            #     "completedNum": 0,        # 完成Coflow数量
            #     "acct": 0,                # 平均CCT
            #     "activeList", "[1,2,3]"   # 活跃Coflow列表
            # }
        ]
        self.coflows = [
            # {
            #     "id": 1,                    # ID
            #     "arriveTime": 100,          # 到达时间，单位为毫秒
            #     "length": 1,                # 长度，单位为Byte
            #     "sentsize": 1,              # 已发送字节数，单位为Byte
            #     "totalsize": 1,             # 总字节数，单位为Byte
            #     "duration": 0,              # 持续时间，单位为毫秒
            # }
        ]
        self.curIndex = []
        self.mlfqs = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.mlfq_step = [
            # []
        ]

        ## 初始化
        self.startEnv()
    
    def run(self):
        print("In run...")
        while True:
            if self.transfering:
                self.oneTransfer()
                self.transfering = False
    
    def oneTransfer(self):
        print("Transfer...")
        self._initializeRecord()
        self.ending = False
        self._configModel()

        # pdb.set_trace()
        obs = self.env.reset()
        for step in range(int(1e8)):
            print("step:", step)
            action = self._genAction(obs)
            obs_n, reward, done, info = self.env.step(action)
            ## stats: id, width, already bytes(b), duration time(ms)
            # print(info["obs"], info["completed"])
            self._recordCoflowStatus(step+1, info["obs"], info["completed"])
            if self.algo in ["Aalo", "M-DRL", "CS-GAIL"]:
                self._recordMLFQ(step+1, action, info["obs"])
            # self.env.coflowsim.getCoflowInfo()
            # break

            obs = obs_n
            if self.ending:
                return
            if done:
                # result, _ = self.env.getResult()
                # print("result: ", result)
                # durations = [e["duration"] for e in self.coflows]
                # print(durations, sum(durations))
                break
        self.ending = True
    
    def getCCT(self):
        return self.steps
    
    def getCoflows(self):
        # print("Coflows:", self.coflows)
        data = []
        for cid in sorted(self.curIndex):
            data.append(self.coflows[cid - 1])
        return data
    
    def getMLFQ(self):
        # print("MLFQ:", self.mlfq_step)
        return self.mlfq_step
    
    def _recordMLFQ(self, step, action, obs):
        obs = eval(obs.split(":")[-1])
        for e in obs:
            size = e[2]
            pos = 0
            while pos < len(action) and action[pos] < size:
                pos += 1
            self.mlfqs[pos] += 1
        # print("MLFQ:", self.mlfq_step, self.mlfqs)
        self.mlfq_step.append([e for e in self.mlfqs])
    
    def _recordCoflowStatus(self, step, obs, com):
        obs = eval(obs.split(":")[-1])
        com = eval(com.split(":")[-1])
        ## 更新Coflow统计信息
        for e in obs:
            cid, width, size, duration = e
            if cid not in self.curIndex:
                self.curIndex.append(cid)
            self.coflows[cid]["sentsize"] = size
            self.coflows[cid]["duration"] = duration
        for e in com:
            cid, width, size, duration = e
            if cid not in self.curIndex:
                self.curIndex.append(cid)
            self.coflows[cid]["sentsize"] = size
            self.coflows[cid]["totalsize"] = size
            self.coflows[cid]["duration"] = duration
        ## 更新每step信息
        N = len(self.coflows)
        comList = []
        for i in range(len(self.coflows)):
            if self.coflows[i]["sentsize"] == self.coflows[i]["totalsize"]:
                comList.append(self.coflows[i]["duration"])
        actList = [e[3] for e in obs]
        acct = (sum(comList)+sum(actList))/(len(comList)+len(actList)) # 毫秒
        self.steps.append({
            "completedNum": len(comList),
            "acct": acct/1024,
            "activeList": "%s"%([e[0]+1 for e in obs])
        })
        # print(self.steps[-1])
    
    def _genAction(self, obs):
        # ["SCF", "SEBF", "Aalo", "M-DRL", "CS-GAIL"]
        assert self.algo in ALGOS, "algo not in ALGOS"
        AALO_THRESHOLD = np.array([1.0485760E7*(10**i) for i in range(9)])
        if self.algo == "SCF":
            return None
        if self.algo == "SEBF":
            return None
        if self.algo == "Aalo":
            return AALO_THRESHOLD
        if self.algo == "M-DRL":
            if self.selectedModel == None:
                return AALO_THRESHOLD
            else:
                return self.curModel.getAction(obs)
        if self.algo == "CS-GAIL":
            if self.selectedModel == None:
                return AALO_THRESHOLD
            else:
                return np.array([1.0485760E7*(10**i) for i in range(9)])
    
    def _configJava(self):
        # Configure the jpype environment
        jarpath = os.path.join(os.path.abspath("."))
        startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=%s/target/coflowsim-0.2.0-SNAPSHOT.jar"%(jarpath), convertStrings=False)

    def startEnv(self):
        assert self.algo in ALGOS, "algo not in ALGOS"
        assert self.traceID > 0 and self.traceID <= len(INNER_TRACES), "trace not in INNER_TRACES"

        traceFile = INNER_TRACES[self.traceID-1]["path"]
        method = "dark"
        if self.algo == "SCF":
            method = "scf"
        if self.algo == "SEBF":
            method = "sebf"
        args = [method, "COFLOW-BENCHMARK", traceFile]
        CoflowGym = JClass("coflowsim.CoflowGym")
        gym = CoflowGym(args)
        print("Args:", args)
        if self.algo == "CS-GAIL":
            self.env = CoflowKDEEnv(gym, debug=False)
        else:
            self.env = CoflowSimEnv(gym, debug=False)
        
        self._initializeRecord()
    
    def _initializeRecord(self):
        traceFile = INNER_TRACES[self.traceID-1]["path"]
        time, mappers, reducers, shuffle_t = util.parse_benchmark(traceFile)
        N = len(time)
        self.coflows = []
        self.steps = []
        for i in range(N):
            self.coflows.append({
                "id": i+1,
                "arriveTime": int(time[i]/1024),
                "length": int(shuffle_t[i]*1024*1024/(mappers[i]*reducers[i])), # Byte
                "sentsize": 0,
                "totalsize": shuffle_t[i]*1024*1024, # Byte
                "duration": 0,
            })
        self.mlfqs = [0]*(self.env.action_space.shape[0]+1)
        self.mlfq_step = []
        self.curIndex = []

    def _configModel(self):
        if self.algo == "M-DRL":
            curModel = [e for e in MDRL_MODELS if e["model"] == self.selectedModel and e["datasource"] == self.traceID][0]
            print("curModel:", curModel)
            self.curModel = DDPGModel(self.env.action_space.shape[0],
                                    self.env.observation_space.shape[0],
                                    self.env.observation_space.shape[0],
                                    curModel["dir"]+curModel["model"])
        if self.algo == "CS-GAIL":
            curModel = [e for e in GAIL_MODELS if e["model"] == self.selectedModel][0]
            pass

    ## 系统重置
    def reset(self):
        # if not self.ending:
        #     return
        self.startEnv()
    
    def setTraceID(self, tid):
        self.traceID = tid
    
    def setAlgo(self, algo):
        if algo == "Aalo" or algo == "aalo":
            self.algo = "Aalo"
        else:
            if algo.upper() in ALGOS:
                self.algo = algo.upper()
        if algo == "M-DRL":
            self.models = [e["model"] for e in MDRL_MODELS if e["datasource"] == self.traceID]
            if len(self.models) == 0:
                self.selectedModel = None
            else:
                self.selectedModel = self.models[0]
        if algo == "CS-GAIL":
            self.models = [e["model"] for e in GAIL_MODELS if e["datasource"] == self.traceID]
            if len(self.models) == 0:
                self.selectedModel = None
            else:
                self.selectedModel = self.models[0]

    def setModel(self, model):
        if self.algo not in ["M-DRL", "CS-GAIL"]:
            return
        assert model in self.models or model == None, "Model selected not in list of models!"
        self.selectedModel = model

    def __cacheTraces(self):
        self.cacheTrace = []
        for trace in INNER_TRACES:
            sn, ln, sw, lw = util.classify_analyse(trace["path"])
            _, mappers, reducers, shuffle = util.parse_benchmark(trace["path"])
            shuffle = np.array(shuffle)
            metrics = (sn, ln, sw, lw)
            total_len = sum(([len(e) for e in metrics]))
            total_mbytes = sum(shuffle)
            fnums = [e1*e2 for e1, e2 in zip(mappers, reducers)]
            length = int(max([st/fn for st, fn in zip(shuffle, fnums)]))

            cnt_per = [round(len(e)/total_len, 2) for e in metrics]
            mbyte_per = [round(sum(shuffle[e])/total_mbytes, 4) for e in metrics]
            x, y = util.get_x_y(np.log10(shuffle))
            tdata = {
                "tid": trace["id"],
                "desc": trace["desc"],
                "distname": '分布',
                "dist": [{
                    "metric": 'Coflow数量',
                    "sn": '%d%%'%(cnt_per[0]*100),
                    "ln": '%d%%'%(cnt_per[1]*100),
                    "sw": '%d%%'%(cnt_per[2]*100),
                    "lw": '%d%%'%(cnt_per[3]*100),
                }, {
                    "metric": 'Coflow总字节数',
                    "sn": '%.2f%%'%(mbyte_per[0]*100),
                    "ln": '%.2f%%'%(mbyte_per[1]*100),
                    "sw": '%.2f%%'%(mbyte_per[2]*100),
                    "lw": '%.2f%%'%(mbyte_per[3]*100),
                }],
                "cdf": {
                    "data": [[ex, ey] for ex, ey in zip(x, y)]
                },
                "cnum": total_len,
                "fnum": sum(fnums),
                "length": Utils.formatByte(length),
                "mnum": 150,
            }
            # print(tdata)
            self.cacheTrace.append(tdata)

    def getTraces(self):
        if self.cacheTrace is None or len(self.cacheTrace) == 0:
            self.__cacheTraces()
        return self.cacheTrace

if __name__ == "__main__":
    pass
    collector = Collector(47)
    collector.run()
    print("ok")
    # collector.getTraces()