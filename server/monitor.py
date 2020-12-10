import threading, psutil
import time
import numpy as np

class Utils:
    @staticmethod
    def tsToHMS(a, hms="%H:%M:%S"):
        return time.strftime(hms, time.localtime(a))

    @staticmethod
    def getHMS(ts):
        """
        Return (Hour, Minute, Second)
        """
        x = time.localtime(ts)
        return x.tm_hour, x.tm_min, x.tm_sec
    
    @staticmethod
    def formatByte(mb):
        """
        将MB转化成MB、GB或者TB
        """
        if mb < 1024:
            return Utils.composeString(mb, "MB")
        gb = mb/1024
        if gb < 1024:
            return Utils.composeString(gb, "GB")
        tb = gb/1024
        return Utils.composeString(tb, "TB")
    
    @staticmethod
    def formatFromByte(byte):
        """
        将B转化成B、KB、MB、GB或者TB字符串
        """
        if byte < 1024:
            return Utils.composeString(byte, "B")
        kb = byte/1024
        if kb < 1024:
            return Utils.composeString(kb, "MB")
        return Utils.formatByte(kb/1024)
    
    @staticmethod
    def composeString(num, suffix):
        two = int((num - int(num))*100)
        if two == 0:
            return "%d%s"%(round(num), suffix)
        if two%10 == 0:
            return "%.1f%s"%(num, suffix)
        return "%.2f%s"%((num), suffix)

class Monitor(threading.Thread):
    SYSTEM_TIMESTAMP = 30 #单位秒
    MAX_LEN = 10

    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.setDaemon(True)

        self.threadID = threadID
        self.num = 0
        self.qs = self.COFLOW_DATA = self.CoflowRequest = self.GCCT = self.STEP = None
        self.SYSTEM = None
        self.clock_s = time.time()

        self.globalReset()

        ## Class Variables

    def run(self):
        print("开始线程: ", self.threadID)
        while True:
            self._getSystem()
        print("退出线程: ", self.threadID)
    
    def _getSystem(self):
        now = int(time.time()) # 单位为秒
        _, _, sec = Utils.getHMS(now)
        if now > self.clock_s and sec%self.SYSTEM_TIMESTAMP == 0:
            self.clock_s = now
            cpu = psutil.cpu_percent(interval=1)
            ## TODO: bw
            bw = 100-now%3
            self.SYSTEM["time"].append(now)
            self.SYSTEM["bw"].append(bw)
            self.SYSTEM["cpu"].append(cpu)
    
    def getSystem(self):
        data = {
            "time": [int(e*1000) for e in self.SYSTEM["time"]],
            "bw": self.SYSTEM["bw"],
            "cpu": self.SYSTEM["cpu"]
        }
        return data
    
    def globalReset(self):
        self.qs = [[],[],[],[],[],[],[],[],[],[]]
        self.STEP = self.CoflowRequest = 1
        self.COFLOW_DATA = []
        self.GCCT = {
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
        self.SYSTEM = {
            "time": [], # 时间戳格式
            "bw": [], # 百分比
            "cpu": [] # 百分比
        }
        self.clock_s = time.time()

    def resetEnv(self):
        pass

if __name__ == "__main__":
    print(Utils.composeString(10/3,"MB"))
    print(Utils.composeString(10/4,"MB"))
    print(Utils.composeString(10/2,"MB"))
    exit(0)
    pass
    monitor = Monitor(12)
    monitor.run()

    tbl = 1606286324
    N = 200
    monitor.SYSTEM["time"] = list(range(tbl, tbl+N))
    monitor.SYSTEM["bw"] = [0]*N
    monitor.SYSTEM["cpu"] = [0]*N

    monitor.getSystem()
