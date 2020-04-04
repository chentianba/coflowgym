import numpy as np
from datetime import datetime

def get_h_m_s(second):
    """
    transform from second to hour-minite-second
    """
    if second <= 0:
        return 0, 0, 0
    m, s = divmod(round(second), 60)
    h, m = divmod(m, 60)
    return "%sH %sM %sS"%(h, m, s)

def get_now_time():
    now = datetime.now()
    return "%s-%s-%s-%s-%s-%s"%(now.year, now.month, now.day, now.hour, now.minute, now.second
)

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

if __name__ == "__main__":
    print(cal_limit("scripts/FB2010-1Hr-150-0.txt")) # result is ([1, 21170], [1.0, 8501205.0]) MB
