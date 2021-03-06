import matplotlib.pyplot as plt
import numpy as np
from ddpg import OUNoise

def compare_with_decay():
    a_dim = 1
    N = 1000
    ou = OUNoise(a_dim, mu=0.4)
    ou_states = []
    epsilon = 1
    EXPLORE = 70
    for ep in range(100):
        for i in range(N):
            ou_states.append(max(0.01, epsilon)*ou.noise())
        epsilon -= epsilon/EXPLORE
    random_states = []
    var = 3
    for ep in range(100):
        for i in range(N):
            random_states.append(np.random.normal(np.zeros(a_dim,), var))
            var *= 0.995

    plt.figure()
    plt.subplot(311)
    plt.plot(ou_states)
    plt.plot([1 for _ in range(N)])
    plt.plot([-1 for _ in range(N)])
    plt.plot([0 for _ in range(N)])
    plt.ylabel("OUNoise")
    # plt.figure()
    plt.subplot(312)
    plt.plot(random_states)
    plt.plot([1 for _ in range(N)])
    plt.plot([-1 for _ in range(N)])
    plt.ylabel("Radom Noise")

    ## 
    customed_states = []
    var = 3
    for ep in range(100):
        for i in range(N):
            customed_states.append(np.random.normal(np.zeros(a_dim,), var))
            if i % 300 == 0:
                var *= 0.995
    plt.subplot(313)
    plt.plot(customed_states)

def compare_without_decay():
    a_dim = 1
    N = 1000
    ou = OUNoise(a_dim, mu=0.4)
    ou_states = []
    for i in range(N):
        ou_states.append(ou.noise())
    random_states = []
    var = 1
    for i in range(N):
        random_states.append(np.random.normal(np.zeros(a_dim,), var))
        # var *= 0.995

    plt.figure()
    plt.subplot(211)
    plt.plot(ou_states)
    plt.plot([1 for _ in range(N)])
    plt.plot([-1 for _ in range(N)])
    plt.plot([0 for _ in range(N)])
    plt.ylabel("OUNoise")
    # plt.figure()
    plt.subplot(212)
    plt.plot(random_states)
    plt.plot([1 for _ in range(N)])
    plt.plot([-1 for _ in range(N)])
    plt.ylabel("Radom Noise")

if __name__ == "__main__":
    pass

    compare_with_decay()
    # compare_without_decay()
    plt.show()