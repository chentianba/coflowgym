import matplotlib.pyplot as plt 

benchmark = {
    "dark": 1314760.0,
    "sebf": 1072680.0,
    "fifo": 3963584.0
}
# benchmark = {
#     "dark": 2.4247392E7,
#     "sebf": 1.5005968E7, 
#     "fifo": 4.3473352E7, 
# }

DIR = "results/20201112T141844.150251_DDPG_GAIL/"

def analyse_reward_list():
    file = "log/log.txt"
    with open(file, "r") as f:
        line = f.readline()
        eps = []
        ep = []
        while line:
            if line.startswith("ep_steps"):
                start = line.find("[")
                end = line.find("]")
                ep.append(eval(line[start:end+1]))
            if line.startswith("Episode"):
                eps.append(ep)
                ep = []

            line = f.readline()
        print(len(eps))
        for ep in eps:
            print(len(ep), ep[:5])
            print()

def analyse_result_log():
    file = "doc/log/gail/2_log.txt"
    file = "log/%slog.txt"%(DIR)
    with open(file, "r") as f:
        line = f.readline()
        results = []
        while line:
            if line.startswith("Result"):
                result = eval(line.split(":")[-1])
                if result > 0:
                    results.append(result)
            line = f.readline()
        plot_result_with_ep(results)

def analyse_test_result_log():
    file = "log/%stest_log.txt"%(DIR)
    with open(file, "r") as f:
        line = f.readline()
        results = []
        while line:
            if line.startswith("Test/Result"):
                result = eval(line.split(":")[-1])
                if result > 0:
                    results.append(result)
            line = f.readline()
    plot_result_with_ep(results)

def plot_result_with_ep(results):
    plt.figure()
    plt.plot(results, ".-")
    plt.plot([0, len(results)], [benchmark["dark"]]*2, "orange")
    plt.plot([0, len(results)], [benchmark["sebf"]]*2, "r")
    plt.plot([0, len(results)], [benchmark["fifo"]]*2, "cyan")
    plt.legend(["GAIL", "Aalo", "SEBF", "FIFO"])
    plt.xlabel("Episode")
    plt.ylabel("Total CCT")

def analyse_ddpg_reward_log():
    file = "log/log_ddpg.txt"
    with open(file, "r") as f:
        line = f.readline()
        results = []
        rewards = []
        while line:
            if line.startswith("Result"):
                result = eval(line.split(":")[-1])
                if result > 0:
                    results.append(result)
            if line.find("ep_reward") != -1:
                reward = eval(line.split()[-1])
                rewards.append(reward)
            line = f.readline()
    
    plot_result_with_ep(results)

    plt.figure()
    plt.scatter(results, rewards)

if __name__ == "__main__":
    pass
    # analyse_result_log()

    # analyse_test_result_log()

    analyse_ddpg_reward_log()

    plt.show()
