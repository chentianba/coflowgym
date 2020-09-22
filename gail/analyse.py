import matplotlib.pyplot as plt 

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
    benchmark = {
        "dark": 1314760.0,
        "sebf": 1072680.0,
        "fifo": 3963584.0
    }
    with open(file, "r") as f:
        line = f.readline()
        results = []
        while line:
            if line.startswith("Result"):
                result = eval(line.split(":")[-1])
                if result > 0:
                    results.append(result)
            line = f.readline()
        plt.plot(results, ".-")
        plt.plot([0, len(results)], [benchmark["dark"]]*2, "orange")
        plt.plot([0, len(results)], [benchmark["sebf"]]*2, "r")
        plt.plot([0, len(results)], [benchmark["fifo"]]*2, "cyan")
        plt.legend(["GAIL", "Aalo", "SEBF", "FIFO"])
        plt.xlabel("Episode")
        plt.ylabel("Total CCT")
        plt.show()

if __name__ == "__main__":
    pass
    # analyse_reward_list()
    analyse_result_log()