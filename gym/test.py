if __name__ == "__main__":
    try:
        for _ in range(int(1e4)):
            res = 0
            for i in range(int(1e6)):
                res += i
            print(res)
    except KeyboardInterrupt as e:
        print("KeyboardInterrupt")