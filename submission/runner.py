from bandit import BernoulliBandit
import matplotlib.pyplot as plt

algos = ['epsilon-greedy-t1', 'ucb-t1', 'kl-ucb-t1',
         'thompson-sampling-t1', 'ucb-t2', 'alg-t3', 'alg-t4']

horizons = [100, 400, 1600, 6400, 25600, 102400]

# Task 1:
for i in range(1, 4):
    instance = "../instances/instances-task1/i-{}.txt".format(i)
    banditSample = BernoulliBandit(instance)
    for algo in algos[0:4]:
        for horizon in horizons:
            for seed in range(50):
                regret, high = banditSample.runAlgo(
                    algo, seed, 0.02, 2, 0, horizon)

# # Task 2:
for i in range(1, 6):
    instance = "../instances/instances-task2/i-{}.txt".format(i)
    banditSample = BernoulliBandit(instance)
    for scale in range(1, 16):
        for seed in range(50):
            regret, high = banditSample.runAlgo(
                'ucb-t2', seed, 0.02, 0.02*scale, 0, 10000)

# # Task 3:
for i in range(1, 3):
    instance = "../instances/instances-task3/i-{}.txt".format(i)
    banditSample = BernoulliBandit(instance)
    for horizon in horizons:
        for seed in range(50):
            regret, high = banditSample.runAlgo(
                'alg-t3', seed, 0.02, 0.225, 0, horizon)

# Task 4:
for i in range(1, 3):
    instance = "../instances/instances-task4/i-{}.txt".format(i)
    banditSample = BernoulliBandit(instance)
    for threshold in [0.2, 0.6]:
        for horizon in horizons:
            for seed in range(50):
                regret, high = banditSample.runAlgo(
                    'alg-t4', seed, 0.02, 2, threshold, horizon)
