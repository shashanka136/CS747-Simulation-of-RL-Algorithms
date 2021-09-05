import numpy as np
import matplotlib.pyplot as plt
import argparse

def extract_probs1(instance):
	bandit_arms = []
	with open(instance) as file:
		line = file.readline().lstrip().rstrip()
		rs = [float(r) for r in line.split(' ')]
		while True:
			try:
				line = file.readline().lstrip().rstrip()
				probs = [float(prob) for prob in line.split(' ')]
			except:
				break
			# print(rs, probs)
			bandit_arms += [arm(rs,probs)]
	return bandit_arms

def extract_probs(instance):
	if instance.find('instances-task3') != -1 or instance.find('instances-task4') != -1:
		return extract_probs1(instance)
	
	bandit_arms = []
	with open(instance) as file:
		while True:
			try:
				line = float(file.readline().lstrip().rstrip())
			except:
				break
			rs = [0, 1]
			probs = [1 - line, line]
			# print(rs, probs)
			bandit_arms += [arm(rs,probs)]
	return bandit_arms

class arm:
	def __init__(self, rs, probs):
		self.rewards = rs
		self.probs = probs
		self.n = len(rs)

	def pull(self):
		rnd = np.random.random_sample()
		cur = 0
		for i, prob in enumerate(self.probs):
			cur += prob
			if rnd < cur:
				return self.rewards[i]
		return self.rewards[self.n-1]

class multiarm_bandit:
	def __init__(self, list_arm):
		self.arms = list_arm
		self.n = len(self.arms)
		self.pulls = np.zeros(self.n)
		self.emp_mean = np.zeros(self.n)
	
	def backtozero(self):
		self.pulls = np.zeros(self.n)
		self.emp_mean = np.zeros(self.n)

	def pull_arm(self,i):
		self.emp_mean[i] = (self.emp_mean[i]* self.pulls[i] + self.arms[i].pull())/(self.pulls[i]+1)
		self.pulls[i] += 1

	def pull_uniform(self):
		rnd = np.random.choice(self.n)
		self.pull_arm(rnd)
	
	def eps_greedy(self, eps, hz, seed):
		np.random.seed(seed)
		for _ in range(hz):
			rnd = np.random.random_sample()
			if rnd < eps:
				self.pull_uniform()
			else:
				indexes = np.flatnonzero(self.emp_mean == np.max(self.emp_mean))
				rnd = np.random.choice(len(indexes))
				self.pull_arm(indexes[rnd])

		



if __name__ == '__main__':
	# parser = argparse.ArgumentParser()

	# parser.add_argument()
	# parser.add_argument('--instance', type=str)
	# parser.add_argument('--algorithm', type=str)
	# parser.add_argument('--randomSeed', type=int)
	# parser.add_argument('--epsilon', type=float)
	# parser.add_argument('--scale', type=float)
	# parser.add_argument('--threshold', type=float)
	# parser.add_argument('--horizon', type=int)
	# args = parser.parse_args()
	# instance = args.instance
	# algo = args.algorithm
	# seed = args.randomSeed
	# eps = args.epsilon
	# c = args.scale
	# th = args.threshold
	# hz = args.horizon
	# np.random.seed(seed)
	extract_probs("../instances/instances-task1/i-1.txt")
	extract_probs("../instances/instances-task2/i-1.txt")
	extract_probs("../instances/instances-task3/i-1.txt")
	extract_probs("../instances/instances-task4/i-1.txt")