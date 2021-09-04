import numpy as np
import matplotlib.pyplot as plt
import argparse

def extract_probs(instance):
	pass

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
	



if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument()
	parser.add_argument('--instance', type=str)
	parser.add_argument('--algorithm', type=str)
	parser.add_argument('--randomSeed', type=int)
	parser.add_argument('--epsilon', type=float)
	parser.add_argument('--scale', type=float)
	parser.add_argument('--threshold', type=float)
	parser.add_argument('--horizon', type=int)
	args = parser.parse_args()
	instance = args.instance
	algo = args.algorithm
	seed = args.randomSeed
	eps = args.epsilon
	c = args.scale
	th = args.threshold
	hz = args.horizon
	np.random.seed(seed)

