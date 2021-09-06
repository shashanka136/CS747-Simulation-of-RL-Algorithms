import numpy as np
import matplotlib.pyplot as plt
import argparse


def instance(j,i):
	return "../instances/instances-task" + str(j) + "/i-" + str(i) + ".txt"

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
		self.rewards = np.array(rs)
		self.probs = np.array(probs)
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
		self.arms = np.array(list_arm)
		self.n = len(self.arms)
		self.pulls = np.zeros(self.n)
		self.pstar = np.max([np.dot(x.rewards, x.probs) for x in self.arms])

	def reinit(self):
		self.pulls = np.zeros(self.n)

class eps_greedy:
	def __init__(self, bandit):
		self.bandit = bandit
		self.name = "epsilon-greedy-t1"
		self.emp_mean = np.zeros(bandit.n)

	def reinit(self):
		self.emp_mean = np.zeros(bandit.n)
		self.bandit.reinit()

	def pull_arm(self,i):
		rew = self.bandit.arms[i].pull()
		self.emp_mean[i] = (self.emp_mean[i]* self.bandit.pulls[i] + rew)/(self.bandit.pulls[i]+1)
		self.bandit.pulls[i] += 1
		return rew

	def pull_uniform(self):
		rnd = np.random.choice(self.bandit.n)
		return self.pull_arm(rnd)
	
	def algo(self, eps, c, hz, seed, hzs = []):
		if len(hzs) == 0:
			hzs = [hz]
		self.reinit()
		np.random.seed(seed)
		total_regret = 0.
		regrets = []
		j = 0
		itrloop = np.random.random(hz)
		# print(itrloop)
		for i in range(hz):
			if itrloop[i] < eps:
				total_regret += self.bandit.pstar - self.pull_uniform()
			else:
				rnd = np.argmax(self.emp_mean)
				total_regret += self.bandit.pstar - self.pull_arm(rnd)
			if j < len(hzs) and hzs[j] == i+1:
				print(i, total_regret)
				regrets.append(total_regret)
				j += 1
		return regrets

class ucb:
	def __init__(self, bandit):
		self.bandit = bandit
		self.name = "ucb-t1" #used only when running task1
		self.emp_mean = np.zeros(bandit.n)
		self.ucb_val = np.zeros(bandit.n)
		self.c = 2.

	def reinit(self):
		self.emp_mean = np.zeros(bandit.n)
		self.ucb_val = np.zeros(bandit.n)
		self.bandit.reinit()

	def calc_ucb(self, t):
		for i in range(self.bandit.n):
			self.ucb_val[i] = self.emp_mean[i] + np.sqrt(self.c * np.log(t)/self.bandit.pulls[i])

	def pull_arm(self,i):
		rew = self.bandit.arms[i].pull()
		self.emp_mean[i] = (self.emp_mean[i]* self.bandit.pulls[i] + rew)/(self.bandit.pulls[i]+1)
		self.bandit.pulls[i] += 1
		return rew
	
	def algo(self, eps, c, hz, seed, hzs = []):
		if len(hzs) == 0:
			hzs = [hz]
		self.reinit()
		self.c = c
		np.random.seed(seed)
		total_regret = 0.
		regrets = []
		j = 0
		order = np.arange(self.bandit.n)
		np.random.shuffle(order)
		print(self.bandit.pstar, c)
		# print(order)
		for i in range(min(hz, self.bandit.n)):
			# print(i)
			total_regret += self.bandit.pstar - self.pull_arm(order[i])
			if j < len(hzs) and hzs[j] == i+1:
				regrets.append(total_regret)
				j += 1
		if hz < self.bandit.n:
			return regrets
		for i in range(self.bandit.n, hz):
			self.calc_ucb(i)
			rnd = np.argmax(self.ucb_val)
			total_regret += self.bandit.pstar - self.pull_arm(rnd)
			if j < len(hzs) and hzs[j] == i+1:
				print(i, total_regret)
				regrets.append(total_regret)
				j += 1
		return regrets
		
class kl_ucb:
	def __init__(self, bandit):
		self.bandit = bandit
		self.name = "kl-ucb-t1"
		self.emp_mean = np.zeros(bandit.n)
		self.kl_ucb_val = np.zeros(bandit.n)

	def reinit(self):
		self.emp_mean = np.zeros(bandit.n)
		self.kl_ucb_val = np.zeros(bandit.n)
		self.bandit.reinit()

	def kl(self, x, y):
		return x*np.log(x/y) + (1.-x)*np.log((1-x)/(1-y))

	def calc_kl_ucb(self, t):
		rhs = np.log(t) + 3*np.log(np.log(t))
		for i in range(self.bandit.n):
			l, r = self.emp_mean[i], 1.
			while r-l > 1e-3:
				mid = (r+l)/2
				if self.bandit.pulls[i]*self.kl(self.emp_mean[i], mid) <= rhs:
					l = mid
				else:
					r = mid
			self.kl_ucb_val[i] = l

	def pull_arm(self,i):
		rew = self.bandit.arms[i].pull()
		self.emp_mean[i] = (self.emp_mean[i]* self.bandit.pulls[i] + rew)/(self.bandit.pulls[i]+1)
		self.bandit.pulls[i] += 1
		return rew
	
	def algo(self, eps, c, hz, seed, hzs = []):
		if len(hzs) == 0:
			hzs = [hz]
		regrets = []
		j = 0
		self.reinit()
		np.random.seed(seed)
		total_regret = 0.
		order = np.arange(self.bandit.n)
		np.random.shuffle(order)
		# print(order)
		for i in range(min(hz, self.bandit.n)):
			# print(i)
			total_regret += self.bandit.pstar - self.pull_arm(order[i])
			if j < len(hzs) and hzs[j] == i+1:
				regrets.append(total_regret)
				j += 1
		if hz < self.bandit.n:
			return regrets
		for i in range(self.bandit.n, hz):
			self.calc_kl_ucb(i)
			rnd = np.argmax(self.kl_ucb_val)
			total_regret += self.bandit.pstar - self.pull_arm(rnd)
			if j < len(hzs) and hzs[j] == i+1:
				print(i, total_regret)
				regrets.append(total_regret)
				j += 1
		return regrets

class thompson_sampling:
	def __init__(self, bandit):
		self.bandit = bandit
		self.name = "thompson-sampling-t1"
		self.success = np.zeros(bandit.n)
		self.failure = np.zeros(bandit.n)

	def reinit(self):
		self.success = np.zeros(bandit.n)
		self.failure = np.zeros(bandit.n)
		self.bandit.reinit()

	def pull_arm(self,i):
		rew = self.bandit.arms[i].pull()
		if rew > 0.5:
			self.success[i] += 1
		else:
			self.failure[i] += 1
		self.bandit.pulls[i] += 1
		return rew
	
	def algo(self, eps, c, hz, seed, hzs = []):
		if len(hzs) == 0:
			hzs = [hz]
		self.reinit()
		np.random.seed(seed)
		total_regret = 0.
		regrets =[]
		j = 0
		x = np.zeros(self.bandit.n)
		for itr in range(hz):
			for i in range(self.bandit.n):
				x[i] = np.random.beta(self.success[i]+1, self.failure[i]+1)
			rnd = np.argmax(x)
			total_regret += self.bandit.pstar - self.pull_arm(rnd)
			if j < len(hzs) and hzs[j] == itr+1:
				print(itr, total_regret)
				regrets.append(total_regret)
				j += 1
		return regrets

def run_t1(bandit,instance_num):
	algos = [eps_greedy, ucb, kl_ucb, thompson_sampling]
	algos = [algo(bandit) for algo in algos]
	hzs = [100, 400, 1600, 6400, 25600, 102400]
	seeds = [i for i in range(50)]
	regrets = np.zeros((len(algos), len(hzs)))
	output = instance(1,instance_num) + ', '
	fl = open('../task1_output.txt', 'w')
	for j,seed in enumerate(seeds):
		for i,alg in enumerate(algos):
			current = np.array(alg.algo(0.02, 2, hzs[-1], seed, hzs))
			for k,hz in enumerate(hzs):
				print(output + alg.name + ', ' + str(seed) + ', ' + \
					str(0.02) + ', ' + str(2) + ', ' + str(0) + ', ' + \
					str(hz) + ', ' + str(current[k]) + ', 0', file = fl)
			regrets[i]  += current
			print(j,i, alg.name)
	
	for i in range(len(algos)):
		regrets[i] /= len(seeds)
	plt.figure()
	for i,algo in enumerate(algos):
		plt.plot(hzs, regrets[i], linestyle = '-', marker = 'o', label = algo.name)
	
	plt.xscale('log')
	plt.xlabel('Horizon')
	plt.ylabel('Average Regret')
	plt.title('Task1 -> Instance' + str(instance_num))
	plt.legend()
	plt.savefig('../' + 'Task1 -> Instance' + str(instance_num) + '.jpeg')




if __name__ == '__main__':
	# global output
	# parser = argparse.ArgumentParser()

	# parser.add_argument('--instance', type=str)
	# parser.add_argument('--algorithm', type=str)
	# parser.add_argument('--randomSeed', type=int)
	# parser.add_argument('--epsilon', type=float, default = 0.02)
	# parser.add_argument('--scale', type=float, default = 2)
	# parser.add_argument('--threshold', type=float, default = 0)
	# parser.add_argument('--horizon', type=int)
	# args = parser.parse_args()
	# instance = args.instance
	# algo = args.algorithm
	# seed = args.randomSeed
	# eps = args.epsilon
	# c = args.scale
	# th = args.threshold
	# hz = args.horizon
	# bandit = multiarm_bandit(extract_probs(instance))
	# output = instance + ', ' + algo + ', ' + \
	# 		str(seed) + ', ' + str(eps) + ', ' + \
	# 		str(c) + ', ' + str(th) + ', ' + str(hz) + ', '
	# regret = []
	# highs = 0
	# if algo == "epsilon-greedy-t1":
	# 	regret = eps_greedy(bandit).algo(eps, c, hz, seed)
	# elif algo == "ucb-t1" or algo == "ucb-t2":
	# 	regret = ucb(bandit).algo(eps, c, hz, seed)
	# elif algo == "kl-ucb-t1":
	# 	regret = kl_ucb(bandit).algo(eps, c, hz, seed)
	# elif algo == "thompson-sampling-t1":
	# 	regret = thompson_sampling(bandit).algo(eps, c, hz, seed)
	# elif algo == "algo-t3":
	# 	pass
	# elif algo == "algo-t4":
	# 	pass
	# output += str(regret[0]) + ', '+ str(highs) 
	# print(output)
	for i in range(1,4):
		bandit = multiarm_bandit(extract_probs(instance(1,i)))
		run_t1(bandit, i)
	# print(bandit.eps_greedy_algo(0.61, 82, 0))
	# bandit = ucb(multiarm_bandit(extract_probs("../instances/instances-task1/i-3.txt")))
	# print(bandit.ucb_algo(82, 0))
	# bandit = kl_ucb(multiarm_bandit(extract_probs("../instances/instances-task1/i-3.txt")))
	# print(bandit.kl_ucb_algo(82, 0))
	# bandit = thompson_sampling(multiarm_bandit(extract_probs("../instances/instances-task1/i-3.txt")))
	# print(bandit.thompson_algo(82, 0))
	# extract_probs("../instances/instances-task2/i-1.txt")
	# extract_probs("../instances/instances-task3/i-1.txt")
	# extract_probs("../instances/instances-task4/i-1.txt")