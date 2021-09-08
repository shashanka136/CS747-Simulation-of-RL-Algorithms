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
		self.emp_mean = np.zeros(self.bandit.n)
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
				# print(i, total_regret)
				regrets.append(total_regret)
				j += 1
		return regrets

class ucb:
	def __init__(self, bandit):
		self.bandit = bandit
		self.name = "ucb-t1" #used only when running task1
		self.emp_mean = np.zeros(self.bandit.n)
		self.ucb_val = np.zeros(self.bandit.n)
		self.c = 2.

	def reinit(self):
		self.emp_mean = np.zeros(self.bandit.n)
		self.ucb_val = np.zeros(self.bandit.n)
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
				# print(i, total_regret)
				regrets.append(total_regret)
				j += 1
		return regrets
		
class kl_ucb:
	def __init__(self, bandit):
		self.bandit = bandit
		self.name = "kl-ucb-t1"
		self.emp_mean = np.zeros(self.bandit.n)
		self.kl_ucb_val = np.zeros(self.bandit.n)

	def reinit(self):
		self.emp_mean = np.zeros(self.bandit.n)
		self.kl_ucb_val = np.zeros(self.bandit.n)
		self.bandit.reinit()

	def kl(self, x, y):
		return x*np.log((x+1e-5)/(y+1e-5)) + (1.-x)*np.log((1.+1e-5-x)/(1.+1e-5-y))

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
		for i in range(min(hz, self.bandit.n)):
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
				# print(i, total_regret)
				regrets.append(total_regret)
				j += 1
		return regrets

class thompson_sampling:
	def __init__(self, bandit):
		self.bandit = bandit
		self.name = "thompson-sampling-t1"
		self.success = np.zeros(self.bandit.n)
		self.failure = np.zeros(self.bandit.n)

	def reinit(self):
		self.success = np.zeros(self.bandit.n)
		self.failure = np.zeros(self.bandit.n)
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
				# print(itr, total_regret)
				regrets.append(total_regret)
				j += 1
		return regrets

class algo_t4:
	def __init__(self, bandit, th):
		self.name = "alg-t4"
		self.th = th
		list_arm = []
		for bandit_arm in bandit.arms:
			p = np.sum(bandit_arm.probs*(bandit_arm.rewards > th))
			# print(p, th, bandit_arm.probs, bandit_arm.rewards)
			list_arm.append(arm([0., 1.], [1.-p, p]))
		self.bandit = multiarm_bandit(list_arm)
		self.success = np.zeros(self.bandit.n)
		self.failure = np.zeros(self.bandit.n)

	def reinit(self):
		self.success = np.zeros(self.bandit.n)
		self.failure = np.zeros(self.bandit.n)
		self.bandit.reinit()

	def pull_arm(self,i):
		rew = self.bandit.arms[i].pull()
		if rew > 0.5:
			self.success[i] += 1
		else:
			self.failure[i] += 1
		self.bandit.pulls[i] += 1
		return rew
	
	def algo(self, hz, seed, hzs = []):
		if len(hzs) == 0:
			hzs = [hz]
		self.reinit()
		np.random.seed(seed)
		total_regret = 0.
		total_highs = 0.
		regrets =[]
		highs = []
		j = 0
		x = np.zeros(self.bandit.n)
		for itr in range(hz):
			for i in range(self.bandit.n):
				x[i] = np.random.beta(self.success[i]+1, self.failure[i]+1)
			rnd = np.argmax(x)
			rew = self.pull_arm(rnd)
			total_highs += rew
			total_regret += self.bandit.pstar - rew
			if j < len(hzs) and hzs[j] == itr+1:
				# print(itr, total_regret)
				regrets.append(total_regret)
				highs.append(total_highs)
				j += 1
		return regrets, highs

def run_t1(bandit,instance_num):
	algos = [eps_greedy, ucb, kl_ucb, thompson_sampling]
	algos = [algo(bandit) for algo in algos]
	hzs = [100, 400, 1600, 6400, 25600, 102400]
	seeds = [i for i in range(50)]
	regrets = np.zeros((len(algos), len(hzs)))
	output = instance(1,instance_num) + ', '
	fl = open('../task1_instance' + str(instance_num)+ '_output.txt', 'w')
	for j,seed in enumerate(seeds):
		for i,alg in enumerate(algos):
			current = np.array(alg.algo(0.02, 2, hzs[-1], seed, hzs))
			for k,hz in enumerate(hzs):
				print(output + alg.name + ', ' + str(seed) + ', ' + \
					str(0.02) + ', ' + str(2) + ', ' + str(0) + ', ' + \
					str(hz) + ', ' + str(current[k]) + ', 0', file=fl)
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

def run_t2():
	hz = 10000
	instance_nums = np.arange(1,6)
	alg = []
	for i in instance_nums:
		alg.append(ucb(multiarm_bandit(extract_probs(instance(2,i)))))
	for al in alg:
		al.name = 'ucb-t2'
	scales = np.arange(0.02, 0.31, 0.02)
	# print(scales)
	seeds = [i for i in range(50)]
	regrets = np.zeros((len(instance_nums), len(scales)))
	fl = open('../task2_output.txt', 'w')
	for ind in instance_nums:
		for j,seed in enumerate(seeds):
			for i,c in enumerate(scales):
				current = alg[ind-1].algo(0.02, c, hz, seed)
				current = current[0]
				regrets[ind-1][i] += current
				print(instance(2,ind)  + ', '+ alg[ind-1].name + ', ' + str(seed) + ', ' + \
					str(0.02) + ', ' + str(c) + ', ' + str(0) + ', ' + \
					str(hz) + ', ' + str(current) + ', 0', file = fl)
				print(j,i)
	regrets /= len(seeds)
	plt.figure()
	for ind in instance_nums:
		print(ind, regrets[ind-1])
		plt.plot(scales, regrets[ind-1], linestyle = '-', marker = 'o', label = 'Instance ' + str(ind))
	plt.xlabel('Scale')
	plt.ylabel('Average Regret')
	plt.title('Task2 -> ucb-t2 for 1e4 Horizon')
	plt.legend()
	plt.savefig('../Task2.jpeg')

def run_t3():
	instance_nums = np.arange(1,3)
	alg = []
	for i in instance_nums:
		alg.append(ucb(multiarm_bandit(extract_probs(instance(3,i)))))
	for al in alg:
		al.name = 'alg-t3'
	hzs = [100, 400, 1600, 6400, 25600, 102400]
	seeds = [i for i in range(50)]
	regrets = np.zeros((len(instance_nums), len(hzs)))
	fl = open('../task3_output.txt', 'w')
	for ind in instance_nums:
		for j,seed in enumerate(seeds):
			current = alg[ind-1].algo(0.02, 0.200, hzs[-1], seed, hzs)
			regrets[ind-1] += current
			for k,hz in enumerate(hzs):
				print(instance(3,ind) + ', ' + alg[ind-1].name + ', ' + str(seed) + ', ' + \
					str(0.02) + ', ' + str(.200) + ', ' + str(0) + ', ' + \
					str(hz) + ', ' + str(current[k]) + ', 0', file=fl)
			print(j,ind)
	regrets /= len(seeds)
	for ind in instance_nums:
		plt.figure()
		plt.plot(hzs, regrets[ind-1], linestyle = '-', marker = 'o')
		plt.xscale('log')
		plt.xlabel('Horizon')
		plt.ylabel('Average Regret')
		plt.title('Task3 Instance' + str(ind))
		plt.savefig('../Task3_Instance' + str(ind) + '.jpeg')

def run_t4():
	instance_nums = np.arange(1,3)
	alg = [[] for _ in instance_nums]
	ths = [0.2, 0.6]
	for i,ins in enumerate(instance_nums):
		for th in ths:
			alg[i].append(algo_t4(multiarm_bandit(extract_probs(instance(4,ins))), th))
	hzs = [100, 400, 1600, 6400, 25600, 102400]
	seeds = [i for i in range(50)]
	high_regrets = np.zeros((len(instance_nums), len(ths), len(hzs)))
	fl = open('../task4_output.txt', 'w')
	for ind in instance_nums:
		for j,th in enumerate(ths):
			for k,seed in enumerate(seeds):
				current, highs = alg[ind-1][j].algo(hzs[-1], seed, hzs)
				high_regrets[ind-1][j] += current
				for l,hz in enumerate(hzs):
					print(instance(4,ind) + ', ' + alg[ind-1][j].name + ', ' + str(seed) + ', ' + \
						str(0.02) + ', ' + str(2) + ', ' + str(th) + ', ' + \
						str(hz) + ', ' + str(current[l]) + ', ' + str(highs[l]), file=fl)
				print(j,ind,k)
	high_regrets /= len(seeds)
	for ind in instance_nums:
		for j,th in enumerate(ths):
			plt.figure()
			plt.plot(hzs, high_regrets[ind-1][j], linestyle = '-', marker = 'o')
			plt.xscale('log')
			plt.xlabel('Horizon')
			plt.ylabel('Average HIGHS-REGRET')
			plt.title('Task4 Instance' + str(ind) + ' Threshold = ' + str(th))
			plt.savefig('../Task4_Instance' + str(ind) + '_Threshold' + str(th)+'.jpeg')


if __name__ == '__main__':
	global output
	parser = argparse.ArgumentParser()

	parser.add_argument('--instance', type=str)
	parser.add_argument('--algorithm', type=str)
	parser.add_argument('--randomSeed', type=int)
	parser.add_argument('--epsilon', type=float, default = 0.02)
	parser.add_argument('--scale', type=float, default = 2)
	parser.add_argument('--threshold', type=float, default = 0)
	parser.add_argument('--horizon', type=int)
	args = parser.parse_args()
	instance = args.instance
	algo = args.algorithm
	seed = args.randomSeed
	eps = args.epsilon
	c = args.scale
	th = args.threshold
	hz = args.horizon
	bandit = multiarm_bandit(extract_probs(instance))
	output = instance + ', ' + algo + ', ' + \
			str(seed) + ', ' + str(eps) + ', ' + \
			str(c) + ', ' + str(th) + ', ' + str(hz) + ', '
	regret = []
	highs = [0]
	if algo == "epsilon-greedy-t1":
		regret = eps_greedy(bandit).algo(eps, c, hz, seed)
	elif algo == "ucb-t1" or algo == "ucb-t2":
		regret = ucb(bandit).algo(eps, c, hz, seed)
	elif algo == "kl-ucb-t1":
		regret = kl_ucb(bandit).algo(eps, c, hz, seed)
	elif algo == "thompson-sampling-t1":
		regret = thompson_sampling(bandit).algo(eps, c, hz, seed)
	elif algo == "alg-t3":
		regret = ucb(bandit).algo(eps, 0.2, hz, seed)
	elif algo == "alg-t4":
		regret, highs = algo_t4(bandit, th).algo(hz, seed)
		pass
	# print(regret)
	output += str(regret[0]) + ', '+ str(highs[0]) 
	print(output)
	# run_t4()
	# for i in range(1,4):
	# 	bandit = multiarm_bandit(extract_probs(instance(1,i)))
	# 	run_t1(bandit, i)
	# for i in range(1,2):
	# 	bandit = multiarm_bandit(extract_probs(instance(2,i)))
	# 	run_t2(bandit, i)
	# 	plt.show()
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
