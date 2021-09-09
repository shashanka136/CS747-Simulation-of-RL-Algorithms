import os
import argparse
from collections import defaultdict
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description="plot file data")

parser.add_argument(
  "--file", type=str, 
  metavar="file", default=None, 
  help="path to the file to be plotted"
)
parser.add_argument(
  "--task", type=int, default = 1
)


infile = parser.parse_args().file
assert os.path.isfile(infile), "invalid file path specified"
task = parser.parse_args().task

def parse(line):
  args = line.split(", ")
  return {
    "instance"    : args[0],
    "algorithm"   : args[1],
    "seed"        : int(args[2]),
    "epsilon"     : float(args[3]),
    "scale"       : float(args[4]),
    "threshold"   : float(args[5]),
    "horizon"     : int(args[6]),
    "regret"      : float(args[7]),
    "highs"       : float(args[8])
  }

if task == 1:
  # instance -> algorithm -> horizon -> seed
  plotobj = defaultdict(
    lambda : defaultdict(
      lambda : defaultdict(
        lambda : defaultdict(
          lambda : 0
        )
      )
    )
  )

  count   = 0

  with open(infile, 'r') as f:
    for line in f:
      print(line)
      out = parse(line)
      ins = out["instance"].split("/")[-1].split("-")[1][0]
      al  = out["algorithm"]
      if int(al[-1]) != task:
        continue
      hz  = out["horizon"]
      rs  = out["seed"]
      plotobj[ins][al][hz][rs] += out["regret"]
  
  for ins, insobj in plotobj.items():
    count       += 1
    figure       = plt.figure(count)

    for al, hzobj in insobj.items():
      hz     = list(hzobj.keys())
      regret = list(map(lambda obj: sum(obj.values()) / len(obj), hzobj.values()))
      plt.plot(hz, regret, label=al, linestyle = '-', marker='o')

    plt.title(f'Task{task} -> Instance{ins}')
    plt.xlabel('Horizon')
    plt.ylabel('Average Regret')
    plt.xscale('log')
    plt.legend()
    plt.savefig(f'../Task1 -> Instance{ins}.jpeg')



if task == 2:
  # instance -> scale -> seed
  plotobj = defaultdict(
    lambda : defaultdict(
      lambda : defaultdict(
        lambda : 0
      )
    )
  )

  with open(infile, 'r') as f:
    for line in f:
      out = parse(line)
      ins = out["instance"].split("/")[-1].split("-")[1][0]
      al  = out["algorithm"]
      if int(al[-1]) != task:
        continue
      c   = out["scale"]
      rs  = out["seed"]
      plotobj[ins][c][rs] += out["regret"]

  for ins, insobj in plotobj.items():
    scales    = list(insobj.keys())
    regret    = list(map(lambda obj: sum(obj.values()) / len(obj), insobj.values()))
    plt.plot(scales, regret, label=f'instance {ins}', linestyle = '-', marker='o')

  plt.title('Task2 -> ucb-t2 for 1e4 Horizon')
  plt.xlabel('Scale')
  plt.ylabel('Average Regret')
  plt.legend()
  plt.savefig('../Task2.jpeg')


if task == 3:
  # instance -> scale -> horizon -> seed
  plotobj = defaultdict(
    lambda : defaultdict(
      lambda : defaultdict(
        lambda : defaultdict(
          lambda : 0
        )
      )
    )
  )

  count   = 0

  with open(infile, 'r') as f:
    for line in f:
      out = parse(line)
      ins = out["instance"].split("/")[-1].split("-")[1][0]
      al  = out["algorithm"]
      if int(al[-1]) != task:
        continue
      c   = out["scale"]
      hz  = out["horizon"]
      rs  = out["seed"]
      plotobj[ins][c][hz][rs] += out["regret"]
  
  for ins, insobj in plotobj.items():
    count       += 1
    figure       = plt.figure(count)

    for c, hzobj in insobj.items():
      hz     = list(hzobj.keys())
      regret = list(map(lambda obj: sum(obj.values()) / len(obj), hzobj.values()))
      plt.plot(hz, regret, linestyle = '-', marker='o')

    plt.title(f'Task{task} Instance{ins}')
    plt.xlabel('Horizon')
    plt.ylabel('Average Regret')
    plt.xscale('log')
    plt.savefig(f'../Task3_Instance{ins}.jpeg')


if task == 4:
  # instance -> threshold -> horizon -> seed
  plotobj = defaultdict(
    lambda : defaultdict(
      lambda : defaultdict(
        lambda : defaultdict(
          lambda : 0
        )
      )
    )
  )

  count   = 0

  with open(infile, 'r') as f:
    for line in f:
      out = parse(line)
      ins = out["instance"].split("/")[-1].split("-")[1][0]
      al  = out["algorithm"]
      if int(al[-1]) != task:
        continue
      print(line)
      th  = out["threshold"]
      hz  = out["horizon"]
      rs  = out["seed"]
      plotobj[ins][th][hz][rs] += out["regret"]
  
  for ins, insobj in plotobj.items():
    for th, hzobj in insobj.items():
      count       += 1
      figure       = plt.figure(count)

      hz     = list(hzobj.keys())
      regret = list(map(lambda obj: sum(obj.values()) / len(obj), hzobj.values()))

      plt.plot(hz, regret, linestyle = '-', marker='o')
      plt.title(f'Task4 Instance{ins} Threshold = {th}')
      plt.xlabel('Horizon')
      plt.ylabel('Average HIGHS-REGRET')
      plt.xscale('log')
      plt.savefig(f'../Task4_Instance{ins}_Threshold{th}.jpeg')
