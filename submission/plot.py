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

infile = parser.parse_args().file
assert os.path.isfile(infile), "invalid file path specified"
task   = infile.split("-")[0][1]

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
    "highs"       : int(args[8])
  }

if task == "1":
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
      out = parse(line)
      ins = out["instance"].split("/")[-1].split("-")[1][0]
      al  = out["algorithm"]
      hz  = out["horizon"]
      rs  = out["seed"]
      plotobj[ins][al][hz][rs] += out["regret"]
  
  for ins, insobj in plotobj.items():
    count       += 1
    figure       = plt.figure(count)

    for al, hzobj in insobj.items():
      hz     = list(hzobj.keys())
      regret = list(map(lambda obj: sum(obj.values()) / len(obj), hzobj.values()))
      plt.plot(hz, regret, label=al, marker='o')

    plt.title(f'Task {task}, Instance {ins}')
    plt.xlabel('Horizon')
    plt.ylabel('Avg Regret')
    plt.xscale("log")
    plt.legend()
    plt.savefig('../' + 'Task1 -> Instance{ins}.jpeg')



if task == "2":
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
      c   = out["scale"]
      rs  = out["seed"]
      plotobj[ins][c][rs] += out["regret"]

  for ins, insobj in plotobj.items():
    scales    = list(insobj.keys())
    regret    = list(map(lambda obj: sum(obj.values()) / len(obj), insobj.values()))
    plt.plot(scales, regret, label=f'instance {ins}', marker='o')

  plt.title(f"Task {task}, Algo {out['algorithm']}")
  plt.xlabel("Scale")
  plt.ylabel("Avg Regret")
  plt.legend()


if task == "3":
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
      plt.plot(hz, regret, label=f'scale {c}', marker='o')

    plt.title(f'Task {task}, Instance {ins}')
    plt.xlabel('Horizon')
    plt.ylabel('Avg Regret')
    plt.xscale("log")
    plt.legend()


if task == "4":
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

      plt.plot(hz, regret, marker='o')
      plt.title(f'Task4 Instance{ins} Threshold = {th}')
      plt.xlabel('Horizon')
      plt.ylabel('Average HIGHS-REGRET')
      plt.xscale("log")
