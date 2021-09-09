file = 'outputData.txt'

with open(file) as f, open('new.txt', 'w') as w:
    for line in f.read().splitlines():
        if 'epsilon-greedy-t1' not in line:
            w.write(line.join(', ') + '\n')
