file1 = 'outputData.txt'
file2 = '../task1_instance1_output.txt'
with open(file1) as f1, open(file2) as f2, open('new.txt', 'w') as w:
    for line in f1.read().splitlines():
        if 'epsilon-greedy-t1' not in line:
            w.write(line.join(', ') + '\n')
        else:
            w.write(f2.readline())
