import numpy as np

f = open('plot_queue_data.txt', 'r')
content = f.read().split()
a = []
for i in content:
	a.append(int(i))

mean = np.mean(a)
maximum = np.amax(a, axis=0)
minimum = np.amin(a, axis=0)

print("mean:", mean)
print("max:", maximum)
print("min:", minimum)

