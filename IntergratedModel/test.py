
import numpy as np
import matplotlib.pyplot as plt

plt.axis([0, 100, 0, 1])
plt.ion()

xs = [0, 0]
ys = [1, 1]

for i in range(100):
    y = np.random.random()
    xs[0] = xs[1]
    ys[0] = ys[1]
    xs[1] = i
    ys[1] = y
    plt.plot(xs, ys)
    # plt.pause(0.1)