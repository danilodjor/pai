import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import resample


x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)

#x, y = resample(x, y, n_samples=1000)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_trisurf(x[:,0], x[:,1], y, cmap='viridis')
#ax.scatter(x[:,0], x[:,1], y.max() + 10)

plt.show()
