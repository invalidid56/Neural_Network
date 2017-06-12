import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

x = np.arange(-5, 5, 0.25)  # points in the x axis
y = np.arange(-5, 5, 0.25)  # points in the y axis
X, Y = np.meshgrid(x, y)  # create the "base grid"
Z = X + Y  # points in the z axis

fig = plt.figure()
ax = fig.gca(projection='3d')  # 3d axes instance
surf = ax.plot_surface(X, Y, Z,  # data values (2D Arryas)
                       rstride=2,  # row step size
                       cstride=2,  # column step size
                       cmap=cm.RdPu,  # colour map
                       linewidth=1,  # wireframe line width
                       antialiased=True)

ax.set_title('Hyperbolic Paraboloid')  # title
ax.set_xlabel('x label')  # x label
ax.set_ylabel('y label')  # y label
ax.set_zlabel('z label')  # z label
fig.colorbar(surf, shrink=0.5, aspect=5)  # colour bar

ax.view_init(elev=30, azim=70)  # elevation & angle
ax.dist = 8  # distance from the plot
plt.show()
