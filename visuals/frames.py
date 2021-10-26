from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import special_ortho_group

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)



arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)
def plotFrame(T0f, ax):
    x0 = T0f[0,3]; y0 = T0f[1,3]; z0 = T0f[2,3]
    x0 = 0
    y0 = 0
    z0 = 0
    xw1  = T0f[0,0]; xw2  = T0f[1,0]; xw3  = T0f[2,0]
    yw1  = T0f[0,1]; yw2  = T0f[1,1]; yw3  = T0f[2,1]
    zw1  = T0f[0,2]; zw2  = T0f[1,2]; zw3  = T0f[2,2]
    a = Arrow3D(x0 + xw1*np.array([0, 1.0]), y0+xw2*np.array([0, 1.0]), z0+xw3*np.array([0, 1.0]), color='r', **arrow_prop_dict)
    ax.add_artist(a)
    a = Arrow3D(x0 + yw1*np.array([0, 1.0]), y0+yw2*np.array([0, 1.0]), z0+yw3*np.array([0, 1.0]), color='g', **arrow_prop_dict)
    ax.add_artist(a)
    a = Arrow3D(x0 + zw1*np.array([0, 1.0]), y0+zw2*np.array([0, 1.0]), z0+zw3*np.array([0, 1.0]), color='b', **arrow_prop_dict)
    ax.add_artist(a)




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# a = Arrow3D([0, 1], [0, 0], [0, 0], color='r', **arrow_prop_dict)
# ax.add_artist(a)
# a = Arrow3D(np.array([0, 1.0]), np.array([0, 1.0]), np.array([0, 1.0]), color='b', **arrow_prop_dict)
# ax.add_artist(a)

print(special_ortho_group.rvs(3))
T = np.random.rand(4,4)
T[:3,:3] = special_ortho_group.rvs(3)
print(T)
plotFrame(T, ax)

# a = Arrow3D([0, 0], [0, 0], [0, 1], color='g', **arrow_prop_dict)
# ax.add_artist(a)

# ax.text(0.0, 0.0, -0.1, r'$o$')
# ax.text(1.1, 0, 0, r'$x$')
# ax.text(0, 1.1, 0, r'$y$')
# ax.text(0, 0, 1.1, r'$z$')

# ax.view_init(azim=-90, elev=90)
# ax.set_axis_off()
plt.show()

