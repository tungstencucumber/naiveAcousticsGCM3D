import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
'''
potential = np.loadtxt('gorkov.csv', dtype=np.float32, skiprows=1)
reflector_grid = np.arange(3., 3.7, 0.2)
cut_offset_grid = np.arange(-1., 4.1, 0.2)
cut_radius_grid = np.arange(10., 16.1, 0.2)
X, Y, Z = np.meshgrid(cut_offset_grid, reflector_grid, cut_radius_grid)
print(X.shape)

data = potential[:31*26*4].reshape(4,26,31)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.contourf(X[:, :, 0], Y[:, :, 0], data[:, :, 0], zdir='z', offset=Z.min(), levels=31)
ax.contourf(X[0, :, :], data[0, :, :], Z[0, :, :], zdir='y', offset=Y.min(), levels=31)
C = ax.contourf(data[:, -1, :], Y[:, -1, :], Z[:, -1, :], zdir='x', offset=X.max(), levels=31)

xmin, xmax = X.min(), X.max()
ymin, ymax = Y.min(), Y.max()
zmin, zmax = Z.min(), Z.max()
ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

ax.set(xlabel='Смещение выреза', ylabel='Радиус отражателя', zlabel='Радиус выреза')

fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1)

plt.show()
'''

z = np.hstack((np.loadtxt('reflector-cut-incursion-3.csv', dtype=np.float32, skiprows=1), np.loadtxt('rc-profile.csv', dtype=np.float32, skiprows=1))).T
r_grid = np.arange(3., 7.1, 0.5)
c_grid = np.arange(3., 16.1, 0.5)


z0 = np.loadtxt('sector-44-madness.csv', dtype=np.float32, skiprows=1).T
r_sub_grid = np.linspace(4., 5.5, 31)
c_sub_grid = np.linspace(3.5, 5., 31)
'''
potential = np.loadtxt('reflector-cut-incursion.csv', dtype=np.float32, skiprows=1)
reflector_grid = np.linspace(3.8, 4.1, 16)
cut_radius_grid = np.linspace(8.6, 9.0, 21)
m = reflector_grid.shape[0]
n = cut_radius_grid.shape[0]
z0 = potential.reshape(m,n).T

potential_2 = np.loadtxt('reflector-cut-incursion-2.csv', dtype=np.float32, skiprows=1)
reflector_grid_2 = np.linspace(4.1, 4.3, 11)
cut_radius_grid_2 = np.linspace(6.2, 7.2, 21)
m = reflector_grid_2.shape[0]
n = cut_radius_grid_2.shape[0]
z2 = potential_2.reshape(m,n).T
'''

gd = []
h = []
alphas = []
for i in range(3):
    filename = f'gd_log_{i}.csv'
    f = open(filename, 'r')
    line = f.readline().split(' ')
    h.append(float(line[3]))
    alphas.append(float(line[5]))
    gd.append(np.loadtxt(f, dtype=np.float32))
    f.close()

fig, ax = plt.subplots(figsize=(14,8))
norm = cm.colors.Normalize(vmin=z.min(), vmax=z.max())
cf = ax.pcolormesh(r_grid, c_grid, z, norm=norm)
cf0 = ax.pcolormesh(r_sub_grid, c_sub_grid, z0, norm=norm)
fig.colorbar(cf, ax=ax)
for i in range(3):
    ax.plot(gd[i][:,1], gd[i][:,3], '*-', label=fr'$\alpha$ = {alphas[i]}')
    ax.scatter(gd[i][0,1], gd[i][0,3])
ax.set(xlim=(2.75, 7.25), ylim=(2.75, 16.25))
'''
xbounds, ybounds = ax.get_xbound(), ax.get_ybound()
xmin, xmax, shape = ax.get_xticks()[0], ax.get_xticks()[-1], ax.get_xticks().shape[0]
xticks = np.linspace(xmin, xmax, shape * int(200. / 10. / 2. / np.pi))
ax.set_xticks(xticks, np.round(xticks, 2))
ymin, ymax, shape = ax.get_yticks()[0], ax.get_yticks()[-1], ax.get_yticks().shape[0]
yticks = np.linspace(ymin, ymax, shape * int(200. / 10. / 2. / np.pi))
ax.set_yticks(yticks, labels=np.round(yticks, 2) )
ax.grid(which='both', linestyle='--')
ax.set(xlim=(xbounds), ylim=(ybounds))
'''
ax.legend(loc='best', fontsize=14)
ax.set_title(r'$\alpha$ -- learning rate', fontsize=17)
ax.set_xlabel(r'Reflector base radius $R_b$', fontsize=16)
ax.set_ylabel(r'Spherical cut radius $R$', fontsize=16)
plt.savefig('reflector-cut-profile.png', dpi=500)
plt.show()

'''
ax = plt.figure().add_subplot(projection='3d')

ax.plot(parameters_log[:,1], parameters_log[:,2], parameters_log[:,3])
ax.set_zlabel('\n\n\nРадиус кривизны выреза в отражателе')

plt.savefig('gdc.png')
plt.show()
'''
