import matplotlib.pyplot as plt
from pyevtk.hl import gridToVTK
import numpy as np
import multiprocessing

from acoustics_solver_3d import NaiveAcousticsSolver3D

# geometry sizes
num_points_x = 35
num_points_y = 35
num_points_z = 65
space_step = 10
x_size = space_step * (num_points_x - 1)
y_size = space_step * (num_points_y - 1)
z_size = space_step * (num_points_z - 1)
# wavelength -- no less than 4 * space_step
wavelength = 200

# signal recording parameters
total_signal_recording_time = 0.6
time_step_between_records = 0.0015

# rheology limits
rho_min = 2000
rho_max = 5000
cp_min = 2000
cp_max = 4000

dump_dir = "."
###
emitter_radius = 3.858
reflector_radius = 5.
cut_offset = 1.
cut_radius = 13.


def f(R):
    # Option 1: create manually a mask for simple bright inclusion
    mask = np.zeros(shape=(num_points_x, num_points_y, num_points_z), dtype=float)
    relfector_shape = np.mgrid[-0.5 * num_points_x:0.5 * num_points_x,
                        -0.5 * num_points_y:0.5 * num_points_y,
                        0:num_points_z] * space_step
    mask += (np.sqrt(relfector_shape[0]**2 + relfector_shape[1]**2) < R[1] * wavelength / (2 * np.pi)) * \
            (np.sqrt(relfector_shape[0]**2 + relfector_shape[1]**2 + \
                    (relfector_shape[2] - R[2] * wavelength / (2 * np.pi))**2) >= R[3] * wavelength / (2 * np.pi))
    
    outer_body = (relfector_shape[0] - 0.7 * R[1] * wavelength / (2 * np.pi))**2 + relfector_shape[1]**2 + \
                    (relfector_shape[2] - 0.5 * R[3] * wavelength / (2 * np.pi))**2 < (2. * wavelength / (2 * np.pi))**2
    # mask += outer_body * 0.05

    # # Option 2: get (0; 1] mask representing layered structure
    # # WARN: take a look first at https://github.com/avasyukov/quasi_marmousi and review the parameters
    # mask = get_mask(num_points_x, num_points_y, 5, 8, 3, 10)

    # acoustics parameters of media from the mask created above
    cp = cp_min + (cp_max - cp_min) * mask
    rho = rho_min + (rho_max - rho_min) * mask

    # excitation pulse space size
    source_width = R[0] * wavelength / np.pi
    source_length = 4 * time_step_between_records

    solver = NaiveAcousticsSolver3D(x_size, y_size, z_size, cp, rho,
                                    total_signal_recording_time, time_step_between_records, wavelength,
                                    source_width, source_length,
                                    source_center_in_percents_x=50, source_center_in_percents_y=50,
                                    verbose=False, dump_vtk=True, dump_dir=dump_dir)
    signal, potential = solver.forward()
    threshold = int((R[2] + R[3]) * wavelength / (2 * np.pi) / space_step - 0.5 * wavelength / space_step)
    return np.min(potential[:,:,threshold:threshold+10])

def grad_f(R, f0, h=0.05):
    print("Calculate d/dx...")
    # ddx = (f(R + np.array([h, 0., 0., 0.])) - f0) / h
    ddx = 0.
    print("Calculate d/dy...")
    ddy = (f(R + np.array([0., h, 0., 0.])) - f0) / h
    print("Calculate d/dz...")
    # ddz = (f(R + np.array([0., 0., h, 0.])) - f0) / h
    ddz = 0.
    print("Calculate d/dxi...")
    ddxi = (f(R + np.array([0., 0., 0., h])) - f0) / h
    return np.array([ddx, ddy, ddz, ddxi])

def gradient_descent(R0, h=0.3, fname=None):
    R = R0
    alpha = 10.
    f_prev = 1.
    f0 = 0.
    parameters_log = [np.copy(R)]
    while np.abs(f0 - f_prev) > 1e-6:
        print("\nResidual:", np.abs(f0 - f_prev))
        f_prev = f0
        print("Calculate f0...", end=' ')
        f0 = f(R)
        print(f0, "at", R)
        g = grad_f(R, f0)
        print("Gradient:", g)
        print("Norm:", np.linalg.norm(g))
        R -= alpha * g
        parameters_log.append(np.copy(R))
        if fname:
            np.savetxt(fname, parameters_log, header=f'# h {h} alpha {alpha}')
    return R, f0

def gradient_descent_pooled(R0, h=0.3, fname=None):
    R = R0
    alpha = 20.
    f_prev = 1.
    f0 = 0.
    parameters_log = [np.copy(R)]
    pool = multiprocessing.Pool()
    while np.abs(f0 - f_prev) > 1e-6:
        print("\nResidual:", np.abs(f0 - f_prev))
        f_prev = f0
        print("Calculate f0...", end=' ')
        points = [1., 1., 1.]
        points[0] *= R
        points[1] = R + np.array([0., h, 0., 0.])
        points[2] = R + np.array([0., 0., 0., h])
        result = pool.map(func=f, iterable=points)
        f0 = result[0]
        print(f0, "at", R)
        g = np.array([0., (result[1] - result[0]) / h, 0., (result[2] - result[0]) / h])
        print("Gradient:", g)
        print("Norm:", np.linalg.norm(g))
        R -= alpha * g
        parameters_log.append(np.copy(R))
        if fname:
            np.savetxt(fname, parameters_log, header=f'# h {h} alpha {alpha}')
    return R, f0

if __name__ == '__main__':
    R = np.array([emitter_radius, reflector_radius, cut_offset, cut_radius])
    f = f(R)
    print(f)
    '''
    R0 = np.array([emitter_radius, reflector_radius, cut_offset, cut_radius])
    h = 2. * np.pi * space_step / wavelength
    R, f0 = gradient_descent_pooled(R0, h=h, fname='gd_log_2.csv')
    '''
    '''
    e = 3.8
    reflector_grid = np.linspace(4., 5.5, 31)
    h = 1.
    cut_radius_grid = np.linspace(3.5, 5., 31)

    potential_profile = []
    pool = multiprocessing.Pool()

    for r in reflector_grid:
        dump = []
        R = [np.array([e, r, h, c]) for c in cut_radius_grid[:8]]
        result = pool.map(func=f, iterable=R)
        for i in range(8):
            print(R[i], ":", result[i])
        dump.extend(result)

        R = [np.array([e, r, h, c]) for c in cut_radius_grid[8:16]]
        result = pool.map(func=f, iterable=R)
        for i in range(8):
            print(R[i], ":", result[i])
        dump.extend(result)

        R = [np.array([e, r, h, c]) for c in cut_radius_grid[16:24]]
        result = pool.map(func=f, iterable=R)
        for i in range(8):
            print(R[i], ":", result[i])
        dump.extend(result)

        R = [np.array([e, r, h, c]) for c in cut_radius_grid[24:]]
        result = pool.map(func=f, iterable=R)
        for i in range(7):
            print(R[i], ":", result[i])
        dump.extend(result)

        potential_profile.append(dump)
    
    pool.close()
    pool.join()
    np.savetxt('sector-44-madness.csv', potential_profile, header='gorkov')
'''
    
###
# gridToVTK(
#     os.path.join(dump_dir, "signal"),
#     np.linspace(0.0, num_points_x, num_points_x, endpoint=True),
#     np.linspace(0.0, num_points_y, num_points_y, endpoint=True),
#     np.linspace(0.0, signal.shape[2], signal.shape[2], endpoint=True),
#     pointData={"signal": signal.T.ravel()}
# )

# Draw the mask and the signal
# fig, ax = plt.subplots(nrows=2, ncols=1)

# ax[0].set_aspect('equal')
# im0 = ax[0].pcolormesh(np.flip(mask[45, :, :].T, 0))
# fig.colorbar(im0, ax=ax[0], orientation='horizontal')
# ax[0].set_title("mask")

# im1 = ax[1].contour(np.flip(potential[int(num_points_x / 2), :, :].T, 0))
# fig.colorbar(im1, ax=ax[1], orientation='horizontal')
# ax[1].set_title("Gor\'kov potential")
# plt.tight_layout()

# Draw the Gor'kov potential
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# U = np.flip(potential[int(num_points_x / 2), :, :], 0)
# X, Y = np.mgrid[0:U.shape[0], 0:U.shape[1]]

# U_profile = ax.plot_surface(X, Y, U, cmap=cm.coolwarm)

# fig.colorbar(U_profile)

# Draw 3D parameter gradient descent plot
# parameters_log = np.array(parameters_log)
# print(parameters_log)

# ax = plt.figure().add_subplot(projection='3d')

# ax.plot(parameters_log[:,1], parameters_log[:,2], parameters_log[:,3])

# plt.show()
