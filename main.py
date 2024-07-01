import matplotlib.pyplot as plt
from pyevtk.hl import gridToVTK
import numpy as np
import os

from acoustics_solver_3d import NaiveAcousticsSolver3D

# geometry sizes
num_points_x = 80
num_points_y = 80
num_points_z = 100
space_step = 5
x_size = space_step * (num_points_x - 1)
y_size = space_step * (num_points_y - 1)
z_size = space_step * (num_points_z - 1)

wavelength = 200
dimensionless_emitter_radius = 3.858
dimensionless_reflector_radius = 5.092
dimensionless_reflection_surfaces_radius = 13.5

# rheology limits
rho_min = 2000
rho_max = 5000
cp_min = 2000
cp_max = 4000

# Option 1: create manually a mask for simple bright inclusion
mask = np.zeros(shape=(num_points_x, num_points_y, num_points_z), dtype=float)
relfector_shape = np.mgrid[-0.5 * num_points_x:0.5 * num_points_x,
                    -0.5 * num_points_y:0.5 * num_points_y,
                    0:num_points_z] * space_step
mask += (np.sqrt(relfector_shape[0]**2 + relfector_shape[1]**2) < dimensionless_reflector_radius * wavelength / (2 * np.pi)) * \
                    (np.sqrt(relfector_shape[0]**2 + relfector_shape[1]**2 + relfector_shape[2]**2) >= dimensionless_reflection_surfaces_radius * wavelength / (2 * np.pi))

# # Option 2: get (0; 1] mask representing layered structure
# # WARN: take a look first at https://github.com/avasyukov/quasi_marmousi and review the parameters
# mask = get_mask(num_points_x, num_points_y, 5, 8, 3, 10)

# acoustics parameters of media from the mask created above
cp = cp_min + (cp_max - cp_min) * mask
rho = rho_min + (rho_max - rho_min) * mask

# signal recording parameters
total_signal_recording_time = 0.9
time_step_between_records = 0.0015

# excitation pulse space size
source_width = dimensionless_emitter_radius * wavelength / np.pi
source_length = 4 * time_step_between_records

dump_dir = "."

solver = NaiveAcousticsSolver3D(x_size, y_size, z_size, cp, rho,
                                total_signal_recording_time, time_step_between_records, wavelength,
                                source_width, source_length,
                                source_center_in_percents_x=50, source_center_in_percents_y=50,
                                verbose=True, dump_vtk=True, dump_dir=dump_dir)
signal, potential = solver.forward()
print(signal.shape)

gridToVTK(
    os.path.join(dump_dir, "signal"),
    np.linspace(0.0, num_points_x, num_points_x, endpoint=True),
    np.linspace(0.0, num_points_y, num_points_y, endpoint=True),
    np.linspace(0.0, signal.shape[2], signal.shape[2], endpoint=True),
    pointData={"signal": signal.T.ravel()}
)

# Draw the mask and the signal
fig, ax = plt.subplots(nrows=2, ncols=1)

ax[0].set_aspect('equal')
im0 = ax[0].pcolormesh(np.flip(mask[45, :, :].T, 0))
fig.colorbar(im0, ax=ax[0], orientation='horizontal')
ax[0].set_title("mask")

im1 = ax[1].contour(np.flip(potential[int(num_points_x / 2), :, :].T, 0))
fig.colorbar(im1, ax=ax[1], orientation='horizontal')
ax[1].set_title("Gor\'kov potential")

plt.tight_layout()
plt.show()

