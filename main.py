import matplotlib.pyplot as plt
from pyevtk.hl import gridToVTK
import numpy as np
import os

from acoustics_solver_3d import NaiveAcousticsSolver3D

# geometry sizes
num_points_x = 100
num_points_y = 100
num_points_z = 30
space_step = 10
x_size = space_step * (num_points_x - 1)
y_size = space_step * (num_points_y - 1)
z_size = space_step * (num_points_z - 1)

# rheology limits
rho_min = 2000
rho_max = 3000
cp_min = 2000
cp_max = 4000

# Option 1: create manually a mask for simple bright inclusion
mask = np.zeros(shape=(num_points_x, num_points_y, num_points_z), dtype=float)
mask[40:50, 40:50, 10:20] = 1

# # Option 2: get (0; 1] mask representing layered structure
# # WARN: take a look first at https://github.com/avasyukov/quasi_marmousi and review the parameters
# mask = get_mask(num_points_x, num_points_y, 5, 8, 3, 10)

# acoustics parameters of media from the mask created above
cp = cp_min + (cp_max - cp_min) * mask
rho = rho_min + (rho_max - rho_min) * mask

# signal recording parameters
total_signal_recording_time = 0.225
time_step_between_records = 0.0015

# excitation pulse space size
source_width = 50 * space_step
source_length = 4 * time_step_between_records

dump_dir = "."

solver = NaiveAcousticsSolver3D(x_size, y_size, z_size, cp, rho,
                                total_signal_recording_time, time_step_between_records,
                                source_width, source_length,
                                source_center_in_percents_x=50, source_center_in_percents_y=50,
                                verbose=True, dump_vtk=True, dump_dir=dump_dir)
signal = solver.forward()
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
im0 = ax[0].pcolormesh(mask[45, :, :].T)
fig.colorbar(im0, ax=ax[0], orientation='horizontal')
ax[0].set_title("mask")

im1 = ax[1].pcolormesh(signal[45, :, :].T)
fig.colorbar(im1, ax=ax[1], orientation='horizontal')
ax[1].set_title("signal")

plt.tight_layout()
plt.show()

