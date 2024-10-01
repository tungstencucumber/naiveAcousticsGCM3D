import numpy as np
import os

class NaiveAcousticsSolver3D:

    def __init__(self, x_size, y_size, z_size, cp, rho, target_time, recording_time_step, wavelength,
                 source_width, source_length_in_time, source_center_in_percents_x=50, source_center_in_percents_y=50,
                 dump_vtk=False, dump_dir="data", verbose=False):

        assert (cp.shape == rho.shape)

        self.dump_vtk = dump_vtk
        self.dump_dir = dump_dir
        self.verbose = verbose

        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size
        self.num_points_x, self.num_points_y, self.num_points_z = cp.shape
        self.hx = self.x_size / (self.num_points_x - 1)
        self.hy = self.y_size / (self.num_points_y - 1)
        self.hz = self.z_size / (self.num_points_z - 1)

        assert (abs(self.hx - self.hy) < 0.01 * (self.hx + self.hy))
        assert (abs(self.hx - self.hz) < 0.01 * (self.hx + self.hz))

        self.cp = cp
        self.rho = rho
        self.K = np.square(self.cp) * self.rho

        self.T = 0

        max_cp = np.max(self.cp)
        numerical_method_recommended_tau = 0.3 * min(self.hx / max_cp, self.hy / max_cp, self.hz / max_cp)

        if self.verbose:
            print("Numerical time step recommendation:", numerical_method_recommended_tau)

        self.number_of_records = int(target_time / recording_time_step)
        self.steps_per_record = max(int(recording_time_step / numerical_method_recommended_tau), 1)
        self.tau = recording_time_step / self.steps_per_record
        self.wavelength = wavelength

        if self.verbose:
            print("Doing %d data records, %d steps per record, total %d steps, time step is %f, final time %f" %
                  (self.number_of_records, self.steps_per_record,
                   self.number_of_records * self.steps_per_record, self.tau, target_time))

        self.x = np.linspace(0.0, self.x_size, self.num_points_x, endpoint=True)
        self.y = np.linspace(0.0, self.y_size, self.num_points_y, endpoint=True)
        self.z = np.linspace(0.0, self.z_size, self.num_points_z, endpoint=True)

        if self.dump_vtk:
            from pyevtk.hl import gridToVTK
            gridToVTK(os.path.join(self.dump_dir, "params"), self.x, self.y, self.z,
                      pointData={"Cp": self.cp.T.ravel(), "rho": self.rho.T.ravel(), "K": self.K.T.ravel()})

        source_half_width_in_points = int(source_width / (2 * self.hx))
        source_center_x_idx = int(self.num_points_x * source_center_in_percents_x / 100)
        source_center_y_idx = int(self.num_points_y * source_center_in_percents_y / 100)
        self.source_x_start_point = source_center_x_idx - source_half_width_in_points
        self.source_x_end_point = source_center_x_idx + source_half_width_in_points
        self.source_y_start_point = source_center_y_idx - source_half_width_in_points
        self.source_y_end_point = source_center_y_idx + source_half_width_in_points

        # TODO: replace it with border function
        source_length_in_space = source_length_in_time * np.max(cp)
        source_length_in_points = int(source_length_in_space / self.hz)
        self.source_z_start_point = 1
        self.source_z_end_point = 1 + source_length_in_points

        if self.verbose:
            print("Grid shape", cp.shape)
            print("The source is located from %d to %d; from %d to %d; from %d to %d"
                  % (self.source_x_start_point, self.source_x_end_point,
                     self.source_y_start_point, self.source_y_end_point,
                     self.source_z_start_point, self.source_z_end_point))

    def forward(self):
        if self.dump_vtk:
            from pyevtk.hl import gridToVTK

        # We are going to use GCM for 3D acoustics.
        # The math, magic and matrices used below
        # are described in details in https://keldysh.ru/council/3/D00202403/kazakov_ao_diss.pdf

        Ax = np.zeros((self.num_points_x, self.num_points_y, self.num_points_z, 4, 4))
        Ax[:, :, :, 0, 3] = np.power(self.rho, -1)
        Ax[:, :, :, 3, 0] = self.K

        Ay = np.zeros((self.num_points_x, self.num_points_y, self.num_points_z, 4, 4))
        Ay[:, :, :, 1, 3] = np.power(self.rho, -1)
        Ay[:, :, :, 3, 1] = self.K

        Az = np.zeros((self.num_points_x, self.num_points_y, self.num_points_z, 4, 4))
        Az[:, :, :, 2, 3] = np.power(self.rho, -1)
        Az[:, :, :, 3, 2] = self.K

        Ux = np.zeros((self.num_points_x, self.num_points_y, self.num_points_z, 4, 4))
        Ux[:, :, :, 0, 0] = 1
        Ux[:, :, :, 0, 3] = np.power(self.cp * self.rho, -1)
        Ux[:, :, :, 1, 0] = 1
        Ux[:, :, :, 1, 3] = - np.power(self.cp * self.rho, -1)
        Ux[:, :, :, 2, 1] = 2
        Ux[:, :, :, 3, 2] = 2
        Ux = Ux / np.sqrt(2)

        Ux1 = np.zeros((self.num_points_x, self.num_points_y, self.num_points_z, 4, 4))
        Ux1[:, :, :, 0, 0] = 1
        Ux1[:, :, :, 0, 1] = 1
        Ux1[:, :, :, 1, 2] = 1
        Ux1[:, :, :, 2, 3] = 1
        Ux1[:, :, :, 3, 0] = self.cp * self.rho
        Ux1[:, :, :, 3, 1] = - self.cp * self.rho
        Ux1 = Ux1 / np.sqrt(2)

        Uy = np.zeros((self.num_points_x, self.num_points_y, self.num_points_z, 4, 4))
        Uy[:, :, :, 0, 1] = 1
        Uy[:, :, :, 0, 3] = np.power(self.cp * self.rho, -1)
        Uy[:, :, :, 1, 1] = 1
        Uy[:, :, :, 1, 3] = - np.power(self.cp * self.rho, -1)
        Uy[:, :, :, 2, 0] = 2
        Uy[:, :, :, 3, 2] = 2
        Uy = Uy / np.sqrt(2)

        Uy1 = np.zeros((self.num_points_x, self.num_points_y, self.num_points_z, 4, 4))
        Uy1[:, :, :, 0, 2] = 1
        Uy1[:, :, :, 1, 0] = 1
        Uy1[:, :, :, 1, 1] = 1
        Uy1[:, :, :, 2, 3] = 1
        Uy1[:, :, :, 3, 0] = self.cp * self.rho
        Uy1[:, :, :, 3, 1] = - self.cp * self.rho
        Uy1 = Uy1 / np.sqrt(2)

        Uz = np.zeros((self.num_points_x, self.num_points_y, self.num_points_z, 4, 4))
        Uz[:, :, :, 0, 2] = 1
        Uz[:, :, :, 0, 3] = np.power(self.cp * self.rho, -1)
        Uz[:, :, :, 1, 2] = 1
        Uz[:, :, :, 1, 3] = - np.power(self.cp * self.rho, -1)
        Uz[:, :, :, 2, 0] = 2
        Uz[:, :, :, 3, 1] = 2
        Uz = Uz / np.sqrt(2)

        Uz1 = np.zeros((self.num_points_x, self.num_points_y, self.num_points_z, 4, 4))
        Uz1[:, :, :, 0, 2] = 1
        Uz1[:, :, :, 1, 3] = 1
        Uz1[:, :, :, 2, 0] = 1
        Uz1[:, :, :, 2, 1] = 1
        Uz1[:, :, :, 3, 0] = self.cp * self.rho
        Uz1[:, :, :, 3, 1] = - self.cp * self.rho
        Uz1 = Uz1 / np.sqrt(2)

        # Initial state
        u_prev = np.zeros((self.num_points_x, self.num_points_y, self.num_points_z, 4))

        # TODO: replace it with border function
        u_prev[
            self.source_x_start_point:self.source_x_end_point,
            self.source_y_start_point:self.source_y_end_point,
            self.source_z_start_point:self.source_z_end_point,
            2
        ] = np.min(self.cp)
        u_border = np.mgrid[
                            -0.5*(self.source_x_end_point - self.source_x_start_point):0.5*(self.source_x_end_point - self.source_x_start_point),
                            -0.5*(self.source_y_end_point - self.source_y_start_point):0.5*(self.source_y_end_point - self.source_y_start_point),
                            -0.5*(self.source_z_end_point - self.source_z_start_point):0.5*(self.source_z_end_point - self.source_z_start_point)
                            ]
        mask = (u_border[0]**2 + u_border[1]**2 < 0.25*(self.source_x_end_point - self.source_x_start_point)**2) 
                # * (u_border[0]**2 + u_border[1]**2 >= 0.25)
        u_prev[
            self.source_x_start_point:self.source_x_end_point,
            self.source_y_start_point:self.source_y_end_point,
            self.source_z_start_point:self.source_z_end_point,
            3
        ][mask] = (np.min(self.cp) * np.min(self.cp) * np.min(self.rho) * mask)[mask]

        # The response recorded will be stored here
        buffer = np.zeros((self.num_points_x, self.num_points_y, self.number_of_records))

        pressure_buffer = np.zeros((self.num_points_x, self.num_points_y, self.num_points_z))
        pressure_squared_buffer = np.zeros((self.num_points_x, self.num_points_y, self.num_points_z))
        velocity_buffer = np.zeros((self.num_points_x, self.num_points_y, self.num_points_z, 3))
        velocity_norm_buffer = np.zeros((self.num_points_x, self.num_points_y, self.num_points_z))
        gorkov_potential = np.zeros((self.num_points_x, self.num_points_y, self.num_points_z))

        if self.dump_vtk:
            gridToVTK(os.path.join(self.dump_dir, "u" + str(0)), self.x, self.y, self.z,
                      pointData={"vx": u_prev[:, :, :, 0].T.ravel(),
                                 "vy": u_prev[:, :, :, 1].T.ravel(),
                                 "vz": u_prev[:, :, :, 2].T.ravel(),
                                 "p": u_prev[:, :, :, 3].T.ravel(),
                                 "gor'kov": gorkov_potential.T.ravel()})

        # Spacial steps for characterictics (they are constant, since the model is linear).
        steps = self.tau * self.cp

        # We are going to use these formulas - https://math.stackexchange.com/a/889571 - for interpolation

        # positive direction
        c1p = steps * (- self.hx + steps) / (2 * self.hx * self.hx)
        c2p = (self.hx + steps) * (- self.hx + steps) / (- self.hx * self.hx)
        c3p = (self.hx + steps) * steps / (2 * self.hx * self.hx)
        c_pos = (c1p, c2p, c3p)

        # negative direction
        c1m = steps * (self.hx + steps) / (2 * self.hx * self.hx)
        c2m = (- self.hx + steps) * (self.hx + steps) / (- self.hx * self.hx)
        c3m = (- self.hx + steps) * steps / (2 * self.hx * self.hx)
        c_neg = (c1m, c2m, c3m)

        # Time stepping using space-split scheme.
        for i in range(self.number_of_records):
            if self.verbose:
                print("Stepping for the time record", i, end=': ')

            for j in range(self.steps_per_record):
                if self.verbose:
                    print("Step", j, end=' ')
                    if j == self.steps_per_record - 1:
                        print("\n", end='')

                # Three steps of 3D space-split scheme
                u_next, inv_next = self.do_split_step(u_prev, Uz, Uz1, c_neg, c_pos, i, direction=2)
                u_next, inv_next = self.do_split_step(u_next, Uy, Uy1, c_neg, c_pos, i, direction=1)
                u_next, inv_next = self.do_split_step(u_next, Ux, Ux1, c_neg, c_pos, i, direction=0)

                # TODO: emitting and reflecting top border
                form = np.min(self.cp) * np.min(self.cp) * np.min(self.rho) * mask[:, :, 0] * np.cos(2 * np.pi * self.T / self.wavelength * np.min(self.cp))
                p0 = u_next[:, :, 0, 3]
                p0[self.source_x_start_point:self.source_x_end_point, self.source_y_start_point:self.source_y_end_point][mask[:, :, 0]] = form[mask[:, :, 0]]
                u_next[:, :, 0, 3] = p0
                u_next[:, :, 0, 2] = inv_next[:, :, 0, 1] + p0 / (self.cp[:, :, 0] * self.rho[:, :, 0])

                buffer[:, :, -i-1] = np.copy(u_next[:, :, 1, 3])

                # averaging data across all snapshots for Gor'kov potential estimation
                pressure_buffer += np.copy(u_next[:, :, :, 3])
                pressure_squared_buffer += u_next[:, :, :, 3]**2
                velocity_buffer += np.copy(u_next[:, :, :, :3])
                velocity_norm_buffer += np.linalg.norm(u_next[:, :, :, :3], axis=3)**2

                # Gor'kov potential calculation (in a dimensionless form)
                p_dimensionless = pressure_buffer / (self.rho[:,:,:] * self.cp[:,:,:] * np.min(self.cp) * (i * self.steps_per_record + j + 1))
                p_sq_dimensionless = pressure_squared_buffer / (self.rho[:,:,:] * self.cp[:,:,:] * np.min(self.cp) )**2 / (i * self.steps_per_record + j + 1)
                v_dimensionless = velocity_buffer / (np.min(self.cp) * (i * self.steps_per_record + j + 1))
                v_norm_dimensionless = velocity_norm_buffer / (np.min(self.cp) )**2 / (i * self.steps_per_record + j + 1)

                p_fluc_squared = p_sq_dimensionless - p_dimensionless**2
                v_fluc_squared = v_norm_dimensionless - np.linalg.norm(v_dimensionless, axis=3)**2
                p_fluc_squared[p_fluc_squared < 0.] = 0.
                v_fluc_squared[v_fluc_squared < 0.] = 0.
                gorkov_potential = np.sqrt(p_fluc_squared) / 3.0 - np.sqrt(v_fluc_squared) / 2.0

                if self.dump_vtk:
                    gridToVTK(os.path.join(self.dump_dir, "u" + str(i + 1)), self.x, self.y, self.z,
                              pointData={"vx": u_next[:, :, :, 0].T.ravel(),
                                         "vy": u_next[:, :, :, 1].T.ravel(),
                                         "vz": u_next[:, :, :, 2].T.ravel(),
                                         "p": u_next[:, :, :, 3].T.ravel(),
                                         "gor'kov": gorkov_potential.T.ravel()})

                u_prev = np.copy(u_next)

                self.T += self.tau

        return buffer, gorkov_potential
    
    def do_split_step(self, u_prev, U, U1, c_neg, c_pos, i, direction=-1):
        return _do_split_step((self.num_points_x, self.num_points_y, self.num_points_z), u_prev, U, U1, c_neg, c_pos, i, direction)

def _do_split_step(shape, u_prev, U, U1, c_neg, c_pos, i, direction=-1):
    c1m, c2m, c3m = c_neg
    c1p, c2p, c3p = c_pos

    if direction == 0:
        u_lefts = np.pad(u_prev, ((1, 0), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)[:-1, :, :, :]
        # u_lefts = np.concatenate((np.zeros((1, shape[1], shape[2], 4), dtype=np.float64), u_prev), axis=0)[:-1, :, :, :]
        u_rights = np.pad(u_prev, ((0, 1), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)[1:, :, :, :]
        # u_rights = np.concatenate((u_prev, np.zeros((1, shape[1], shape[2], 4), dtype=np.float64)), axis=0)[1:, :, :, :]
    elif direction == 1:
        u_lefts = np.pad(u_prev, ((0, 0), (1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)[:, :-1, :, :]
        # u_lefts = np.concatenate((np.zeros((shape[0], 1, shape[2], 4), dtype=np.float64), u_prev), axis=1)[:, :-1, :, :]
        u_rights = np.pad(u_prev, ((0, 0), (0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)[:, 1:, :, :]
        # u_rights = np.concatenate((u_prev, np.zeros((shape[0], 1, shape[2], 4), dtype=np.float64)), axis=1)[:, 1:, :, :]
    elif direction == 2:
        # absorptive border conditions
        u_lefts = np.pad(u_prev, ((0, 0), (0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)[:, :, :-1, :]
        # u_lefts = np.concatenate((np.zeros((shape[0], shape[1], 1, 4), dtype=np.float64), u_prev), axis=2)[:, :, :-1, :]
        # reflective border conditions
        # u_lefts = np.pad(u_prev, ((0, 0), (0, 0), (1, 0), (0, 0)), mode='reflect')[:, :, :-1, :]
        # u_lefts[:,:,0,3] = np.copy(-u_lefts[:,:,0,3])

        # absorptive border conditions
        u_rights = np.pad(u_prev, ((0, 0), (0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)[:, :, 1:, :]
        # u_rights = np.concatenate((u_prev, np.zeros((shape[0], shape[1], 1, 4), dtype=np.float64)), axis=2)[:, :, 1:, :]
        # reflective border conditions
        # u_rights = np.pad(u_prev, ((0, 0), (0, 0), (0, 1), (0, 0)), mode='reflect')[:, :, 1:, :]
        # u_rights[:,:,-1,3] = np.copy(-u_rights[:,:,-1,3])
    else:
        raise RuntimeError("Incorrect axis for 3D method")

    # rieman_invs_here = np.empty((*shape, 4))
    # rieman_invs_left = np.empty((*shape, 4))
    # rieman_invs_right = np.empty((*shape, 4))
    # for g in range(shape[0]):
    #     for q in range(shape[1]):
    #         for i in range(shape[2]):
    #             for j in range(4):
    #                 rieman_invs_here[g,q,i,j] = np.sum(U[g,q,i,j,:] * u_prev[g,q,i,:])
    #                 rieman_invs_left[g,q,i,j] = np.sum(U[g,q,i,j,:] * u_lefts[g,q,i,:])
    #                 rieman_invs_right[g,q,i,j] = np.sum(U[g,q,i,j,:] * u_rights[g,q,i,:])

    rieman_invs_here = np.einsum('gqijk,gqik->gqij', U, u_prev)
    rieman_invs_left = np.einsum('gqijk,gqik->gqij', U, u_lefts)
    rieman_invs_right = np.einsum('gqijk,gqik->gqij', U, u_rights)

    # limiter_min = np.empty((*shape, 4))
    # limiter_max = np.empty((*shape, 4))
    # for g in range(shape[0]):
    #     for q in range(shape[1]):
    #         for i in range(shape[2]):
    #             for j in range(4):
    #                 limiter_min[g,q,i,j] = min([rieman_invs_here[g,q,i,j], rieman_invs_left[g,q,i,j], rieman_invs_right[g,q,i,j]])
    #                 limiter_max[g,q,i,j] = max([rieman_invs_here[g,q,i,j], rieman_invs_left[g,q,i,j], rieman_invs_right[g,q,i,j]])
    
    limiter_min = np.min([rieman_invs_here, rieman_invs_left, rieman_invs_right], axis=0)
    limiter_max = np.max([rieman_invs_here, rieman_invs_left, rieman_invs_right], axis=0)

    rieman_invs_pos = c1m * rieman_invs_left[:, :, :, 0] + c2m * rieman_invs_here[:, :, :, 0] + c3m * rieman_invs_right[:, :, :, 0]
    rieman_invs_neg = c1p * rieman_invs_left[:, :, :, 1] + c2p * rieman_invs_here[:, :, :, 1] + c3p * rieman_invs_right[:, :, :, 1]
    rieman_invs_zero_1 = rieman_invs_here[:, :, :, 2]
    rieman_invs_zero_2 = rieman_invs_here[:, :, :, 3]

    riemans_next = np.zeros((shape[0], shape[1], shape[2], 4))
    riemans_next[:, :, :, 0] = rieman_invs_pos
    riemans_next[:, :, :, 1] = rieman_invs_neg
    riemans_next[:, :, :, 2] = rieman_invs_zero_1
    riemans_next[:, :, :, 3] = rieman_invs_zero_2

    # riemans_next = np.empty((*shape, 4))
    # for g in range(shape[0]):
    #     for q in range(shape[1]):
    #         for i in range(shape[2]):
    #             for j in range(4):
    #                 riemans_next[g,q,i,j] = max([riemans_next[g,q,i,j], limiter_min[g,q,i,j]])
    #                 riemans_next[g,q,i,j] = min([riemans_next[g,q,i,j], limiter_max[g,q,i,j]])
    riemans_next = np.max([riemans_next, limiter_min], axis=0)
    riemans_next = np.min([riemans_next, limiter_max], axis=0)

    # u_next = np.empty((*shape, 4))
    # for g in range(shape[0]):
    #     for q in range(shape[1]):
    #         for i in range(shape[2]):
    #             for j in range(4):
    #                 u_next[g,q,i,j] = np.sum(U1[g,q,i,j,:] * riemans_next[g,q,i,:])
    u_next = np.einsum('gqijk,gqik->gqij', U1, riemans_next)
    return u_next, riemans_next
