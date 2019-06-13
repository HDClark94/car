import numpy as np

class gridcell(object):
    '''
    grid cell class
    '''

    def __init__(self, lowest_grid_scale=0.1, grid_scale_jump=1.5, cells_per_scale=1, n_scales=3):

        self.lowest_grid_scale = lowest_grid_scale
        self.grid_scale_jump = grid_scale_jump
        self.n_scales = n_scales
        self.grid_scales = [lowest_grid_scale]
        for i in range(n_scales):
            self.grid_scales.append(self.grid_scales[-1] * self.grid_scale_jump)
        self.grid_scales = np.array(self.grid_scales)

        self.cells_per_scale = cells_per_scale # not used atm
        self.cell_offsets = 0 # not used atm

        tmp = []
        for i in range(self.cells_per_scale):
            tmp.append(self.grid_scales)

        self.grid_scales = np.array(tmp).flatten()

    def sine_wave_grid_firing(self, position):
        # for firing rate, we input position and a scale
        # y = sin(xk) x = position, k = 2pi/scale

        cell_firing_rates = np.sin(position*(np.pi*2)/self.grid_scales) # this should be a np.array() of size n_cells*n_scales (flattened)

        # clip negative firing rates
        cell_firing_rates[cell_firing_rates<0] = 0

        return cell_firing_rates