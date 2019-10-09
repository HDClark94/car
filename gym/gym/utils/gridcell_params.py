class gridcell_params(object):
    '''
    parameter class
    '''

    def __init__(self, lowest_grid_scale=0.1, grid_scale_jump=1.5, cells_per_scale=1, n_scales=3):

        self.lowest_grid_scale = lowest_grid_scale
        self.grid_scale_jump = grid_scale_jump
        self.n_scales = n_scales
        self.grid_scales = [lowest_grid_scale]
        self.cells_per_scale = self.cells_per_scale  # not used atm
        self.cell_offsets = 0  # not used atm