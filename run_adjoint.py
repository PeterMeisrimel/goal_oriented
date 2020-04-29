# -*- coding: utf-8 -*-
"""
Created on Mon Jul 7 11:11:20 2017
@author: Peter Meisrimel
"""
from __future__ import division
import pylab as pl
import numpy as np
from run_simulation import run_simulation
import dolfin as dol

## Basic running for visualization of adjoint solution
# \n run_mode = 101
class run_adjoint(run_simulation):
    ## Basic initialization
    # @param self .
    # @param file_read parameters read from file
    # @return None
    def __init__(self, file_read):
        run_simulation.__init__(self, file_read)
        
        if file_read['problem_type'] in [12]: ## 2 D problem using FEniCs
            self.pvd = dol.File(file_read['saving_path'] + self.save_file_add + '.pvd')
        else:
            self.fig, self.ax = pl.subplots(figsize = (12, 10))
            self.leg = []
            self.save = 'adjoint_sol_' + self.save_file_add + '_t_' + str(file_read['problem_t_end'])
            
            self.ax.set_title('Adjoint solution', fontsize = self.fontsize + 2)
            self.ax.set_xlabel(r'$t$', fontsize = self.fontsize)
            self.ax.set_ylabel(r'$z$', rotation = 0, fontsize = self.fontsize, labelpad = 20)
            self.ax.tick_params(labelsize=20)
        
    ## main function for executing a run
    # @param self .
    # @param file_read parameters read from file
    # @return None
    def simulate(self, file_read):
        Time_integ, u0, z0 = self.init_parameters(file_read, tol = 0, DG = 1, DWR = 1)
            
        times, adj_sol = Time_integ.run_adj(u0, z0)
        
        if file_read['problem_type'] in [12]: ## 2 D problem using FEniCs
            sol = dol.Function(adj_sol[times[0]].function_space())
            for t in times:
                sol.assign(adj_sol[t])
                self.pvd << sol
        else:
            dim = len(adj_sol[0].vector().get_local())
            sol = np.zeros((dim, len(times)))
            
            for idx, t in enumerate(times):
                sol[:, idx] = adj_sol[t].vector().get_local()
                
            for i in range(dim):
                self.ax.plot(times, sol[i, :], marker = None, c = next(self.colors), label = file_read['legend'])
                self.leg.append(r'$u_{}$'.format(i))
                
    ## General function for priting plots
    # @param self .
    # @param file_read parameters read from file
    # @param dpi dpi value for image saving
    # @return None
    def print_plot(self, file_read, dpi = 100):
        if file_read['problem_type'] not in [12]: ## 2 D problem using FEniCs
            self.ax.legend(loc = 0, fontsize = self.fontsize - 4)
            self.fig.savefig(file_read['saving_path'] + self.save + self.save_format, dpi = dpi)