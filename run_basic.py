# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:03:21 2016
@author: Peter Meisrimel
"""
from __future__ import division
import pylab as pl
from run_simulation import run_simulation

## Basic running case providing results of tol vs functional error for varying input files
# \n run_mode = 0
class run_basic(run_simulation):
    ## Basic initialization
    # @param self .
    # @param file_read parameters read from file
    # @return None
    def __init__(self, file_read):
        run_simulation.__init__(self, file_read)
        
        self.fig, self.ax = pl.subplots(figsize = (12, 10))
        self.save = 'basic_tol_vs_error_' + 't_' + str(file_read['problem_t_end']).replace('.', ',') + '_' + self.save_file_add
        
        self.ax.set_title('Tolerance vs. Error', fontsize = self.fontsize + 2)
        self.ax.set_xlabel(r'Tolerance $\tau$', fontsize = self.fontsize)
        self.ax.set_ylabel(r'$e^J$', rotation = 0, fontsize = self.fontsize+10, labelpad = 20)
        self.ax.set_xscale('log'); self.ax.set_yscale('log')
        self.ax.grid(b = True, which = 'major')
        self.ax.tick_params(labelsize=20)
        
        self.tol_list = [10**-i for i in range(file_read['tol_step_start'], file_read['tol_step_start'] + file_read['tol_steps'])]
        self.ax.invert_xaxis()
        self.ax.plot(self.tol_list, self.tol_list, '--k', lw = 2, label = r'$\tau$')
        
    ## main function for executing a run
    # @param self .
    # @param file_read parameters read from file
    # @return None
    def simulate(self, file_read):
        # error & estimate lists
        err_list = []

        if file_read['save_results']:
            import time
            file_t = time.strftime("%Y-%m-%d-%H-%M-%S")    
            self.myfile = open(file_read['saving_path'] + 'basic_' + file_read['filename_self'] + '_{}.txt'.format(file_t), 'w')            
            self.save_file_header(file_read)
            self.save_file_write(('tol_list', self.tol_list))
            
        for tol in self.tol_list:
            print('...running tol = %.2e' % (tol))
            Time_integ, u0, dt0 = self.init_parameters(file_read, tol = tol)
            
            err, _, _, _ = Time_integ.run(dt0, u0, j_ref = self.Prob.solution_int())
            err_list.append(err)
            #if err < 1e-16:
            #    self.ax.set_ylim(bottom = 1e-16)
                
        self.ax.plot(self.tol_list, err_list, marker = next(self.markers), c = next(self.colors),
                     label = file_read['legend'])
        print('Order estimation through the slopes : ')
        
        for i in range(len(err_list)-1):
            slope = pl.log(err_list[i]/err_list[i+1])/pl.log(10)
            print(slope)
            
        if file_read['save_results']:
            self.save_file_write(('err_list', err_list))
            self.save_file_end()
    
    ## General function for priting plots
    # @param self .
    # @param file_read parameters read from file
    # @param dpi dpi value for image saving
    # @return None
    def print_plot(self, file_read, dpi = 100):
        self.ax.legend(**self.para_legend)
        self.fig.savefig(file_read['saving_path'] + self.save + self.save_format, dpi = dpi)