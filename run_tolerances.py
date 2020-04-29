# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 16:09:34 2015
@author: Peter Meisrimel
"""
from __future__ import division
from run_simulation import run_simulation
import numpy as np
import pylab as pl

## Comparison of various input files
# \n run_mode = 2
class run_tolerances(run_simulation):
    ## Basic initialization
    # @param self .
    # @param file_read parameters read from file
    # @param tols_min minimum tolerance used in run
    # @param tols_max maximum tolerance used in run
    # @return None
    def __init__(self, file_read, tols_min, tols_max):
        run_simulation.__init__(self, file_read)
        tol_list = [10**-i for i in range(tols_min, tols_max)]
        
        self.fig_tol_err_int, self.ax_tol_err_int = pl.subplots(figsize = (12, 10))
        self.fig_tol_err_int_save = 'tol_vs_error_' + self.save_file_add + '_t_' + str(file_read['problem_t_end']).replace('.', ',')
        self.ax_tol_err_int.set_title(r'Tolerance vs. Error', fontsize = self.fontsize + 2)
        self.ax_tol_err_int.set_xlabel(r'Tolerance $\tau$', fontsize = self.fontsize)
        self.ax_tol_err_int.set_ylabel(r'$e^\mathcal{J}$', rotation = 0, labelpad = 20, fontsize = self.fontsize + 10)
        self.ax_tol_err_int.set_xscale('log'); self.ax_tol_err_int.set_yscale('log')
        self.ax_tol_err_int.tick_params(labelsize=20)
        self.ax_tol_err_int.grid(b = True, which = 'major')
        self.ax_tol_err_int.invert_xaxis()
        self.ax_tol_err_int.plot(tol_list, tol_list, '--k', lw = 2, label = r'$\tau$')
        
        self.fig_err_work, self.ax_err_work = pl.subplots(figsize = (12, 10))
        self.fig_err_work_save = 'comp_eff_' + self.save_file_add + '_t_' + str(file_read['problem_t_end']).replace('.', ',')
        self.ax_err_work.set_title(r'Computational efficiency', fontsize = self.fontsize + 2)
        self.ax_err_work.set_xlabel(r'Comp. time', fontsize = self.fontsize)
        self.ax_err_work.set_ylabel(r'$e^\mathcal{J}$', rotation = 0, fontsize = self.fontsize + 10, labelpad = 20)
        self.ax_err_work.set_xscale('log'); self.ax_err_work.set_yscale('log')
        self.ax_err_work.grid(b = True, which = 'major')
        self.ax_err_work.tick_params(labelsize=20)
        
        self.fig_quality, self.ax_quality = pl.subplots(figsize = (12, 10))
        self.fig_quality_save = 'grid_quality_' + self.save_file_add + '_t_' + str(file_read['problem_t_end']).replace('.', ',')
        self.ax_quality.set_title(r'Grid Quality', fontsize = self.fontsize + 2)
        self.ax_quality.set_xlabel('# grid points', fontsize = self.fontsize)
        self.ax_quality.set_ylabel(r'$e^\mathcal{J}$', rotation = 0, fontsize = self.fontsize + 10, labelpad = 20)
        self.ax_quality.set_xscale('log'); self.ax_quality.set_yscale('log')
        self.ax_quality.grid(b = True, which = 'major')
        self.ax_quality.tick_params(labelsize=20)
        
        if file_read['save_results']:
            import time
            file_t = time.strftime("%Y-%m-%d-%H-%M-%S")    
            self.myfile = open(file_read['saving_path'] + 'tol_' + file_read['filename_self'] + '_{}.txt'.format(file_t), 'w')            
            self.save_file_header(file_read)
        
        self.tol_fig_ax = {}
        
        for tol in tol_list:
            fig, ax = pl.subplots(figsize = (12, 10))
            fig_save = 'dts_' + self.save_file_add + '_t_' + str(file_read['problem_t_end']) + '_tol_' + '%0.0e' % (tol)
            ax.set_title(r'Timestepsizes, $\tau = 10^{%2i}$' % (np.log10(tol)), fontsize = self.fontsize + 2)
            ax.set_xlabel(r't', fontsize = self.fontsize)
            ax.set_ylabel(r'$\Delta t$', labelpad = 20, fontsize = self.fontsize, rotation = 0)
            ax.set_yscale('log')
            ax.grid(b = True, which = 'major')
            ax.tick_params(labelsize=20)
            self.tol_fig_ax[tol] = (fig, ax, fig_save)
        
    ## main function for executing a run
    # @param self .
    # @param file_read parameters read from file
    # @return None
    def simulate(self, file_read):
        tol_list = [10**-i for i in range(file_read['tol_step_start'], file_read['tol_step_start'] + file_read['tol_steps'])]
        
        # error & estimate lists
        self.DWR = file_read['dwr_scheme'] # > 0 for DWR case
        errs, steps, times, errs_real = [],[],[],[]
        if self.DWR: # DWR case
            times.append(0) # as times are adding up previous comp. times
            mesh, est_list = None, None
            err, j_prev = np.inf, 0
        
        startup_done = False
        m = next(self.markers); c = next(self.colors); ls = next(self.linestyles)
        for tol in tol_list[:1] + tol_list: # one extra for startup
            fig, ax, fig_save = self.tol_fig_ax[tol]
            if not startup_done:
                print('...performing startup with tol = %.2e' % (tol))
            else:
                print('...running tol = %.2e' % (tol))
            
            # initialize and run
            if self.DWR:
                Time_integ, u0, z0 = self.init_parameters(file_read, tol = tol, DG = 1, DWR = file_read['dwr_scheme'])
                j_ref = self.Prob.solution_int()
                err, time_run, err_real, est_list, j_prev, mesh = Time_integ.run(tol, u0, z0, mesh = mesh, est = err,
                                                                                 est_list = est_list, j_prev = j_prev, j_ref = j_ref)
                time_run = time_run + times[-1]
                num_steps = mesh.num_cells()
            else:
                Time_integ, u0, dt0 = self.init_parameters(file_read, tol = tol)
                j_ref = self.Prob.solution_int()
                err, time_run, num_steps, j_val = Time_integ.run(dt0, u0, j_ref = j_ref)
                times_res = Time_integ.times
            
            # saving of results
            if startup_done: # baisc case, store results
                errs.append(err)
                times.append(time_run)
                steps.append(num_steps)
                if self.DWR:
                    errs_real.append(err_real)
                else:
                    ax.plot(times_res[:-1], np.diff(np.array(times_res)),
                            marker = None if len(times_res) > 80 else m, c = c, linestyle = ls, label = file_read['legend'])
            else: # startup, no storing, reset initial values
                startup_done = True
                if self.DWR:
                    self.mesh, est_list = None, None
                    err, j_prev = np.inf, 0
        
        if self.DWR: times = times[1:] # remove zero again
        
        label = file_read['legend']
        if self.DWR:
            m2, c2 = next(self.markers), next(self.colors)
            self.ax_tol_err_int.plot(tol_list, errs, marker = m, c = c, label = label + ' Est')
            self.ax_err_work   .plot(times   , errs, marker = m, c = c, label = label + ' Est')
            self.ax_quality    .plot(steps   , errs, marker = m, c = c, label = label + ' Est')
            
            self.ax_tol_err_int.plot(tol_list, errs_real, marker = m2, c = c2, label = label + ' Err')
            self.ax_err_work   .plot(times   , errs_real, marker = m2, c = c2, label = label + ' Err')
            self.ax_quality    .plot(steps   , errs_real, marker = m2, c = c2, label = label + ' Err')
        else:
            self.ax_tol_err_int.plot(tol_list, errs, marker = m, c = c, label = label)
            self.ax_err_work   .plot(times   , errs, marker = m, c = c, label = label)
            self.ax_quality    .plot(steps   , errs, marker = m, c = c, label = label)
            
        if file_read['save_results']:
            self.myfile.write('\n' + file_read['filename_self'] + '  ' + file_read['legend'])
            self.save_file_write(('tols', tol_list), ('errs', errs), ('times', times), ('steps', steps))
            if self.DWR:
                self.save_file_write(('errs_real', errs_real))

    ## General function for priting plots
    # @param self .
    # @param file_read parameters read from file
    # @param dpi dpi value for image saving
    # @return None
    def print_plot(self, file_read, dpi = 100):
        if file_read['save_results']: 
            self.save_file_end()
        path = file_read['saving_path']

        self.ax_tol_err_int.legend(**self.para_legend)
        self.fig_tol_err_int.savefig(path + self.fig_tol_err_int_save + self.save_format, dpi = dpi)
        
        self.ax_err_work.legend(**self.para_legend)
        self.fig_err_work.savefig(path + self.fig_err_work_save + self.save_format, dpi = dpi)
        
        self.ax_quality.legend(**self.para_legend)
        self.fig_quality.savefig(path + self.fig_quality_save + self.save_format, dpi = dpi)
        
        for tol in self.tol_fig_ax.keys():
            fig, ax, fig_save = self.tol_fig_ax[tol]
            ax.legend(**self.para_legend)
            fig.savefig(path + fig_save + self.save_format, dpi = dpi)