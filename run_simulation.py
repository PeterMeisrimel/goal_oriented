# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 13:22:54 2017

@author: Peter Meisrimel, Lund University
"""

from __future__ import division
import numpy as np
import pylab as pl
import itertools

## Generic parent class for various run modes
class run_simulation():
    ## Basic initialization
    # @param self .
    # @param file_read parameters read from file
    # @return None
    def __init__(self, file_read):
        self.save_file_add = 'prob_' + str(file_read['problem_type']) + '_func_' + str(file_read['func_type'])
        if int(file_read['problem_type']) == 0:
            self.save_file_add = self.save_file_add + '_k_' + str(file_read['problem_stiff_para'])
        elif int(file_read['problem_type']) in [10, 11, 12]:
            self.save_file_add = self.save_file_add + '_dx_' + str(file_read['problem_meshsize'])
        elif int(file_read['problem_type']) in [20, 21]:
            self.save_file_add = self.save_file_add + '_u0_' + str(file_read['fsi_u0'])
        self.save_format = file_read['file_ending']
        self.reset_markers()
        pl.rcParams['lines.linewidth'] = 3
        pl.rcParams['lines.markersize'] = 17
        self.fontsize = 32
        self.para_legend = {'loc': 0, 'fontsize': self.fontsize - 4, 'fancybox': True, 'framealpha': 0.5}
        self.adjoint = False
        
    ## Basic file information to write into save file
    # @param self .
    # @param file_read parameters read from file
    # @return None
    def save_file_header(self, file_read):
        self.myfile.write('PROBLEM_TYPE   : {}'.format(file_read['problem_type']))
        self.myfile.write('\nSTIFFNESS_PARA : {}'.format(file_read['problem_stiff_para']))
        self.myfile.write('\nGRIDSIZE       : {}'.format(file_read['problem_meshsize']))
        self.myfile.write('\nT_END          : {}'.format(file_read['problem_t_end']))
        self.myfile.write('\nFUNC_TYPE      : {}'.format(file_read['func_type']))
        if file_read['problem_type'] in (20, 21):
            self.myfile.write('\nD1             : {}'.format(file_read['fsi_d1']))
            self.myfile.write('\nD2             : {}'.format(file_read['fsi_d2']))
            self.myfile.write('\nK1             : {}'.format(file_read['fsi_k1']))
            self.myfile.write('\nK2             : {}'.format(file_read['fsi_k2']))
            self.myfile.write('\nFIXPOINT TOL   : {}'.format(file_read['fsi_tol_fixpoint']))
            self.myfile.write('\nU 0            : {}'.format(file_read['fsi_u0']))
            
    ## Data to write into file, is done by passing a tuples
    # @param self .
    # @param args Tuples of data to write into file, format : (String, (List|Array))*
    # @return None
    def save_file_write(self, *args):
        for i in args:
            if type(i[1]) in (list, np.ndarray):
                self.myfile.writelines('\n' + i[0] + ' = ' + str(list(i[1])))
            else:
                self.myfile.writelines('\n' + i[0] + ' = %10.50f' % i[1])
    
    ## Function to end writing to file process and close the file
    # @param self .
    # @return None
    def save_file_end(self):
        self.myfile.close()
        
    ## Function to (re)set markers, colors and linestyles for plots to cycle
    # @param self .
    # @return None
    def reset_markers(self):
        self.markers = itertools.cycle('o^D*sP|')
        self.colors = itertools.cycle('gbrckmy')
        self.linestyles = itertools.cycle(['-', '--', ':', '-.'])
        
    def simulate(self, file_read):
        pass
    
    ## that one ugly functions in which all the values and functions are exchanged violating common access conventions of class paramters and functions
    # @param self .
    # @param file_read parameters read from file
    # @param tol tolerance
    # @param DG choose if to use DG scheme or not, default will take parameter read from file
    # @param DWR scheme to be used in DWR, 0 in non DWR case
    # @return Time integration class, u0, dt0 (Class, u0, zo for DWR)
    def init_parameters(self, file_read, tol, DG = None, DWR = 0):
        if DG is None:
            DG = file_read['dg']
            
        t_end = file_read['problem_t_end']
        func = file_read['func_type']
        quad = file_read['problem_extended']
        
        prob_type = file_read['problem_type']
        
        if   prob_type == -1 and DG == 0:
            from problem_verify import Problem_verification
            Prob = Problem_verification(t_end, func)
        if   prob_type == -1 and DG == 1:
            from problem_verify import Problem_verification_DG
            Prob = Problem_verification_DG(t_end, func)
        elif prob_type == -2 and DG == 0:
            from problem_verify import Problem_verification_time
            Prob = Problem_verification_time(t_end, func)
        elif prob_type == -2 and DG == 1:
            from problem_verify import Problem_verification_time_DG
            Prob = Problem_verification_time_DG(t_end, func)
        elif prob_type == 0 and DG == 0:
            from problem_basic import Problem_Basic
            Prob = Problem_Basic(t_end, func, file_read['problem_stiff_para'])
        elif prob_type == 0 and DG == 1:
            from problem_basic import Problem_Basic_DG
            Prob = Problem_Basic_DG(t_end, func, file_read['problem_stiff_para'], True if DWR else False)
        elif prob_type == 12:
            from problem_src_adv import Problem_heat_source_adv
            Prob = Problem_heat_source_adv(file_read['problem_meshsize'], t_end,
                                           file_read['direction'], True if DWR else False)
        elif prob_type == 20:
            raise ValueError('Currently not implemented')
        elif prob_type == 21:
            from problem_FSI import Problem_FSI_2D
            Prob = Problem_FSI_2D(file_read['problem_meshsize'], t_end = t_end, 
                                  D1 = file_read['fsi_d1'], D2 = file_read['fsi_d2'], 
                                  K1 = file_read['fsi_k1'], K2 = file_read['fsi_k2'],
                                  flux = file_read['fsi_flux'], u0 = file_read['fsi_u0'])
        else:
            raise RuntimeError('invalid problem type')
        self.Prob = Prob
        
        ## Initialize time integration scheme
        j = Prob.j
        t_end = Prob.t_end
        phat = 1 # order of lower order time integration scheme
        if DWR == 0: # NON DWR
            if DG == 0: # RK CASE
                if prob_type == 21: # FSI CASE
                    from schemes_FSI import SDIRK2_FSI_2D
                    fp_tol = file_read['fsi_tol_fixpoint']
                    if file_read['fsi_tol_fp_fac'] is not None:
                        fp_tol = tol*file_read['fsi_tol_fp_fac']
                    OSM = SDIRK2_FSI_2D(Prob.compute_matrices, t_end, fp_tol)
                        
                else: # Classic RK stuff
                    from schemes_RK import RK_get_class
                    OSM = RK_get_class(Prob.get_rhs(), file_read['time_int_type'])
                    phat = OSM.order_low
            else: # DG CASE
                pass # is part of Problem class
        else: # DWR CASE
            if file_read['adjoint_steps'] is not None:
                file_read['dwr_start_cells'] = file_read['adjoint_steps']
            from time_int_DWR import Time_int_DWR
        ## END DWR CASE
        
        ## Initialize timestep controller
        if DWR == 0:
            from dt_controllers import Timestep_controller
            DT_CONT_CLASS = Timestep_controller(tol = tol, order_low = phat, f_min = file_read['dt_cont_f_min'], 
                                                f_max = file_read['dt_cont_f_max'], f_safety = file_read['dt_cont_f_safety'])
            dt_cont  = DT_CONT_CLASS.get_controller(file_read['dt_cont_type'])
            dt0_cont = DT_CONT_CLASS.get_dt0_controller(file_read['dt_cont_dt0'])
            
        ## Initialize Time integration class
        if DWR == 0:
            if DG == 0: # RK CASE
                from time_int import Time_int
                Time_integ = Time_int(t_end, j, do_step = OSM.do_step, quad = quad,
                                      get_2_norm_sq = Prob.get_2_norm_sq,
                                      err_est_type = file_read['err_est_type'], get_dt = dt_cont)
            else: # DG CASE
                from time_int_DG import Time_int_DG
                Time_integ = Time_int_DG(t_end, j, do_step = Prob.do_step, get_2_norm_sq = Prob.get_2_norm_sq,
                                      err_est_type = file_read['err_est_type'], get_dt = dt_cont,
                                      do_step_err = Prob.do_step_err)
        else: # DWR CASE
            Time_integ = Time_int_DWR(t_end, file_read['dwr_start_cells'], file_read['dwr_refine_perc'],
                                      do_step = Prob.do_step, do_step_adj = Prob.do_step_adj, 
                                      get_dwr_est = Prob.get_dwr_est, j = j)
        # END DWR CASE
        
        # return parameters/classes accordingly
        if DWR != 0:
            return Time_integ, Prob.u0, Prob.z0
        else:
            if DG == 0:
                return Time_integ, Prob.u0, dt0_cont(tol)
            else:
                return Time_integ, Prob.uold, dt0_cont(tol)
    
    ## General function for priting plots
    # @param self pointer to self
    # @param file_read parameters read from file
    # @param dpi dpi value for image saving
    # @return None
    def print_plot(self, file_read, dpi = 1000):
        pass
