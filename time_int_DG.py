# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 17:08:36 2016
@author: Peter Meisrimel
"""
from __future__ import division
from time_int import Time_int

## time-integration class
class Time_int_DG(Time_int):
    ## Initialize function
    # @param self .
    # @param t_end t_e
    # @param j density funciton j
    # @param do_step function to perform a single forward step
    # @param get_2_norm_sq function to get squared 2-norm of a given state vector
    # @param err_est_type which type of error estimate to use
    # @param get_dt timestep controller
    # @param do_step_err function to perform a single forward step with error estimation
    # @return None
    def __init__(self, t_end, j = None, do_step = None, get_2_norm_sq = None,
                 err_est_type = 0, get_dt = None, do_step_err = None):
        ## end time
        self.t_end = t_end
        ## timestep evaluation function
        self.j = j
        self.do_step = do_step
        self.do_step_err = do_step_err
        self.err_est_type = err_est_type
        self.get_2_norm_sq = get_2_norm_sq
        self.get_dt = get_dt
        ## time discretization
        self.times = []
        
    ## Solve problem forward in time, given fixed timesteps
    # @param self .
    # @param dt fixed stepsize dt
    # @param uold initial u0
    # @return j_sol  number of steps
    def forward(self, dt, uold):
        j_val, t = 0., 0.
        steps = int(self.t_end/dt)
        self.j_new = self.j(t, uold)
            
        for i in range(steps):
            unew = self.do_step(t, dt, uold)
            j_up, _ = self.quad_trapezoidal(t, dt, unew)
            j_val += j_up
            uold = unew
            t += dt
        return j_val, steps
    
    ## Solve problem forward in time with variable step sizes
    # @param self .
    # @param dt initial dt0
    # @param uold initial u0
    # @return j_sol, number of steps
    def forward_le(self, dt, uold):
        j_val, steps, t = 0., 0, 0.
        self.times.append(0)
        self.j_new = self.j(t, uold)
        
        while t < self.t_end - 1e-15:
            if dt < 1.e-15:
                raise ValueError('timestep critically low with dt = {} at t = {}'.format(dt, t))
            dt = min(dt, self.t_end - t)
            
            unew, l_err = self.do_step_err(t, dt, uold)
            j_up, j_quad_est = self.quad_trapezoidal(t, dt, unew)
            
            t += dt
            j_val += j_up
            uold = unew
            steps += 1
            self.times.append(t)
            err_est = self.err_est(t, unew, l_err, j_quad_est)
            dt = self.get_dt(err_est, dt)
        return j_val, steps