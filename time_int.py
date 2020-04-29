# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 17:08:36 2016
@author: Peter Meisrimel
"""
from __future__ import division
import numpy as np

## time-integration class
class Time_int():
    ## Initialize function
    # @param self .
    # @param t_end t_e
    # @param j density function
    # @param do_step function for doing a single forward step
    # @param quad which style of quadrature to use, 0 for trapezoidal, 1 for time-integration
    # @param get_2_norm_sq function to get squared 2-norm of a given state vector
    # @param err_est_type which error estimate to use
    # @param get_dt timestep controller
    # @return None
    def __init__(self, t_end, j = None, do_step = None, quad = 0, get_2_norm_sq = None, err_est_type = 0, get_dt = None):
        ## end time
        self.t_end = t_end
        ## timestep evaluation function
        self.j = j
        self.do_step = do_step
        self.quad = quad
        self.err_est_type = err_est_type
        self.get_2_norm_sq = get_2_norm_sq
        self.get_dt = get_dt
        
        ## time discretization
        self.times = []
        
    ## Function to get error estimate
    # @param self .
    # @param t time
    # @param unew state vector after timestep
    # @param l_err local error estimate
    # @param quad_est quadrature error estimate
    # @return error estimate
    def err_est(self, t, unew, l_err, quad_est):
        if   self.err_est_type == -1:
            return np.sqrt(self.get_2_norm_sq(l_err) + quad_est**2)
        elif self.err_est_type ==  0:
            return np.sqrt(self.get_2_norm_sq(l_err))
        elif self.err_est_type ==  1:
            return abs(self.j(t, unew) - self.j(t, unew - l_err))
        elif self.err_est_type ==  2:
            return abs(quad_est)
        elif self.err_est_type ==  3:
            return abs(self.j(t, unew) - self.j(t, unew - l_err)) + abs(quad_est)
        else:
            raise ValueError('invalid err_est_type')
        
    ## Trapezoidal rule quadrature
    # @param self .
    # @param t time
    # @param dt stepsize
    # @param u unew
    # @return quadrature increment, quadrature error estimate
    def quad_trapezoidal(self, t, dt, u):
        self.j_old = self.j_new
        self.j_new = self.j(t + dt, u)
        
        j_up = 0.5*dt*(self.j_old + self.j_new)
        j_est = 0.5*dt*abs(self.j_old - self.j_new)
        return j_up, j_est
    
    ## Solve problem forward in time, given fixed timesteps
    # @param self .
    # @param dt fixed stepsize dt
    # @param uold initial u0
    # @return j_sol, number of steps
    def forward(self, dt, uold):
        j_val, t = 0., 0.
        steps = int(self.t_end/dt)
        if not self.quad:
            self.j_new = self.j(t, uold)
        for i in range(steps):
            if self.quad:
                unew, _, j_up, _ = self.do_step(t, dt, uold, j = self.j)
            else:
                unew, _ = self.do_step(t, dt, uold)
                j_up, _ = self.quad_trapezoidal(t, dt, unew)
            j_val += j_up
            uold = unew
            t += dt
        return j_val, steps
        
    ## Solve problem forward in time with variable step sizes
    # @param self .
    # @param dt dt0
    # @param uold initial u0
    # @return j_sol, number of steps
    def forward_le(self, dt, uold):
        j_val, steps, t = 0., 0, 0.
        self.times.append(0)
        
        if not self.quad:
            self.j_new = self.j(t, uold)
        
        while t < self.t_end - 1e-15:
            if dt < 1.e-15:
                raise ValueError('timestep critically low with dt = {} at t = {}'.format(dt, t))
            dt = min(dt, self.t_end - t)
            if self.quad:
                unew, l_err, j_up, j_quad_est = self.do_step(t, dt, uold, j = self.j)
            else:
                unew, l_err = self.do_step(t, dt, uold)
                j_up, j_quad_est = self.quad_trapezoidal(t, dt, unew)
            t += dt
            j_val += j_up
            uold = unew
            steps += 1
            self.times.append(t)
            err_est = self.err_est(t, unew, l_err, j_quad_est)
            dt = self.get_dt(err_est, dt)
        return j_val, steps
    
    ## Generic function to perform a full forward run
    # @param self .
    # @param dt0 initial stepsize dt0
    # @param u0 initial u0
    # @param adaptive set to 0 for constant timesteps, 1 for adaptivity
    # @param j_ref reference J(u)
    # @return Error, runtime, number of steps, j_sol
    def run(self, dt0, u0, adaptive = 1, j_ref = 0):
        import time
        start_time = time.time()
        
        # do normal forward run
        if adaptive == 0: # fixed dt
            j_sol, steps = self.forward(dt0, u0)
        elif adaptive == 1: # time adaptive
            j_sol, steps = self.forward_le(dt0, u0)
        else:
            raise RuntimeError('invalid adaptive parameter')
        forward_time = time.time() - start_time
        
        # get error by reference solution
        #print('J NUM', j_sol, 'EX SOL', j_ref)
        true_error = float(abs(j_ref - j_sol))
        print('{} steps done in {} seconds'.format(steps, forward_time))
        return true_error, forward_time, steps, j_sol