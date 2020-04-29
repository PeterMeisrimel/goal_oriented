# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:59:24 2017
@author: Peter Meisrimel
"""
from __future__ import division
from numpy import inf

## class for timestep controllers and functions to find initial timesteps
class Timestep_controller():
    ## initialization function
    # @param self .
    # @param tol tolerance
    # @param f_min lower limit for change rate of timestep
    # @param f_max upper limit for change rate of timestep
    # @param f_safety safety factor to multiply every timestep with
    # @param order_low order of lower order method
    # @return None
    def __init__(self, tol = 1, f_min = 0, f_max = inf, f_safety = 1, order_low = 1):
        self.tol = tol
        self.f_min = f_min
        self.f_max = f_max
        self.f_safety = f_safety
        self.phat = order_low
    
    ## Deadbeat controller for classic norm based error estimates
    # @param self .
    # @param err error estimate
    # @param dt previous timestep
    # @return new timestepsize
    def deadbeat_classic(self, err, dt):
        ind = (self.f_safety * self.tol/abs(err))**(1/(self.phat + 1))
        return dt * max(min(ind, self.f_max), self.f_min)     
    
    ## take tolerance with appropriate power as initial timestep
    # @param self .
    # @param cont_type type of controller being used
    # @return /f$ tol^{1/(phat + 1)}/f$
    def dt0_tol_scaled(self, cont_type):
        return self.tol**(1/(self.phat + 1))
        
    ## take tolerance as initial timestep
    # @param self .
    # @param cont_type type of controller being used
    # @return self.tol
    def dt0_tol(self, cont_type):
        return self.tol     
    
    ## function to return correct controller
    # @param self .
    # @param cont_type type of controller being used
    # @return controller
    def get_controller(self, cont_type):
        if cont_type == 0:
            return self.deadbeat_classic
        else:
            raise ValueError('Timestep controller not defined')

    ## function to return \f$ \Delta t_0 \f$ correct controller
    # @param self .
    # @param cont_type type of controller being used
    # @return controller
    def get_dt0_controller(self, cont_type):
        if cont_type == 0:
            return self.dt0_tol_scaled
        elif cont_type == 1:
            return self.dt0_tol
        else:
            raise ValueError('Initial timestep scheme not defined')