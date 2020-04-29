# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 16:09:34 2015
@author: Peter Meisrimel
"""
from __future__ import division
import dolfin as dol
import numpy as np

## Head class for defining Problems
class Problem():
    ## 2 norm function for scalar case
    # @param self .
    # @param u state vector
    # @return /f$ \|u\|_2^2 /f$
    def get_2_norm_sq(self, u):
        return np.linalg.norm(u, 2)**2
        
## header class for general problem using a FEniCs based FE discretization
class Problem_FE(Problem):
    ## Marker if problem has a time dependant term to take care of in time-integration
    # to be set as True in respective constructors if needed
    time_dep = False
    ## Initialization
    # @param self .
    # @param adjoint addtional initialization of adjoint functions if True
    # @return None
    def __init__(self, adjoint = False):
        self.dt     = dol.Constant(0)
        self.uold   = dol.interpolate(self.u0_expr, self.V)
        self.u0     = dol.interpolate(self.u0_expr, self.V)
        self.unew   = dol.Function(self.V)
        self.ulow   = dol.Function(self.V)
        self.v      = dol.TestFunction(self.V)
        self.utrial = dol.TrialFunction(self.V)
        
        if adjoint:
            self.zold   = dol.interpolate(self.z0_expr, self.V)
            self.z0     = dol.interpolate(self.z0_expr, self.V)
            self.znew   = dol.Function(self.V)
            self.ztrial = dol.TrialFunction(self.V)
    
    ## 2 norm in FEniCs case
    # @param self .
    # @param u state vector (function)
    # @return /f$ \|u\|_2^2 /f$
    def get_2_norm_sq(self, u):
        return dol.assemble(dol.inner(u, u)*dol.dx)
    
    ## Do forward step, including local error estimate
    # @param self .
    # @param t time
    # @param dt timestep
    # @param uold state vector
    # @return unew, l_err = unew - ulow
    def do_step_err(self, t, dt, uold):
        self.uold.assign(uold)
        self.dt.assign(dt)
        if self.time_dep:  
            self.f1.t = t
            self.f2.t = t + np.float64(dt)
        self.solver.solve()
        self.solver_low.solve()
        return self.unew, self.unew - self.ulow
    
    ## do forward step, without local error estimate
    # @param self .
    # @param t time
    # @param dt timestep
    # @param uold state vector
    # @return unew
    def do_step(self, t, dt, uold):
        self.uold.assign(uold)
        self.dt.assign(dt)
        if self.time_dep:  
            self.f1.t = t
            self.f2.t = t + np.float64(dt)
        self.solver.solve()
        return self.unew
    
    ## do "forward" step of adjoint problem
    # @param self .
    # @param t time
    # @param dt timestep
    # @param zold state vector
    # @return znew
    def do_step_adj(self, t, dt, zold):
        self.zold.assign(zold)
        self.dt.assign(dt)
        self.solver_adj.solve()
        return self.znew