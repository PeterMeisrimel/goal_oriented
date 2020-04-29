# -*- coding: utf-8 -*-
"""
Created on Fri Feb 02 09:54:36 2017
@author: Peter Meisrimel
credit to Patrick Farell
"""
from __future__ import division
import dolfin as dol
import numpy as np
from time_int import Time_int

## DG based time-integration class for building and solving adjoint equations
# Contains methods to solve forward and backward in time, including adaptivity
class Time_int_DWR(Time_int):
    ## Initialize function
    # @param self .
    # @param t_end end time
    # @param start_cells number of cells in initial grid
    # @param perc percentage for refinement
    # @param do_step forward timestep function
    # @param do_step_adj adjoint timestep function
    # @param get_dwr_est function for dwr estimate calculation
    # @param j density function
    # @return None
    def __init__(self, t_end, start_cells, perc, do_step = None, do_step_adj = None,
                 get_dwr_est = None, j = None):
        ## time discretization
        self.mesh = dol.IntervalMesh(int(start_cells), 0, int(t_end)) 
        ## percentage of cells to refine in a single refinement step
        self.perc = perc
        ## end time
        self.t_end = t_end
        self.do_step = do_step
        self.do_step_adj = do_step_adj
        ## dwr estimate function
        self.get_dwr_est = get_dwr_est
        
        self.j = j
        ## time discretization
        self.times = []
    
    ## Solve problem forward in time, given fixed timesteps
    # @param self .
    # @param u0 initial value
    # @param mesh mesh used for solving
    # @return j_sol, dictionary of solution
    def forward(self, u0, mesh):
        j_val = 0
        solns = {}
        uold = u0.copy(deepcopy = True)
        solns[0] = uold.copy(deepcopy = True)
        times = list(mesh.coordinates().ravel())
        times.sort()
        t = times[0]
        
        self.j_new = self.j(t, uold)
        for i in range(len(times)-1):
            t_new = times[i+1]
            dt = t_new - t
            unew = self.do_step(t, dt, uold)
            solns[t_new] = unew.copy(deepcopy = True) # store new value
            j_up, _ = self.quad_trapezoidal(t, dt, unew)
            
            j_val += j_up
            uold.assign(unew)
            t = t_new
        return j_val, solns
    
    ## Solving of adjoint problem
    # @param self .
    # @param z0 terminal value
    # @param mesh mesh used for solving
    # @return solution as dictionary
    def solve_adjoint(self, z0, mesh):
        zold = z0.copy(deepcopy = True)
        adj_solns = {}
        adj_solns[self.t_end] = zold.copy(deepcopy = True)
        times = list(mesh.coordinates().ravel())
        times.sort()
        times.reverse()
        t = times[0]
        for i in range(len(times)-1):
            t_new = times[i+1]
            dt = t_new - t
            znew = self.do_step_adj(t, dt, zold)
            adj_solns[t_new] = znew.copy(deepcopy = True) # store new value
            zold.assign(znew)
            t = t_new
        return adj_solns
    
    ## F
    # @param self .
    # @param tol tolerance
    # @param u0 initial value of forward problem
    # @param z0 terminal value of adjoint problem
    # @param mesh mesh from previous tolerance, None in case of initial mesh
    # @param est estimate from previous tolerance, None in case of initial mesh
    # @param est_list error estimate list from previous tolerance, None in case of initial mesh
    # @param j_prev functional result from previous run
    # @param j_ref reference solution
    # @return DWR Est, runtime, DWR Err, estimate list, j_sol, mesh
    def run(self, tol, u0, z0, mesh = None, est = np.inf, est_list = None, j_prev = 0, j_ref = 0):
        # runtime measurements
        import time
        start_time = time.time()
        j_val = j_prev # take previous result for J as base
        
        if mesh is not None: # mesh only None for initial grid
            self.mesh = mesh
        while True:
            # check if current grid meets tolerance bound
            if est <= tol: # grid sufficient
                break
            
            # refine grid, except for initial grid case
            if est_list is not None: # None if no grid set
                self.mesh = self.refine_basic(self.mesh, self.perc, abs(est_list))
            
            j_val, sol_coarse = self.forward(u0.copy(deepcopy = True), self.mesh)
            
            adj_coarse = self.solve_adjoint(z0.copy(deepcopy = True), self.mesh)
            mesh_fine = self.refine_basic(self.mesh, 1, [1]*(len(sol_coarse.keys()) - 1))
            
            #self.forward(u0.copy(deepcopy = True), mesh_fine) # needed in non linear case
            adj_fine = self.solve_adjoint(z0.copy(deepcopy = True), mesh_fine)
            
            t_fine = list(adj_fine.keys())
            t_fine.sort()
            
            est_list = self.get_dwr_est(sol_coarse, adj_coarse, adj_fine)
            
            est = sum(est_list)
            print('DWR est : {} for {} cells'.format(est, self.mesh.num_cells()))
        true_error = abs(j_ref - j_val)
        print('TRUE ERROR', true_error)
        dwr_time = time.time() - start_time
        
        return est, dwr_time, true_error, est_list, j_val, self.mesh
    
    ## DWR refinement routine \n
    # splits 'self.perc' of all timeinterval into half, starting by the ones with largest weights
    # @param self .
    # @param mesh mesh to refine
    # @param perc percentage of mesh cells to refine
    # @param weights DWR error weights 
    # @return (list) new and refine time discretization
    def refine_basic(self, mesh, perc, weights):
        num = int(np.ceil(len(weights)*perc))
        markers = dol.MeshFunction('bool', mesh, dim = 1)
        comb = list(zip(weights, range(len(weights))))
        comb.sort(key = lambda a : a[0], reverse = True) # sort by weights
        for i in range(num):
            markers[comb[i][1]] = True
        return dol.refine(mesh, markers)
    
    ## simple routine to solve forward and backward problem, returning solution of backward problem
    # @param self .
    # @param u0 initial value of forward problem
    # @param z0 terminal value of adjoint problem
    # @return list of times used, dictionary of adjoint solution with times as keys
    def run_adj(self, u0, z0):
        self.forward(u0, self.mesh)
        adj_sol = self.solve_adjoint(z0, self.mesh)
        times = list(self.mesh.coordinates().ravel())
        times.sort()
        return times, adj_sol