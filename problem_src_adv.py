# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:45:27 2017
@author: Peter Meisrimel
"""
from __future__ import division
import numpy as np
import dolfin as dol
from problem import Problem_FE

## Class for convection diffusion problem with source term
# \n problem_type = 12
class Problem_heat_source_adv(Problem_FE):
    time_dep = True
    ## Initialization basics
    # @param self .
    # @param gridsize for spatial discretization
    # @param t_end end time of problem
    # @param direction direction of advection, 1 = right, 0 = left
    # @param adjoint set to True to initialize adjoint related parameters
    # @return None
    def __init__(self, gridsize = 32, t_end = 6, direction = 1, adjoint = False):
        ## end time
        self.t_end = t_end
        
        self.u0_expr = dol.Expression('1 + 0.2*sin(pi*x[1])*sin(pi*x[0]/3)', degree = 1)
        
        if adjoint:
            self.z0_expr = dol.Expression("0", degree = 0)
        
        ## heat source term for weak formulation for Crank Nicolson, IE uses only f1
        self.f1 = dol.Expression('5*pow(t,3)', t = 0, degree = 2)
        self.f2 = dol.Expression('5*pow(t,3)', t = 0, degree = 2)
        
        ## Size of grid, number of cells = (2*gridsize)*gridsize*2
        self.gridsize = gridsize
        ## Mesh 
        mesh = dol.RectangleMesh(dol.Point(0,0), dol.Point(3,1), 3*self.gridsize, self.gridsize)
        ## Function space for solution vector
        self.V = dol.FunctionSpace(mesh, "CG", 1)
        
        Problem_FE.__init__(self, adjoint)
        ## Function space for velocity
        V_velo = dol.VectorFunctionSpace(mesh, "CG", 2)
        
        ## direction of velocity
        self.direction = direction
        
        # create dol.Measures for relevant subdomains
        colours = dol.MeshFunction("size_t", mesh, dim = 2) # Colouring for mesh to identify various subdomains
        colours.set_all(0) # default colour
        
        # heat source
        heat_src = dol.CompiledSubDomain("x[0] >= 0.25 && x[0] <= 0.75 && x[1] >= 0.25 && x[1] <= 0.75") # identify subdomain by condition
        heat_src.mark(colours, 2) # change colour of domain
        self.dx_src = dol.Measure("dx", subdomain_data=colours)(2) # define domain in the needed way for function definition
        
        # Area of interest
        subdom = dol.CompiledSubDomain("x[0] >= 2.25 && x[0] <= 2.75 && x[1] >= 0.25 && x[1] <= 0.75")
        subdom.mark(colours, 1) # change colour of domain
        self.dx_obs = dol.Measure("dx", subdomain_data = colours)(1) # define domain in the needed way for function definition
        self.dx_cont = self.dx_obs
        
        ## u used in Crank Nicolson
        self.ucn = 0.5 * (self.utrial + self.uold)
        if   self.direction ==  1: self.velocity = dol.interpolate(dol.Expression((' 1', '0'), degree = 0), V_velo)
        elif self.direction == -1: self.velocity = dol.interpolate(dol.Expression(('-1', '0'), degree = 0), V_velo)
        else: raise ValueError('invalid advection direction, please try again')
        
        self.heat_cond = dol.Constant(0.01)
        self.outflow = dol.Constant(0.15)
        self.velo = dol.Constant(0.5)
        ## Variational formulation for Crank Nicolson
        F = (dol.inner(self.v, self.utrial - self.uold)*dol.dx
           + self.dt*self.heat_cond*dol.inner(dol.grad(self.ucn), dol.grad(self.v))*dol.dx
           - 0.5*self.dt*dol.inner(self.f1 + self.f2, self.v)*self.dx_src
           + self.velo*self.dt*dol.dot(dol.grad(self.ucn), self.velocity)*self.v*dol.dx
           + self.dt*self.heat_cond*self.outflow*dol.inner(self.ucn, self.v)*dol.ds)
        prob = dol.LinearVariationalProblem(dol.lhs(F), dol.rhs(F), self.unew)
        self.solver = dol.LinearVariationalSolver(prob)
        
        ## Variational formulation for Implicit Euler
        Flow = (dol.inner(self.v, self.utrial - self.uold)*dol.dx
                      + self.dt*self.heat_cond*dol.inner(dol.grad(self.utrial), dol.grad(self.v))*dol.dx
                      - self.dt*dol.inner(self.f2, self.v)*self.dx_src
                      + self.velo*self.dt*dol.dot(dol.grad(self.utrial), self.velocity)*self.v*dol.dx
                      + self.dt*self.heat_cond*self.outflow*dol.inner(self.utrial, self.v)*dol.ds)
        problow = dol.LinearVariationalProblem(dol.lhs(Flow), dol.rhs(Flow), self.ulow)
        self.solver_low = dol.LinearVariationalSolver(problow)
        
        if adjoint:
            adj_src = dol.Function(self.V)
            adj_src.interpolate(dol.Constant(1/self.t_end))
                    
            self.src_weight = dol.Expression("((0.25 < x[0]) and (0.75 > x[0]) and (0.25 < x[1]) and (0.75 > x[1]))? 1 : 0", degree = 0)
                    
            class LeftBoundary(dol.SubDomain):
                def inside(self, x, on_boundary):
                    return on_boundary and dol.near(x[0], 0)
                        
            class RightBoundary(dol.SubDomain):
                def inside(self, x, on_boundary):
                    return on_boundary and dol.near(x[0], 3)
                            
            left = LeftBoundary()
            right = RightBoundary()
                            
            boundaries = dol.MeshFunction("size_t", mesh, dim = 1)
            boundaries.set_all(0)
            left.mark(boundaries, 1)
            right.mark(boundaries, 3)
                    
            ds_self = dol.Measure('ds', domain = mesh, subdomain_data=boundaries)
            zcn = 0.5*(self.ztrial + self.zold)
    
            Fadj = (dol.inner(self.ztrial - self.zold, self.v)*dol.dx -
                   self.dt*self.heat_cond*dol.inner(dol.grad(zcn), dol.grad(self.v))*dol.dx +
                   self.dt*dol.inner(adj_src, self.v)*self.dx_obs +
                   self.direction*self.dt*self.velo*dol.dot(self.velocity, dol.grad(zcn))*self.v*dol.dx -
                   self.velo*self.dt*dol.inner(zcn, self.v)*ds_self(1) -
                   self.velo*self.dt*dol.inner(zcn, self.v)*ds_self(3) - 
                   self.outflow*self.heat_cond*self.dt*dol.inner(zcn, self.v)*dol.ds)
            probadj = dol.LinearVariationalProblem(dol.lhs(Fadj), dol.rhs(Fadj), self.znew)
            self.solver_adj = dol.LinearVariationalSolver(probadj)
            
    ## Density function
    # @param self .
    # @param t time
    # @param u state vector
    # @return j(t, u)
    def j(self, t, u):
        return dol.assemble(u*self.dx_obs)/self.t_end
        
    ## DWR error estimation
    # @param self .
    # @param solns forward solution
    # @param adj_coarse adjoint solution of coarse grid
    # @param adj_fine adjoint solution of fine grid
    # @return Error estimate vector
    def get_dwr_est(self, solns, adj_coarse, adj_fine):
        times_coarse = list(adj_coarse.keys())
        times_coarse.sort()
            
        times_fine = list(adj_fine.keys())
        times_fine.sort()
        
        ntimesteps = len(times_coarse) - 1
        est = np.zeros(ntimesteps)
        
        u0 = solns[times_coarse[0]]
        
        div0 = self.heat_cond*dol.div(dol.grad(u0))
        adv0 = self.direction*self.velo*dol.dot(dol.grad(u0), self.velocity)
        zc0 = adj_coarse[times_coarse[0]];
        f0 = dol.Function(self.V); self.f1.t = times_coarse[0]; f0.interpolate(self.f1)
        f0 = f0*self.src_weight
        zf0 = adj_fine[times_fine[0]];
        for t in range(ntimesteps):
            u1 = solns[times_coarse[t + 1]]
            div1 = self.heat_cond*dol.div(dol.grad(u1))
            adv1 = self.direction*self.velo*dol.dot(dol.grad(u1), self.velocity)
            f1 = dol.Function(self.V); self.f1.t = times_coarse[t + 1]; f1.interpolate(self.f1)
            f1 = f1*self.src_weight
            zc1 = adj_coarse[times_coarse[t + 1]]
            zf1 = adj_fine[times_fine[2*t + 1]]; zf2 = adj_fine[times_fine[2*t + 2]];
            dt = dol.Constant(times_coarse[t + 1] - times_coarse[t])

            ut = (u1 - u0)/dt
            res_left = ut - div0 + adv0 - f0
            res_right = ut - div1 + adv1 - f1
            res_mid = (res_left + res_right)/2
            zz_left = zf0 - zc0
            zz_right = zf2 - zc1
            zz_mid = zf1 - (zc0 + zc1)/2

            part_left = dol.assemble(res_left*zz_left*dol.dx)
            part_right = dol.assemble(res_right*zz_right*dol.dx)
            part_mid = dol.assemble(res_mid*zz_mid*dol.dx)
                
            est[t] = abs(dt/4*(part_left + 2*part_mid + part_right))
                
            u0, div0, zc0, f0, zf0 = u1, div1, zc1, f1, zf2
        return est
    
    ## Reference solution to functional for a given density function
    # @param self .
    # @return Reference solution
    def solution_int(self):
        if self.t_end == 6. and self.gridsize == 32 and self.direction == 1: # tol = 1.e-6
            return 1.04505531781361082899195480422349646687507629394531 # tol = 1e-8
        if self.t_end == 3. and self.gridsize == 32 and self.direction == -1: # tol = 1.e-6
            return 0.24177465491821051313259260950871976092457771301270 # tol = 1e-9
        print('RuntimeWarning: No reference solution available')
        return 0.