# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 16:09:34 2015
@author: Peter Meisrimel
credit to Patrick Farell
"""
from __future__ import division
import dolfin as dol
from problem import Problem, Problem_FE
import numpy as np

## Class for 2x2 scalar problem from Section 5.1
class Problem_Basic(Problem):
    ## Initilization
    # @param self .
    # @param t_end |f$ t_e |f$
    # @param func determines which functional to use, c.f. j function
    # @param para_stiff k value
    # @return None
    def __init__(self, t_end = None, func = None, para_stiff = None):
        self.t_end = 2 if t_end is None else t_end
        
        self.func = func
        self.k = para_stiff
        
        self.u0 = np.array([1., 1.])
        self.A  = np.array([[-1., 1.],[0, self.k]])
        
    ## function to get pointer to right-hand side
    # @param self .
    # @return right-hand side function
    def get_rhs(self):
        def rhs(t, u):
            return self.A@u
        return rhs
        
    ## density function j 
    # @param self .
    # @param t time
    # @param u input vector
    # @return j(t, u)
    def j(self, t, u):
        if self.func == 1:    return u[0]
        elif self.func == 2:  return u[1]
        elif self.func == 10: return u[0] * u[1]
        elif self.func == 21: return u[0]**2
        elif self.func == 22: return u[1]**2
        elif self.func == 31: return u[0]**3
        elif self.func == 32: return u[1]**3
        else:
            raise RuntimeError('invalid functional for this type of problem')
            
    ## Exact solution evaluated in functional j
    # @param self .
    # @return J(u)
    def solution_int(self):
        from numpy import exp
        u1, u2 = 1., 1.
        k = float(self.k)
        if self.func == 1:
            def res(t):
                if k != -1: return (exp(-t)*(-k**2*u1 + exp((1 + k)*t)*u2 + k*(-u1 + u2)))/(k*(1 + k))
                else:       return -exp(-t)*(u1 + u2 + t*u2)
        elif self.func == 21:
            def res(t):
                if k != -1: return (exp(-2*t)*(-k**4*u1**2 - k**3*u1*(u1 - 2*u2) - exp(2*(1 + k)*t)*u2**2 + k**2*(u1**2 + 
                                    4*exp((1 + k)*t)*u1*u2 - u2**2) + k*(u1**2 + 2*(-1 + 2*exp((1 + k)*t))*u1*u2 + (1 - 4*exp((1 + 
                                    k)*t) + exp(2*(1 + k)*t))*u2**2)))/(2*(-1 + k)*k*(1 + k)**2)
                else:       return -(1/4)*exp(-2*t)*(2*u1**2 + (1 + 2*t + 2*t**2)*u2**2 + 2*u1*(u2 + 2*t*u2))
        elif self.func == 31:
            def res(t):
                if k != -1: return (1/(3*(-2 + k)*k*(1 + k)**3*(-1 + 2*k)))*exp(-3*t)*(-2*k**6*u1**3 - k**5*u1**2*(u1 - 6*u2) + 
                                         2*exp(3*(1 + k)*t)*u2**3 + k**4*u1*(7*u1**2 + 3*(-1 + 6*exp((1 + k)*t))*u1*u2 - 6*u2**2) + 
                                         k**3*(7*u1**3 + 9*(-2 + 3*exp((1 + k)*t))*u1**2*u2 + 9*(1 - 4*exp((1 + k)*t) + exp(2*(1 + 
                                         k)*t))*u1*u2**2 + 2*u2**3) - k**2*(u1**3 + 3*u1**2*u2 + 9*(-1 + 2*exp((1 + k)*t) + exp(2*(1 + 
                                         k)*t))*u1*u2**2 - (-5 + 18*exp((1 + k)*t) - 9*exp(2*(1 + k)*t) + 2*exp(3*(1 + k)*t))*u2**3) - 
                                         k*(2*u1**3 + 3*(-2 + 3*exp((1 + k)*t))*u1**2*u2 + 6*(1 - 3*exp((1 + k)*t) + 3*exp(2*(1 + 
                                         k)*t))*u1*u2**2 + (-2 + 9*exp((1 + k)*t) - 18*exp(2*(1 + k)*t) + 5*exp(3*(1 + k)*t))*u2**3))
                else:       return 1/27*exp(-3*t)*(-9*u1**3 - 3*(2 + 6*t + 9*t**2)*u1*u2**2 - (2 + 6*t + 9*t**2 + 9*t**3)*u2**3 - 
                                    9*u1**2*(u2 + 3*t*u2))
        elif self.func == 2:
            def res(t):          return (exp(k*t)*u2)/k
        elif self.func == 22:
            def res(t):          return (exp(2*k*t)*u2**2)/(2*k)
        elif self.func == 32:
            def res(t):          return (exp(3*k*t)*u2**3)/(3*k)
        elif self.func == 10:
            def res(t):
                if k != -1: return (exp((-1 + k)*t)*u2*(2*k**2*u1 - exp((1 + k)*t)*u2 + k*(2*u1 + (-2 + exp((1 + 
                                    k)*t))*u2)))/(2*k*(-1 + k**2))
                else:       return -(1/4)*exp(-2*t)*u2*(2*u1 + u2 + 2*t*u2)
        else:                    raise ValueError('No reference solution available')
        return res(self.t_end) - res(0)

## Class for 2x2 scalar problem using CN
class Problem_Basic_DG(Problem_FE, Problem_Basic):
    ## Initialization
    # @param self .
    # @param t_end t_e
    # @param func which functional to use c.f. self.f
    # @param para_stiff k value
    # @param adjoint set True if using adjoints
    # @return None
    def __init__(self, t_end = None, func = None, para_stiff = None, adjoint = False):
        Problem_Basic.__init__(self, t_end = t_end, func = func, para_stiff = para_stiff)
        
        self.k = dol.Constant(self.k)
        self.u0_expr = dol.Constant(self.u0) # initial value
        
        mesh = dol.UnitIntervalMesh(1)
        R_elem = dol.FiniteElement("R", mesh.ufl_cell(), 0)
        V_elem = dol.MixedElement([R_elem, R_elem])
        self.V = dol.FunctionSpace(mesh, V_elem)
        
        if adjoint:
            self.z0_expr = dol.Constant(np.array([0., 0.])) # initial value adj
            
        Problem_FE.__init__(self, adjoint)
        
        (self.v1     , self.v2)      = dol.split(self.v)
        (self.u1trial, self.u2trial) = dol.split(self.utrial)
        (self.u1old  , self.u2old)   = dol.split(self.uold)
        (self.u1low  , self.u2low)   = dol.split(self.ulow)
        
        ## Crank nicolson weak formulation
        F = (dol.inner(self.v1, self.u1trial - self.u1old + self.dt * (0.5 * (self.u1old + self.u1trial) - 0.5 * (self.u2old + self.u2trial)))*dol.dx
           + dol.inner(self.v2, self.u2trial - self.u2old - self.k * self.dt * 0.5 * (self.u2old + self.u2trial))*dol.dx)
        prob = dol.LinearVariationalProblem(dol.lhs(F), dol.rhs(F), self.unew)
        self.solver = dol.LinearVariationalSolver(prob)
        ## Implicit Euler weak formulation for error estimation
        Flow = (dol.inner(self.v1, self.u1trial - self.u1old + self.dt * (self.u1trial - self.u2trial))*dol.dx
              + dol.inner(self.v2, self.u2trial - self.u2old - self.k * self.dt * self.u2trial)*dol.dx)
        problow = dol.LinearVariationalProblem(dol.lhs(Flow), dol.rhs(Flow), self.ulow)
        self.solver_low = dol.LinearVariationalSolver(problow)

        if adjoint:
            (self.z1old  , self.z2old)   = dol.split(self.zold)
            (self.z1trial, self.z2trial) = dol.split(self.ztrial)
            
            if self.func not in [1, 2]:
                raise ValueError('DWR not (yet) implemented for this functional')
                
            adj_src = dol.Function(self.V)
            if   self.func == 1:    adj_src.interpolate(dol.Constant((1, 0)))
            elif self.func == 2:    adj_src.interpolate(dol.Constant((0, 1)))
            
            src1, src2 = dol.split(adj_src)
 
            Fadj = (dol.inner(self.z1trial - self.z1old + 0.5 * self.dt * (-self.z1trial - self.z1old + 2*src1), self.v1)*dol.dx +
                    dol.inner(self.z2trial - self.z2old + 0.5 * self.dt * ( self.z1trial + self.z1old + self.k*(self.z2trial + self.z2old) + 2*src2), self.v2)*dol.dx)
            prob_adj = dol.LinearVariationalProblem(dol.lhs(Fadj), dol.rhs(Fadj), self.znew)
            self.solver_adj = dol.LinearVariationalSolver(prob_adj)
    
    ## Function to calculate 2 norm squared            
    # @param self .
    # @param u vector
    # @return |f$ \|u\|_2^2 |f$
    def get_2_norm_sq(self, u):
        return dol.assemble(dol.inner(u,u)*dol.dx)
        
    ## density function
    # @param self pointer to itself
    # @param t time
    # @param u input vector
    # @return j(t,u)
    def j(self, t, u):
        return Problem_Basic.j(self, t, dol.assemble(dol.dot(u,self.v)*dol.dx).get_local())
        
    ## Construct DWR estimatese_abs = file_read['use_abs'],
    # @param self pointer to itself
    # @param solns forward solution
    # @param adj_coarse adjoint solution on coarse grid
    # @param adj_fine adjoint solution on fine grid
    # @return estimate vector
    def get_dwr_est(self, solns, adj_coarse, adj_fine):
        k = float(self.k)
        times_coarse = list(adj_coarse.keys())
        times_coarse.sort()
            
        times_fine = list(adj_fine.keys())
        times_fine.sort()
            
        ntimesteps = len(times_coarse) - 1
        est = np.zeros(ntimesteps)
            
        u10, u20 = solns[times_coarse[0]].vector().get_local()
        zc10, zc20 = adj_coarse[times_coarse[0]].vector().get_local()
        zf10, zf20 = adj_fine[times_fine[0]].vector().get_local()
        for t in range(ntimesteps):
            # acess values from vectors directly
            u11, u21 = solns[times_coarse[t+1]].vector().get_local()
            zc11, zc21 = adj_coarse[times_coarse[t+1]].vector().get_local()
            zf11, zf21 = adj_fine[times_fine[2*t + 1]].vector().get_local()
            zf12, zf22 = adj_fine[times_fine[2*t + 2]].vector().get_local()
            dt = float(times_coarse[t + 1] - times_coarse[t])
            
            ut1 = (u10 - u11)/dt
            res_1_left = ut1 + u10 - u20
            res_1_right = ut1 + u11 - u21
            res_1_mid = (res_1_left + res_1_right)/2
            z1_left = zf10 - zc10
            z1_right = zf12 - zc11
            z1_mid = zf11 - (zc10 + zc11)/2
            
            part_1_left = res_1_left*z1_left
            part_1_right = res_1_right*z1_right
            part_1_mid = res_1_mid*z1_mid
            
            ut2 = (u20 - u21)/dt
            res_2_left = ut2 - k*u20
            res_2_right = ut2 - k*u21
            res_2_mid = (res_2_left + res_2_right)/2
            z2_left = zf20 - zc20
            z2_right = zf22 - zc21
            z2_mid = zf21 - (zc20 + zc21)/2
            
            part_2_left = res_2_left*z2_left
            part_2_right = res_2_right*z2_right
            part_2_mid = res_2_mid*z2_mid
            
            est[t] = (abs(dt/4*(part_1_left + 2*part_1_mid + part_1_right) + 
                      dt/4*(part_2_left + 2*part_2_mid + part_2_right)))
            # shift of values by one
            u10, u20, zc10, zc20, zf10, zf20 = u11, u21, zc11, zc21, zf12, zf22
        return est