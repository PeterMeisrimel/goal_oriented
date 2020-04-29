# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 16:38:46 2017
@author: Peter Meisrimel
"""
from __future__ import division
import dolfin as dol
from problem import Problem, Problem_FE
import numpy as np

## Verification est problem \f$ u'(t) = -u(t), \quad u(0) = 1 \f$.
# \n problem type = -1
class Problem_verification(Problem):
    ## init function
    # @param self .
    # @param t_end end time of problem
    # @param func which density function to use
    # @return None
    def __init__(self, t_end = None, func = None):
        self.t_end = 1 if t_end is None else t_end
        
        self.func = func
        
        self.u0 = np.array([1.])
        
    ## density function :\f$ j(t, u) \f$
    # @param self .
    # @param t time
    # @param u state vector
    # @return j(t, u)
    def j(self, t, u):
        if   self.func == 1:  return u[0]
        elif self.func == 21: return u[0]**2
        elif self.func == 31: return u[0]**3
        else: raise RuntimeError('invalid functional for this type of problem')
     
    ## Function to get a pointer to rhs function
    # @param self .
    # @return rhs function
    def get_rhs(self):
        def rhs(t, u):
            return -u
        return rhs
    
    ## Return solution of integral over time
    # @param self .
    # @return Reference solution
    def solution_int(self):
        from numpy import exp
        if self.func == 1:
            def res(t):  return -exp(-t)
        elif self.func == 21:
            def res(t):  return -exp(-2*t)*0.5
        elif self.func == 31:
            def res(t):  return -exp(-3*t)/3.
        else:
            print('RuntimeWarning: No reference solution available')
            return 0.
        return res(self.t_end) - res(0)

## Verification problem for CN case
class Problem_verification_DG(Problem_verification, Problem_FE):
    ## Initialize
    # @param self .
    # @param t_end t_e
    # @param func which density function to use, compare self.j
    # @return None
    def __init__(self, t_end = None, func = None):
        Problem_verification.__init__(self, t_end, func)
        
        self.u0_expr = dol.Constant(1)
        self.V = dol.FunctionSpace(dol.UnitIntervalMesh(1), 'DG', 0)
        
        Problem_FE.__init__(self)
        # CN, higher order
        F = dol.inner(self.v, self.utrial - self.uold + self.dt * 0.5 * (self.uold + self.utrial))*dol.dx
        prob = dol.LinearVariationalProblem(dol.lhs(F), dol.rhs(F), self.unew)
        self.solver = dol.LinearVariationalSolver(prob)
        # IE, lower order
        Flow = dol.inner(self.v, self.utrial - self.uold + self.dt * self.utrial)*dol.dx
        problow = dol.LinearVariationalProblem(dol.lhs(Flow), dol.rhs(Flow), self.ulow)
        self.solver_low = dol.LinearVariationalSolver(problow)
        
    ## density function
    # @param self pointer to itself
    # @param t time
    # @param u input vector
    # @return j(t, u)
    def j(self, t, u):
        return Problem_verification.j(self, t, dol.assemble(dol.dot(u,self.v)*dol.dx).get_local())
            
## Verification est problem \f$ u'(t) = -u(t) + t^2, \quad u(0) = 1 \f$
# \n problem_type = -2
class Problem_verification_time(Problem_verification):
    ## Gets right-hand side function as a pointer
    # @param self .
    # @return right-hand side function
    def get_rhs(self):
        def rhs(t, u):
            return np.array([-u[0] + t**2])
        return rhs
    
    ## Integral solution
    # @param self .
    # @return Reference solution
    def solution_int(self):
        if self.func == 1:
            def res(t):  return t**3/3. - t**2 + 2*t + np.exp(-t)
        elif self.func == 21:
            def res(t):  return np.exp(-2*t)*(-15 + 60*np.exp(t)*(2 + t**2) + 
                                2*np.exp(2*t)*t*(60 - 60*t + 40*t**2 - 15*t**3 + 3*t**4))/30.
        elif self.func == 31:
            def res(t):  return np.exp(-3*t)*(140 - 315*np.exp(t)*(3 - 2*t + 2*t**2) + 1260*np.exp(2*t)*(12 + 8*t + 8*t**2 + t**4) +
                                12*np.exp(3*t)*t*(280 - 420*t + 420*t**2 - 280*t**3 + 126*t**4 - 35*t**5 + 5*t**6))/420.
        else:
            print('RuntimeWarning: No reference solution available')
            return 0.
        return res(self.t_end) - res(0)
    
## Time dependant verification problem for using CN
class Problem_verification_time_DG(Problem_verification_time, Problem_verification_DG, Problem_FE):
    time_dep = True
    ## Initialize
    # @param self .
    # @param t_end t_e
    # @param func which density function to use, compare self.j
    # @return None
    def __init__(self, t_end = None, func = None):
        Problem_verification.__init__(self, t_end, func)
        
        self.u0_expr = dol.Constant(1)
        self.V = dol.FunctionSpace(dol.UnitIntervalMesh(1), 'DG', 0)
        
        # rhs
        self.f1 = dol.Expression('pow(t, 2)', t = 0, degree = 2)
        self.f2 = dol.Expression('pow(t, 2)', t = 0, degree = 2)
        
        Problem_FE.__init__(self)
        # CN, higher order
        F = (dol.inner(self.v, self.utrial - self.uold + self.dt * 0.5 * (self.uold + self.utrial))*dol.dx
             - 0.5*self.dt*dol.inner(self.f1 + self.f2, self.v)*dol.dx)
        prob = dol.LinearVariationalProblem(dol.lhs(F), dol.rhs(F), self.unew)
        self.solver = dol.LinearVariationalSolver(prob)
        # IE, lower order
        Flow = (dol.inner(self.v, self.utrial - self.uold + self.dt * self.utrial)*dol.dx
                - self.dt*dol.inner(self.f2, self.v)*dol.dx)
        problow = dol.LinearVariationalProblem(dol.lhs(Flow), dol.rhs(Flow), self.ulow)
        self.solver_low = dol.LinearVariationalSolver(problow)