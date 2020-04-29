# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 13:39:21 2016
@author: Peter Meisrimel
"""
from __future__ import division
from scipy.optimize import newton_krylov
import numpy as np

## Parent class for general one step methods, embedded RK schemes
class One_step_method():
    ## RK schemes will be denoted by their Butcher tableau;
    # A matrix of coefficients, c row sums of A, b weights, bhat embedded weights, k stage derivatives
    
    ## Basic initialization
    # @param self .
    # @param rhs right-hand side function
    # @return None
    def __init__(self, rhs):
        ## right-hand side function
        self.rhs = rhs
        
        self.stages = len(self.b)
        
        self.c = [sum([self.A[i,j] for j in range(self.stages)]) for i in range(self.stages)]
        ## placeholder list for stage derivatives
        self.k = [0]*self.stages
        
    ## Using RK for quadrature
    # @param self .
    # @param t time
    # @param uold unew
    # @param dt stepsize
    # @param j density function j
    # @param diag_plus default assumes strict lower part of A in butcher tableau only, diag_plus tells how many additional diagonal should be used
    # @return quadrature increment, quadrature error estimate
    def do_quadrature(self, t, uold, dt, j, diag_plus = 0):
        j_update, j_quad_est = 0, 0
        
        for i in range(self.stages):
            j_stage = j(t + dt*self.c[i], uold + dt*sum([self.A[i, m]*self.k[m] for m in range(i + diag_plus)]))
            j_update += self.b[i] * j_stage
            j_quad_est += (self.b[i] - self.b_hat[i]) * j_stage
        return dt*j_update, dt*abs(j_quad_est)

## Contruct root problem for DIRK schemes to be used in Newton
# @param t current time point
# @param dt timestep
# @param u u vector
# @param rhs right-hand side function
# @param gamma diagonal value of DIRK scheme
# @param c c value corresponding to current stage
# @return root problem function
def f_root(t, dt, u, rhs, gamma = 1, c = 1):
    def f(x):
        return rhs(t + c*dt, u + dt*gamma*x) - x
    return f

## Function to select and initialize correct RK scheme according to input file 
# @param rhs right-hand side function
# @param method_no indentifier for method to be used
# @return pointer to initialized scheme
def RK_get_class(rhs, method_no):
    if   method_no == -21: return Impl_SDIRK2_Ellsiep (rhs)
    elif method_no == -32: return Impl_SDIRK3_Cash    (rhs)
    elif method_no == 21 : return Expl_RK_Heun_Euler  (rhs)
    elif method_no == 32 : return Expl_RK_Bog_Shamp   (rhs)
    elif method_no == 45 : return Expl_RK_Fehlberg    (rhs)
    elif method_no == 541: return Expl_RK_Cash_Karp   (rhs)
    elif method_no == 542: return Expl_RK_Do_Pri      (rhs)
    raise ValueError('Scheme with desired orders not (yet) implemented')

## Generic DRIK scheme
class Impl_DIRK(One_step_method):
    ## general function to perform a single step
    # @param self .
    # @param t time
    # @param dt stepsize
    # @param uold current u value
    # @param j density function, not None when using time-integration for quadrature
    # @return unew, l_err = unew - ulow
    def do_step(self, t, dt, uold, j = None):
        for i in range(self.stages):
            temp = uold + dt*sum([self.A[i,m] * self.k[m] for m in range(i)])
            try:
                self.k[i] = newton_krylov(f_root(t, dt, temp, self.rhs, self.A[i,i], self.c[i]),
                                          uold if i == 0 else self.k[i-1], f_tol = 1.e-10, f_rtol = 1.e-10)
            except ValueError:
                self.k[i] = newton_krylov(f_root(t, dt, temp, self.rhs, self.A[i,i], self.c[i]),
                                          uold, f_tol = 1.e-10, f_rtol = 1.e-10)
        
        unew = uold + dt*sum([self.b[i]*self.k[i] for i in range(self.stages)])
        ulow = uold + dt*sum([self.b_hat[i]*self.k[i] for i in range(self.stages)])
            
        if j is not None:
            j_up, j_quad_est = self.do_quadrature(t, uold, dt, j, 1)
            return unew, unew - ulow, j_up, j_quad_est

        return unew, unew - ulow
        
## Generic explicit RK method
class Expl_RK(One_step_method):
    ## general function to perform a single step
    # @param self .
    # @param t time
    # @param dt stepsize
    # @param uold current u value
    # @param j density function, not None when using time-integration for quadrature
    # @return unew, l_err = unew - ulow
    def do_step(self, t, dt, uold, j = None):

        for i in range(self.stages):
            self.k[i] = self.rhs(t + dt*self.c[i], uold + dt*sum([self.A[i,m] * self.k[m] for m in range(i)]))
        
        unew = uold + dt * sum([self.b[i]     * self.k[i] for i in range(self.stages)])
        ulow = uold + dt * sum([self.b_hat[i] * self.k[i] for i in range(self.stages)])
        
        if j is not None:
            j_up, j_quad_est = self.do_quadrature(t, uold, dt, j, 0)
            return unew, unew - ulow, j_up, j_quad_est

        return unew, unew - ulow

## Implementation of an explicit scheme of orders (2,1)
# \n Heun-Euler method
class Expl_RK_Heun_Euler(Expl_RK):
    order, order_low = 2, 1
    
    A = np.array([[0., 0.],[1., 0.]])
    b = np.array([0.5, 0.5])
    b_hat = np.array([1., 0.])

## Implementation of an explicit scheme of orders (3,2)
# \n Bogacki-Shampine method
class Expl_RK_Bog_Shamp(Expl_RK):
    order, order_low = 3, 2
    
    A = np.array([[0, 0, 0, 0],[1/2, 0, 0, 0],[0, 3/4, 0, 0],[2/9, 1/3, 4/9, 0]])
    b = np.array([2/9, 1/3, 4/9, 0])
    b_hat = np.array([7/24, 1/4, 1/3, 1/8])

## Implementation of an explicit scheme of orders (4,5)
# \n Runge-Kutta Fehlberg method
class Expl_RK_Fehlberg(Expl_RK):
    order, order_low = 5, 4
    
    A = np.array([[0., 0., 0., 0., 0., 0.],
                  [1/4, 0., 0., 0., 0., 0.],
                  [3/32, 9/32, 0., 0., 0., 0.],
                  [1932/2197, -7200/2197, 7296/2197, 0., 0., 0.],
                  [439/216, -8., 3680/513, -845/4104, 0., 0.],
                  [-8/27, 2., -3544/2565, 1859/4104, -11/40, 0.]])
    b = np.array([25/216, 0., 1408/2565, 2197/4104, -1/5, 0.])
    b_hat = np.array([16/135, 0., 6656/12825, 28561/56430, -9/50, 2/55])
        
## Implementation of an explicit scheme of orders (5,4)
# \n Cash-Karp method
class Expl_RK_Cash_Karp(Expl_RK):
    ## orders of method
    order, order_low = 5, 4
    ## A matrix of butcher array
    A = np.array([[0., 0., 0., 0., 0., 0.],
                  [1/5, 0., 0., 0., 0., 0.],
                  [3/40, 9/40, 0., 0., 0., 0.],
                  [3/10, -9/10, 6/5, 0., 0., 0.],
                  [-11/54, 5/2, -70/27, 35/27, 0., 0.],
                  [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096, 0.]])
    ## coefficient vector of main scheme
    b = np.array([37/378, 0., 250/621, 125/594, 0., 512/1771])
    ## coefficient vector of embedded scheme
    b_hat = np.array([2825/27648, 0., 18575/48384, 13525/55296, 277/14336, 1/4])
        
## Implementation of an explicit scheme of orders (5,4)
# \n Dormand-Prince method
class Expl_RK_Do_Pri(Expl_RK):
    order, order_low = 5, 4
    
    A = np.array([[0., 0., 0., 0., 0., 0., 0.],
                  [1/5, 0., 0., 0., 0., 0., 0.],
                  [3/40, 9/40, 0., 0., 0., 0., 0.],
                  [44/45, -56/15, 32/9, 0., 0., 0., 0.],
                  [19372/6561, -25360/2187, 64448/6561, -212/729, 0., 0., 0.],
                  [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0., 0.],
                  [35/384, 0., 500/1113, 125/192, -2187/6784, 11/84, 0.]])
    b = np.array([35/384, 0., 500/1113, 125/192, -2187/6784, 11/84, 0.])
    b_hat = np.array([5179/57600, 0., 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
        
## Implementation of an DIRK scheme of orders (2,1)
# \n Method of Ellsiepen
class Impl_SDIRK2_Ellsiep(Impl_DIRK):
    order, order_low = 2, 1
    
    alpha = 1 - 1/np.sqrt(2)
    alpha_hat = 2 - 5/4*np.sqrt(2)
    
    A = np.array([[alpha, 0],[1 - alpha, alpha]])
    b = np.array([1 - alpha, alpha])
    b_hat = np.array([1 - alpha_hat, alpha_hat])

## Implementation of an DIRK scheme of orders (3,2)
# \n Method of Cash
class Impl_SDIRK3_Cash(Impl_DIRK):
    order, order_low = 3, 2
    
    alpha     = 1.2084966491760101
    beta      = -0.6443631706844691
    gamma     = 0.4358665215084580
    tau_minus = 0.2820667392457705
    alpha_hat = 0.7726301276675511
    beta_hat  = 0.2273698723324489
    
    A = np.array([[gamma, 0, 0], [tau_minus, gamma, 0], [alpha, beta, gamma]])
    b = np.array([alpha, beta, gamma])
    b_hat = np.array([alpha_hat, beta_hat, 0])