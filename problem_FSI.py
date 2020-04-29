# -*- coding: utf-8 -*-
"""
Created on Tue Aug 8 15:48:31 2017
@author: Azahar Monge, Peter Meisrimel
"""
from __future__ import division
from problem import Problem
import numpy as np
import scipy.sparse as sp

## Class for fluid structure interaction problem for 2 coupled linear heat equations
class Problem_FSI(Problem):
    ## Initialization basics
    # @param self .
    # @param gridsize gridsize for spatial discretization
    # @param t_end t_e
    # @param D1 heat equation related parameter, same in licentiate thesis/paper
    # @param D2 heat equation related parameter
    # @param K1 heat equation related parameter, lambda in licentiate thesis/paper
    # @param K2 heat equation related parameter
    # @param flux which subdomain to base flux calculation on
    # @param u0 initial condition
    # @return None
    def __init__(self, gridsize = 50, t_end = None, D1 = 0.1, D2 = 1, K1 = 0.1, K2 = 1, flux = 1, u0 = 0):
        self.n = gridsize
        self.dx = 1/(self.n+1)
        self.t_end = t_end
        self.K1, self.K2 = K1, K2
        self.D1, self.D2 = D1, D2
        self.d1 = self.K1/self.D1 # alpha in licentiate thesis/paper
        self.d2 = self.K2/self.D2
        self.u0_type = u0
        
        self.t_end = 0.5 if t_end is None else t_end
        
        self.gridsize = gridsize
        
        if flux == 1:    self.flux = self.compute_flux1
        elif flux == 2:  self.flux = self.compute_flux2
        else:            raise ValueError('invalid flux selected')
        
    ## Density function
    # @param self .
    # @param t time
    # @param u input vector
    # @return j(t, u)
    def j(self, t, u):
        return self.flux(u)

## Class for fluid structure interaction problem for 2 coupled linear heat equations in 1 dimension
# \n problem_type = 20
class Problem_FSI_1D(Problem_FSI):
    ## Initialization basics
    # @param self .
    # @param gridsize gridsize for spatial discretization
    # @param t_end end time of problem
    # @param D1 heat equation related parameter
    # @param D2 heat equation related parameter
    # @param K1 heat equation related parameter
    # @param K2 heat equation related parameter
    # @param flux which subdomain to base flux calculation on
    # @param u0 initial condition
    # @return None
    def __init__(self, gridsize = 50, t_end = None, D1 = 0.1, D2 = 1, K1 = 0.1, K2 = 1, flux = 1, u0 = 0):
        Problem_FSI.__init__(self, gridsize = gridsize, t_end = t_end, D1 = D1, D2 = D2, K1 = K1, K2 = K2, flux = flux,
                             u0 = u0)
        
        xplot = np.linspace(0, 2, 2 * self.n + 3)
        u0 = -xplot[1:-1]**2 + 2*xplot[1:-1]
        
        self.u0 = np.zeros((2*self.n+1,1))
        for i in range(2*self.n+1):
            self.u0[i,0] = u0[i]
        self.u0 = self.u0.reshape((len(self.y0), 1))
    
    ## function to compute FE Discretization matrices
    # @param self .
    # @return self.n and tons of matrices
    def compute_matrices(self): 
        A1g=np.zeros((self.n,1))
        A2g=np.zeros((self.n,1))
        A1g[self.n-1,0] = -(self.K1/(self.dx**2))
        A2g[0,0] = -(self.K2/(self.dx**2))
        
        Ag1 = np.zeros((1,self.n))
        Ag2 = np.zeros((1,self.n))
        Ag1[0,self.n-1] = -self.K1/(self.dx**2)
        Ag2[0,0] = -self.K2/(self.dx**2)
    
        Agg1 = self.K1/(self.dx**2)
        Agg2 = self.K2/(self.dx**2)
    
        M1g = np.zeros((self.n,1))
        M2g = np.zeros((self.n,1))
        M1g[self.n-1,0] = self.d1/6
        M2g[0,0] = self.d2/6
    
        Mg1 = np.zeros((1,self.n))
        Mg2 = np.zeros((1,self.n))
        Mg1[0,self.n-1] = self.d1/6
        Mg2[0,0] = self.d2/6
    
        Mgg1 = self.d1/3
        Mgg2 = self.d2/3
        
        A1 = np.diag(self.n*[2*self.K1/(self.dx**2)]) + np.diag((self.n-1)*[-(self.K1/(self.dx**2))],-1) + np.diag((self.n-1)*[-(self.K1/(self.dx**2))],1)
        A2 = np.diag(self.n*[2*self.K2/(self.dx**2)]) + np.diag((self.n-1)*[-(self.K2/(self.dx**2))],-1) + np.diag((self.n-1)*[-(self.K2/(self.dx**2))],1)
        
        M1 = np.diag(self.n*[4*self.d1/6]) + np.diag((self.n-1)*[self.d1/6],-1) + np.diag((self.n-1)*[self.d1/6],1)
        M2 = np.diag(self.n*[4*self.d2/6]) + np.diag((self.n-1)*[self.d2/6],-1) + np.diag((self.n-1)*[self.d2/6],1)
    
        return self.n, A1, A2, M1, M2, A1g, A2g, Ag1, Ag2, Agg1, Agg2, M1g, M2g, Mg1, Mg2, Mgg1, Mgg2
    
    ## computation of flux over interface based on evaluation in first region
    # @param self .
    # @param u current state
    # @return flux over boundary calculated from left side
    def compute_flux1(self, u):
        return -(self.K1/self.dx)*(u[self.n]-u[self.n-1])

    ## computation of flux over interface based on evaluation in second region
    # @param self .
    # @param u current state
    # @return flux over boundary calculated from right side
    def compute_flux2(self, u):
        return (self.K2/self.dx)*(u[self.n+1]-u[self.n]) 
    
    ## Reference solution to functional for a givne density function
    # @param self .
    # @return Reference solution
    def solution_int(self):
        print('RuntimeWarining: No reference solution available')
        return 0
            
## Class for fluid structure interaction problem for 2 coupled linear heat equations in 2 dimensions
# \n problem_type = 21
class Problem_FSI_2D(Problem_FSI):
    ## Initialization basics
    # @param self .
    # @param gridsize gridsize for spatial discretization
    # @param t_end end time of problem
    # @param D1 heat equation related parameter
    # @param D2 heat equation related parameter
    # @param K1 heat equation related parameter
    # @param K2 heat equation related parameter
    # @param flux which subdomain to base flux calculation on
    # @param u0 initial condition
    # @return None
    def __init__(self, gridsize = 50, t_end = None, D1 = 0.1, D2 = 1, K1 = 0.1, K2 = 1, flux = 1, u0 = 0):
        Problem_FSI.__init__(self, gridsize = gridsize, t_end = t_end, D1 = D1, D2 = D2, K1 = K1, K2 = K2, flux = flux,
                             u0 = u0)
                             
        if u0 == 7:
            def initial_condition(x,y):
                if x <= 0.5:
                    return 800*(np.sin(np.pi*y)**2)*(np.sin(2*np.pi*x))
                if 0.5 < x < 1.5:
                    return 200*(np.sin(np.pi*y)**2)*(np.sin(np.pi*(x-0.5)))
                return 0
        else:
            raise ValueError('invalid initial condition')
            
        self.u0_type = u0
    
        yy = np.linspace(0, 1, self.n+2)
        xx = np.linspace(0, 2, 2*self.n+3)
        
        # only initialize the inner points
        uu = np.zeros((2*self.n+1, self.n))
        for i in range(2*self.n+1):
            for j in range(self.n):
                uu[i,j] = initial_condition(xx[i+1], yy[j+1])
        self.u0 = uu.reshape(self.n*(2*self.n+1))
            
    ## function to compute FE Discretization matrices
    # @param self .
    # @return self.n and tons of matrices
    def compute_matrices(self):
        n, dx = self.n, self.dx
        A1g = sp.spdiags(-(self.K1/(dx**2))*np.ones(n**2), -n**2 + n, n**2, n, format = 'csr')
        A2g = sp.spdiags(-(self.K2/(dx**2))*np.ones(n**2), 0        , n**2, n, format = 'csr')
        
        Ag1 = sp.spdiags(-(self.K1/(dx**2))*np.ones(n**2), n**2 - n , n, n**2, format = 'csr')
        Ag2 = sp.spdiags(-(self.K2/(dx**2))*np.ones(n**2), 0        , n, n**2, format = 'csr')
    
        Agg1 = sp.spdiags([2*self.K1/(dx**2)*np.ones(n)] + 2*[-0.5 * self.K1/(dx**2)*np.ones(n)], [0, 1, -1], n, n, format = 'csr')
        Agg2 = sp.spdiags([2*self.K2/(dx**2)*np.ones(n)] + 2*[-0.5 * self.K2/(dx**2)*np.ones(n)], [0, 1, -1], n, n, format = 'csr')

        M1g = sp.spdiags([-self.d1/12*np.ones(n**2), self.d1/4 *np.ones(n**2)], [-n**2 + n, -n**2 + n - 1], n**2, n, format='csr')
        M2g = sp.spdiags([-self.d2/12*np.ones(n**2), self.d2/4 *np.ones(n**2)], [0, 1], n**2, n, format='csr')
        
        Mg1 = sp.spdiags([-self.d1/12*np.ones(n**2), [0]*(n**2 - n) + [self.d1/4]*n], [n**2 - n, n**2 - n - 1] , n, n**2, format='csr')
        Mg2 = sp.spdiags([-self.d2/12*np.ones(n), self.d2/4 *np.ones(n)], [0, 1], n, n**2, format='csr')
              
        Mgg1 = sp.spdiags([5*self.d1/12*np.ones(n)] + 2*[-self.d1/24*np.ones(n)], [0, 1, -1], n, n, format = 'csr')
        Mgg2 = sp.spdiags([5*self.d2/12*np.ones(n)] + 2*[-self.d2/24*np.ones(n)], [0, 1, -1], n, n, format = 'csr')

        B = sp.spdiags(4*np.ones(n),  0, n, n, format = 'csr') - \
            sp.spdiags(  np.ones((2,n)),  [-1, 1], n, n, format = 'csr') 
        A1 = sp.kron(sp.spdiags(np.ones(n), 0, n, n, format = 'csr'), B) + \
             sp.kron(sp.spdiags(np.ones((2,n)), [-1, 1], n, n, format = 'csr'), sp.spdiags(-np.ones(n), 0, n, n, format = 'csr'))
        A2 = self.K2/(dx**2)*A1
        A1 = self.K1/(dx**2)*A1
        
        N = sp.spdiags([5/6*np.ones(n)] + 2*[-1/12*np.ones(n)], [0, 1, -1], n, n, format = 'csr')
        N1 = sp.spdiags([-1/12*np.ones(n), 1/4*np.ones(n)], [0, -1], n, n, format = 'csr')
        N2 = sp.spdiags([-1/12*np.ones(n), 1/4*np.ones(n)], [0, 1], n, n, format = 'csr')
        M1 = sp.kron(sp.spdiags(np.ones(n), 0, n, n, format = 'csr'), N) + \
             sp.kron(sp.spdiags(np.ones(n), -1, n, n, format = 'csr'), N1) + \
             sp.kron(sp.spdiags(np.ones(n), 1, n, n, format = 'csr'), N2)
        M2 = self.d2*M1
        M1 = self.d1*M1
        return n, A1, A2, M1, M2, A1g, A2g, Ag1, Ag2, Agg1, Agg2, M1g, M2g, Mg1, Mg2, Mgg1, Mgg2

    ## computation of flux over interface based on evaluation in first region
    # @param self .
    # @param u current state
    # @return flux over boundary calculated from left side
    def compute_flux1(self, u):
        n = self.n
        ug = u[n*n:n*n+n]
        uaux, ugaux = np.zeros(n), np.zeros(n)
        uaux[1:n] = u[n**2 - n:n**2-1]
        ugaux[1:n] = ug[0:n-1]
        vecflux = 2*ug - uaux - ugaux
        return -self.K1*sum(vecflux)
    
    ## computation of flux over interface based on evaluation in second region
    # @param self .
    # @param u current state
    # @return flux over boundary calculated from right side
    def compute_flux2(self, u):
        n = self.n
        ug= u[n*n:n*n+n]
        uaux = np.zeros(n)
        ugaux = np.zeros(n)
        uaux[0:n-1] = u[n**2+n+1:n**2+2*n]
        ugaux[0:n-1] = ug[1:n]
        vecflux = -2*ug + uaux + ugaux
        return self.K2*sum(vecflux)
    
    ## Reference solution to functional for a givne density function
    # @param self .
    # @return Reference solution
    def solution_int(self):
        if self.K1 == 0.01 and self.K2 == 1 and self.D1 == 0.1 and self.D2 == 1:
            if self.t_end == 0.2 and self.gridsize == 20 and self.u0_type == 7:
                #return 0.76530963637800319965265316568547859787940979003906 # tol = 1e-9, fp tol 1e-12
                return 0.76530963594362910740187544433865696191787719726562 # tol = 1e-9, fp tol = 1e-12, dt0 = tol
            if self.t_end == 0.2 and self.gridsize == 40 and self.u0_type == 7:
                return 0.87235410604410879020775837489054538309574127197266 # tol = 1e-8, fp tol = 1e-12, dt0 = tol
            if self.t_end == 0.2 and self.gridsize == 60 and self.u0_type == 7:
                return 0.91029813193594899534133446650230325758457183837891 # tol = 1e-7, fp tol = 1e-12, dt0 = tol
            if self.t_end == 0.2 and self.gridsize == 80 and self.u0_type == 7:
                return 0.92973107351472783488333107015932910144329071044922 # tol = 1e-6, fp tol = 1e-12, dt0 = tol
        print('RuntimeWarining: No reference solution available')
        return 0