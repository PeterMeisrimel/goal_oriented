# -*- coding: utf-8 -*-
"""
Created on Tue Aug 8 16:16:38 2017
@author: Azahar Monge, Peter Meisrimel
"""
from __future__ import division
from schemes_RK import One_step_method
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

## Time integration class for general Thermal FSI problem
class SDIRK2_FSI(One_step_method):
    ## SDIRK 2 paramter
    a = 1 - (1/2)*np.sqrt(2)
    ## SDIRK 2 paramter
    ahat = 2 - (5/4)*np.sqrt(2)
    ## order of embedded scheme
    order_low = 1
    ## general initialization function
    # @param self .
    # @param compute_matrices function for computing FE discretization matrices needed for fixpoint iteration
    # @param t_end t_e
    # @param tol_fixpoint tolerance used in fixpoint iteration 
    # @return None
    def __init__(self, compute_matrices = None, t_end = None, tol_fixpoint = 1e-4):
        ## pointer to function for computing all relevant functions for thermal FSI
        self.compute_matrices = compute_matrices
        ## end time of problem
        self.t_end = t_end
        ## tolerance used for fixpoint iteration
        self.tol_fixpoint = tol_fixpoint
        
        self.n, self.A1, self.A2, self.M1, self.M2, self.A1g, self.A2g, self.Ag1, self.Ag2, \
        self.Agg1, self.Agg2, self.M1g, self.M2g, self.Mg1, self.Mg2, self.Mgg1, self.Mgg2 = self.compute_matrices()
        
## Time integration class for 1D Thermal FSI problem
# /deprecated
class SDIRK2_FSI_1D(SDIRK2_FSI):
    ## One time step of the SDIRK2 method. This contains two DN couplings between subdomains, one for each stage
    # @param self .
    # @param t time
    # @param dt stepsize
    # @param uold current state
    # @param j density function, not None if using time-integration for quadrature
    # @return unew, l_err = unew - ulow
    def do_step(self, t, dt, uold, j = None):
        uold = uold.reshape((len(uold), 1))
        error = np.inf
        u1_start=np.zeros((self.n,1))
        u2_start=np.zeros((self.n,1))
        u1_start = uold[0:self.n]
        ug_start= uold[self.n,0]
        u2_start = uold[self.n+1:2*self.n+1]
        ug_old=ug_start
        
        # Coupling for stage 1
        while error > self.tol_fixpoint: # fixpoit iteration precision
            u1_mid, flux_stage1 = self.solve_dirichlet(self.a*dt,u1_start,ug_start,ug_old)
            u2_mid, ug_mid = self.solve_neumann(self.a*dt,u2_start,ug_start,flux_stage1)
            error = abs(ug_old - ug_mid)
            ug_old = ug_mid
            
        k1_u1 = (1/(self.a*dt))*(u1_mid-u1_start)
        k1_u2 = (1/(self.a*dt))*(u2_mid-u2_start)
        k1_ug = (1/(self.a*dt))*(ug_mid-ug_start)
               
        error = np.inf
        u1_start_stage2 = u1_start + dt*(1-self.a)*k1_u1
        u2_start_stage2 = u2_start + dt*(1-self.a)*k1_u2
        ug_start_stage2 = ug_start + dt*(1-self.a)*k1_ug
        ug_old=ug_start_stage2
        # Coupling for stage 2
        while error > self.tol_fixpoint: # fixpoit iteration precision
            u1, flux_stage2 = self.solve_dirichlet(self.a*dt,u1_start_stage2,ug_start_stage2,ug_old)
            u2, ug_new = self.solve_neumann(self.a*dt,u2_start_stage2,ug_start_stage2,flux_stage2)
            error = abs(ug_old - ug_new)
            ug_old = ug_new
                   
        k2_u1 = (1/(self.a*dt))*(u1-u1_start_stage2)
        k2_u2 = (1/(self.a*dt))*(u2-u2_start_stage2)
        k2_ug = (1/(self.a*dt))*(ug_new-ug_start_stage2)
        # New solutions for u1, u2 and ug
        u1 = u1_start + dt*((1-self.a)*k1_u1+self.a*k2_u1)
        u2 = u2_start + dt*((1-self.a)*k1_u2+self.a*k2_u2)
        ug_new = ug_start + dt*((1-self.a)*k1_ug+self.a*k2_ug)
        # Correction solutions for u1hat, u2hat and ughat
        u1hat = u1_start + dt*((1-self.ahat)*k1_u1+self.ahat*k2_u1)
        u2hat = u2_start + dt*((1-self.ahat)*k1_u2+self.ahat*k2_u2)
        ughat = ug_start + dt*((1-self.ahat)*k1_ug+self.ahat*k2_ug)
        
        u = np.zeros((2*self.n+1,1))
        u[0:self.n] = u1
        u[self.n] = ug_new
        u[self.n+1:2*self.n+1] = u2
        
        localu = np.zeros((2*self.n+1,1))
        localu[0:self.n] = u1-u1hat
        localu[self.n] = ug_new-ughat
        localu[self.n + 1:2 * self.n + 1] = u2-u2hat
        
        if j is not None:
            k1 = np.concatenate([self.a*dt*k1_u1, self.a*dt*k1_ug, self.a*dt*k1_u2])
            k2 = np.concatenate([self.a*dt*k2_u1, self.a*dt*k2_ug, self.a*dt*k2_u2])
            up_1 = j(t + self.a*dt, uold + dt*self.a*k1)
            up_2 = j(t + dt       , uold + dt*((1-self.a)*k1 + self.a*k2))
            
            j_update = dt*((1-self.a)*up_1 + self.a*up_2)
            j_quad_est = dt*abs((self.ahat - self.a)*up_1 + (self.a - self.ahat)*up_2)
            return u.reshape(len(u)), localu.reshape(len(localu)), dt*j_update, dt*j_quad_est
        
        return u.reshape(len(u)), localu.reshape(len(localu))
        
    ## function for computing flux over interface as needed for dirichlet function
    # @param self .
    # @param dt current timestepsize
    # @param u0 u on region from previous timestep?
    # @param u u on region from current timestep?
    # @param ug_0 u on interfacce from current timestep???
    # @param ug_old u on interface from previous timestep?
    # @return flux over interface
    def compute_flux(self, dt, u0, u, ug_0, ug_old):
        return -np.dot(((1/(dt))*self.Mg1 + self.Ag1),u) - np.dot(((1/(dt))*self.Mgg1 + self.Agg1),ug_old) + \
                (1/(dt))*np.dot(self.Mg1,u0) + (1/(dt))*np.dot(self.Mgg1,ug_0)
    
    ## Solve Domain 1 with Dirichlet boundary conditions at the interface
    # @param self .
    # @param dt current timestep
    # @param u0 u on region from previous timestep?
    # @param ug_0 u on interfacce from current timestep???
    # @param ug_old u on interface from previous timestep?
    # @return new solution u, flux over interface
    def solve_dirichlet(self, dt, u0, ug_0, ug_old):  
        u = np.linalg.solve(((1/dt)*self.M1+self.A1),(-np.dot((1/dt)*self.M1g+self.A1g,ug_old) + \
            (1/dt)*np.dot(self.M1,u0) + (1/dt)*np.dot(self.M1g,ug_0)))
        flux = self.compute_flux(dt, u0, u, ug_0, ug_old)
        return u, flux
    
    ## Solve Domain 2 with Neumann boundary conditions at the interface
    # @param self .
    # @param dt current timestep
    # @param u0 u on region ?
    # @param ug_0 u on interface ?
    # @param flux flux over interface
    # @return new solution on region and interface
    def solve_neumann(self, dt, u0, ug_0, flux): 
        B = np.zeros((self.n+1,self.n+1))
        B[0:self.n,0:self.n] = (1/(dt))*self.M2+self.A2
        B[self.n,0:self.n] = (1/(dt))*self.Mg2 + self.Ag2
        B[0:self.n,self.n] = (1/(dt))*self.M2g[:,0] + self.A2g[:,0]
        B[self.n,self.n] = (1/(dt))*self.Mgg2 + self.Agg2
    
        b = np.zeros((self.n+1,1))
        b[0:self.n,0] = (1/(dt))*np.dot(self.M2,u0)[:,0] + (1/(dt))*np.dot(self.M2g,ug_0)[:,0]
        b[self.n,0] = flux + (1/(dt))*np.dot(self.Mg2,u0) + (1/(dt))*np.dot(self.Mgg2,ug_0)
        u = np.linalg.solve(B,b)
        u_new = np.zeros((self.n,1))
        for i in range(self.n):
            u_new[i,0] = u[i,0]
        ug_new = u[self.n,0]
        return u_new, ug_new

## Time integration class for 2D Thermal FSI problem
class SDIRK2_FSI_2D(SDIRK2_FSI):
    ## general initialization function
    # @param self .
    # @param compute_matrices function for computing FE discretization matrices needed for fixpoint iteration
    # @param t_end t_e
    # @param tol_fixpoint tolerance used in fixpoint iteration 
    # @return None
    def __init__(self, compute_matrices = None, t_end = None, tol_fixpoint = 1e-4):
        SDIRK2_FSI.__init__(self, compute_matrices = compute_matrices, t_end = t_end, tol_fixpoint = tol_fixpoint)
        ## B Matrix for Neumann consists of 2 parts, one depedent, the other one independent of the timestep
        # python requires csr/csc format for lin equation solving, which are horrible for assigning values
        # pythons lil format is good for assigning values, but constant format conversion is costly as well
        # Thus the matrix is split into parts according to time dependence
        # change in timesteps are dealt with via multiplying the variable part by factors
        n = self.n
        self.B_fixed = sp.lil_matrix((n**2+n,n**2+n))
        self.B_fixed[0:n**2,0:n**2] = self.A2
        self.B_fixed[n**2:n**2+n,0:n**2] = self.Ag2
        self.B_fixed[0:n**2,n**2:n**2+n] = self.A2g[:,0:n]
        self.B_fixed[n**2:n**2+n,n**2:n**2+n] = self.Agg2
        
        self.B_var = sp.lil_matrix((n**2+n,n**2+n))
        self.B_var[0:n**2,0:n**2] = self.M2
        self.B_var[n**2:n**2+n,0:n**2] = self.Mg2
        self.B_var[0:n**2,n**2:n**2+n] = self.M2g[:,0:n]
        self.B_var[n**2:n**2+n,n**2:n**2+n] = self.Mgg2
        
        self.B_var = self.B_var.tocsr()
        self.B_fixed = self.B_fixed.tocsr()
        
        ## Matrix for calculating flux over boundary        
        self.E1 = sp.spdiags(np.ones(n**2), n**2 - n, n, n**2, format = 'csr')
        ## splitting used in time-integration
        self.splitting = [self.n**2, self.n**2 + self.n]
        
    ## Do single fixpoint for a given SDIRK stage
    # @param self .
    # @param adt a * dt
    # @param uold state vector
    # @return solution 
    def do_SDIRK_step_DN(self, adt, uold):
        uold1, uoldg, uold2 = np.array_split(uold, self.splitting)
        error = np.inf
        uoldg_save = np.copy(uoldg)
        
        while error > self.tol_fixpoint:
            unew1, flux = self.solve_dirichlet(adt, uold1, uoldg_save, uoldg)
            unew2, unewg = self.solve_neumann(adt, uold2, uoldg_save, flux)
            error = np.linalg.norm(uoldg - unewg, 2)
            uoldg = unewg
        
        return np.concatenate([unew1, unewg, unew2])

    ## One time step of the SDIRK2 method. This contains two DN couplings between subdomains, one for each stage
    # @param self .
    # @param t time
    # @param dt stepsize
    # @param uold state vector
    # @param j density function, not None if using time-integration for quadrature
    # @return unew, local error estimate
    def do_step(self, t, dt, uold, j = None):
        # see article "A time-adaptive fluid-structure interaction method for thermal coupling" by Philipp Birken, pp.334
        umid = self.do_SDIRK_step_DN(self.a*dt, uold)
        k1 = (umid - uold)/(self.a*dt)

        u_start_stage2 = uold + dt*(1-self.a)*k1
        
        unew = self.do_SDIRK_step_DN(self.a*dt, u_start_stage2)
        k2 = (unew - u_start_stage2)/(self.a*dt)

        # solution
        #unew = uold + dt * ((1 - self.a)    * k1 + self.a    * k2) # already calculated
        ulow = uold + dt * ((1 - self.ahat) * k1 + self.ahat * k2)

        if j is not None:
            jk1 = j(t + self.a*dt, uold + dt * self.a * k1)
            jk2 = j(t + dt       , unew)
            
            j_update = (1-self.a)*jk1 + self.a*jk2
            j_quad_est = abs((self.ahat - self.a)*jk1 + (self.a - self.ahat)*jk2)
            return unew, unew - ulow, dt*j_update, dt*j_quad_est
        return unew, unew - ulow
    
    ## function for computing flux over interface as needed for dirichlet function
    # @param self .
    # @param dt current timestepsize
    # @param u0 u on region from previous timestep?
    # @param u u on region from current timestep?
    # @param ug_0 u on interfacce from current timestep???
    # @param ug_old u on interface from previous timestep?
    # @return flux over interface
    def compute_flux(self, dt, u0, u, ug_0, ug_old):
        return -((1/dt)*self.Mg1 + self.Ag1).dot(u) - ((1/dt)*self.Mgg1 + self.Agg1).dot(ug_old) + \
               (1/dt)*self.Mg1.dot(u0) + (1/dt)*self.Mgg1.dot(ug_0)

    ## Solve Domain 1 with Dirichlet boundary conditions at the interface
    # @param self .
    # @param dt current timestep
    # @param u0 u on region from previous timestep?
    # @param ug_0 u on interfacce from current timestep???
    # @param ug_old u on interface from previous timestep?
    # @return new solution u, flux over interface
    def solve_dirichlet(self, dt, u0, ug_0, ug_old):
        u = spsolve(((1/dt)*self.M1+self.A1) ,(-((1/dt)*self.M1g+self.A1g).dot(ug_old) + \
            (1/dt)*self.M1.dot(u0) + (1/dt)*self.M1g.dot(ug_0)))
        flux = self.compute_flux(dt, u0, u, ug_0, ug_old)
        return u, flux

    ## Solve Domain 2 with Neumann boundary conditions at the interface
    # @param self .
    # @param dt current timestep
    # @param u0 u on region ?
    # @param ug_0 u on interface ?
    # @param flux flux over interface
    # @return new solution on region and interface
    def solve_neumann(self, dt, u0, ug_0, flux):
        n = self.n
        
        b = np.zeros(n**2+n)
        b[0:n**2] = (1/dt)*self.M2.dot(u0) + (1/dt)*self.M2g.dot(ug_0)
        b[n**2:n**2+n] = flux + (1/dt)*self.Mg2.dot(u0) + (1/dt)*self.Mgg2.dot(ug_0)
        
        # see __init__ for implementation of B Matrix
        u = spsolve((1/dt)*self.B_var + self.B_fixed, b)

        u_new = u[0:n**2]
        ug_new = u[n**2:]
        return u_new, ug_new