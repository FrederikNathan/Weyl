#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 08:51:26 2020

@author: frederik

Module for solution of master equation in time domain

History of module:
v1: speedup computing of steady states and fourier transform. 
v4: with linear interpolation in iteration to reduce correction from O(dt) to O(dt^2)
v9: using SO(3) representation. Using rotating frame interpolator

In case of commensurate frequencies, we average over phase. 
"""
 
T_RELAX          = 12 # time-interval used for relaxing to steady state, in units of tau.
                                      # i.e. relative uncertainty of steady state = e^{-STEADY_STATE_RELATIVE_TMAX}
NMAT_MAX         = 100
T_RES            = 300   # time resolution that enters. 
# print("WARNING - SET T_RES BACK TO 1000 BEFORE USING")
CACHE_ELEMENTS   = 1e6  # Number of entries in cached quantities
PHASE_RESOLUTION = 1500  # Phase resolution of solver 
N_RECORD         = 1000000
# N_CONTOURS       = 200 # NUMBER Of contours in the phase brillouin zone 

import os 
from scipy import *
import sys 
from scipy.interpolate import griddata,LinearNDInterpolator

from units import *
import weyl_liouvillian as wl
import so3 as so3

I3 = eye(3)

# Levi civita tensor
Generator = zeros((3,3,3))
Generator[0,1,2],Generator[0,2,1]=1,-1
Generator[1,2,0],Generator[1,0,2]=1,-1
Generator[2,0,1],Generator[2,1,0]=1,-1
    
[SX,SY,SZ,I2] = [B.SX,B.SY,B.SZ,B.I2]
[sx,sy,sz,i2] = [q.flatten() for q in [SX,SY,SZ,I2]]

ZM = zeros((4,4),dtype=complex)

# Indices in vectorized matrix space, where \rho may be nonzero. (we restrict ourselves to this subspace)
Ind = array([5,6,9,10])



class time_domain_solver():
    """
    Core time-domain solver object
    Takes as input a k-point and parameter set
    
    if save_evolution is specified )As string), evolution is saved at filename specified by the string. 
    """
    def __init__(self,k,parameters,integration_time,evolution_file=None,phase_resolution = PHASE_RESOLUTION):
        self.k = k
        self.parameters = parameters
        self.integration_time = integration_time
        
        assert (evolution_file is None) or type(evolution_file)==str,"save_evolution must be None or str"
        if type(evolution_file)==str:
            
            self.save_evolution = True
            self.evolution_file = evolution_file
        else:
            self.save_evolution = False
            self.evolution_file = ""
        
        # Set parameters in weyl liouvillian module
        wl.set_parameters(parameters)

        # Unpack parameters
        [self.omega1,self.omega2,self.tau,self.vF,self.V0x,self.V0y,self.V0z,self.EF1,self.EF2,self.Mu,self.Temp]=parameters
        
        # Variables derived from parameters
        self.T1 = 2*pi/self.omega1
        self.T2 = 2*pi/self.omega2
        self.V0 = array([self.V0x,self.V0y,self.V0z])
        self.P0 = self.omega1*self.omega2/(2*pi)
        self.A1 = self.EF1/self.omega1
        self.A2 = self.EF2/self.omega2
        self.frequency_ratio = self.omega2/self.omega1
        
        # Get time-integration parameters
        self.t_relax = self.get_t_relax()  # time-interval used for relaxing to steady state
        
        self.is_commensurate,self.frequency_fraction,self.ext_period = self.check_commensurability()
        self.contour_length_index,self.n_contours = self.get_contour_parameters()
        self.contour_length = self.contour_length_index * self.T1 
        self.contour_initializations = self.get_initial_contour_locations()
        self.n_par,self.runs_per_contour = self.find_optimal_parallelization()
        self.tmax = self.contour_length/self.runs_per_contour
        self.phi1_0,self.phi2_0 = self.get_initial_phases()
        self.res = self.get_res()
        self.dt  = self.T1/self.res 
        self.tau_factor = 1-exp(-self.dt/self.tau)
        self.tau_factor_2 = 0.5*(1-exp(-self.dt/self.tau))
        self.tau_exp = exp(-self.dt/self.tau)
      
        # Initialize running variables
        self.ns  = 0
        self.rho = None
        self.t   = None
        self.theta_1 = None
        self.theta_2 = None
        self.n_record = 0 
        
        self.N_cache = max(1,int(CACHE_ELEMENTS/NMAT_MAX))   # number of steps to cache at a time
        self.N_record =  int(N_RECORD/self.n_par + 1 )


        self.n_array   = zeros((phase_resolution,phase_resolution),dtype = int)
        self.phase_mat = zeros((phase_resolution,phase_resolution,3),dtype=float)
        self.phi1_list = zeros((self.N_record,self.n_par),dtype = float)
        self.phi2_list = zeros((self.N_record,self.n_par),dtype = float)
        self.rho_list = zeros((self.N_record,self.n_par,3),dtype = float)
        
        self.interpolation_points =  arange(0.5,PHASE_RESOLUTION+0.5)/PHASE_RESOLUTION * 2*pi
        self.bin_edges            =  arange(PHASE_RESOLUTION+1)/PHASE_RESOLUTION * 2*pi
        self.interpolation_grid   =  tuple(meshgrid(self.interpolation_points,self.interpolation_points))
        
        self.rho_mat = None 
        
        self.solved = False
    def get_t_relax(self):
        a = T_RELAX * self.tau
        b = a/self.T1 
         
        out = int(b+0.9)*self.T1 
        
        return out 
    
    def check_commensurability(self):
        """
        Check commensurability of frequencies
        
        Is determined to be commensurate if the phases of the two modes 
        coincide with periodicitiy up to 1.2 * integration_time
        
        If commensurate, returns 
        
        frequency_fraction = (q,p), 
        
        where q/p = \omega_1/\omega_2, and
        
        extended_period = q*T1 = p*T2 
        
        Returns
        -------
        is_commensurate : bool
            flags whether the frequencies are commensurate.
        frequency_fraction : (p,q), where p,q are integers
            Fraction of frequencies such that q*T1 = p*T2.
            I.e. \omega_1/\omega_2 = q/p
        ext_period : TYPE
            DESCRIPTION.

        """
        self.r = arange(1,self.integration_time*1.2/self.T1)*self.T1/self.T2
        self.r = mod(self.r+0.5,1)-0.5 
        
        vec = where(abs(self.r)<1e-10)[0]+1
        
        if len(vec)>0:
  
            q = amin(vec)
            p = int((q*self.T1)/self.T2 +0.5)
            
            frequency_fraction = (q,p)
            
            ext_period = q*self.T1
        
            is_commensurate = True
            
        
        else:
            
            is_commensurate,frequency_fraction,ext_period =  False,None,None
        
        return is_commensurate,frequency_fraction,ext_period
    
    
    def get_contour_parameters(self):
        """
        Finds optimal integration window  in periods of mode 1, 
        
        T_opt = n_opt * T1, 
        
        such that 
        
        \omega_1*T_opt %2pi = 0
        \omega_2*T_opt %2pi \approx 0
        
        This makes time-integration almost periodic (exactly periodic if ratio
        is commensurate)
        
        if frequencies are commensurate, n_opt = q, where qT_1 = pT_2

        Returns
        -------
        contour_length_index : int
            n_opt
        n_contours : int
            Number of parallel contours used in time integration.
        """
        
        if self.is_commensurate:
            
            contour_length_index = self.frequency_fraction[0]
        
        else :
        
            x = int(self.integration_time/self.T1*1.2)
            # First determine tmax 
            
            n_list   = arange(1,x+1)
            phase_difference   = mod(n_list*self.T1+0.5*self.T2,self.T2)-0.5*self.T2
            cost               = phase_difference/(n_list**0.9)
            
            i0 = argmin(abs(cost))
        
            contour_length_index =  n_list[i0]
            
            
        # Estimate desired number of countours as float
        X = self.integration_time /(contour_length_index * self.T1)

        # Convert estimate to integer
        n_contours = max(1,int(0.5+X))
        
        # Make sure number of contours does not exceed maximal number of parallel runs
        if n_contours >= NMAT_MAX:
            n_contours = NMAT_MAX
            
        return contour_length_index,n_contours
    
    
    
    def cost_function(self,n_par):
        """
        Determine numerical cost of dividing integration into n_par parallel runs

        Parameters
        ----------
        n_par : ndarray(D), int
            candidate values of n_par to investigate

        Returns
        -------
        cost : ndarray(D), float
            cost. cost[z] gives the cost of dividing time-integration into 
            n_par[z] parallel runs

        """
        
        cost = self.t_relax * sqrt(100**2+n_par**2) + self.contour_length*self.n_contours/n_par* sqrt(100**2+n_par**2)
        
        return cost 

    def find_optimal_parallelization(self):
        """
        Find optimal number of parallel runs.
        There is the same number of parallel runs per contour. 
        Number of parallel runs is a multiple of number of contours. 
        There is runs_per_contour parallel runs per contour such that
        
        n_par = n_contours * n_runs_per_contours
        
        1\leq n_par \leq N_MAT_MAX
        
        
        Returns
        -------
        n_par : int
            Number of parallel runs in total.
        runs_per_contour : int
            number of parallel runs per contour.
            
            n_par = n_contours * n_runs_per_contours

        """
        
        n_par_candidates= arange(1,max(2,NMAT_MAX//self.n_contours))*self.n_contours
        cost = self.cost_function(n_par_candidates)
        
        i0 = argmin(cost)
        n_par = n_par_candidates[i0]
        
        runs_per_contour = n_par//self.n_contours

        return n_par,runs_per_contour
        

    def get_initial_contour_locations(self):
        """
        Generate intial location of time-integrations
        Each single contour intersects line \phi_1=0 contour_length_index
        number of times
        
        Intersections are evenly spaced, since contour is (almost) periodic
        So optimal distribution of contours is evenly spaced within the 
        interval [0,2*pi/contour_lenght_index]
        
        We randomize initial positions along the contourto avoid systematic errors 
        from initialization

        Returns
        -------
        phi1 : ndarray(n_contours)
            phi1[z] gives the initial phi1-location of contour z.
        phi2 : ndarray(n_contours)
            phi2[z] gives the initial phi2-location of contour z.

        """
       
        
        # Find intersections of contours on the line phi1 = 0
        phi20 = arange(0,self.n_contours)/self.n_contours*2*pi/self.contour_length_index
        
        # Find initial position by random location along contours
        npr.seed(0)
        
        x=npr.rand(self.n_contours)
                
        phi1 = 2*pi*x
        phi2 = phi20 + mod(self.omega2/self.omega1 * phi1,2*pi)
        
        return phi1,phi2
    
    
        
    def get_initial_phases(self):
        """
        Get initial locations of parallel runs in phase BZ 
        
        The initalizations are evenly spaced on each contour, with runs_per_contour
        initial locations per contour

        Returns
        -------
        phi10_out : ndarray(n_par),float, in [0,2*pi]
            phi1 locations of initializaions.
        phi20_out : ndarray(n_par),float, in [0,2*pi]
            phi2 locations of initializaions.

        """
        phi1list = []
        phi2list = []
        
        for nc in range(0,self.n_contours):
            phi10,phi20 = self.contour_initializations[0][nc],self.contour_initializations[1][nc]
        
            for nr in range(0,self.runs_per_contour):
                dphi1 = 2*pi*nr*self.contour_length_index/self.runs_per_contour
                phi1 = phi10  + dphi1
                phi2 = phi20  + dphi1 * self.omega2/self.omega1
                
                phi1list.append(phi1%(2*pi))
                phi2list.append(phi2%(2*pi))
                    
        
        phi10_out = array(phi1list)
        phi20_out = array(phi2list)
        
        return phi10_out,phi20_out


        
    def get_res(self):
        """
        Get resolution of driving.

        Returns
        -------
        res, int
            Drive resolution.  The time step in the simulation is set to
            dt = T1 /resoluion..

        """
        
        dt0 = amin(abs(array([self.T1/T_RES,self.T2/T_RES,self.tau/T_RES])))*sqrt(0.5)
        res = int(self.T1/dt0)+1
        
        return res
    
        
        
    def generate_cache(self):
        """ 
        t_cachce[nt,z] gives nt-th time step of realization z in the cache. nt runs from 0 to (including) self.N_cache
        t_cache[nt,z] = t[z] + nt*self.dt
        
        self.k_cache[nt,z] gives the self.k+A(t_cache[nt,z])
        self.hvec_cache[nt,z] gives the hvec(self.k_cache[nt,z])
    
        """
        B0 = B.tic()
 
    
        # npr.seed(int(t))
        
        dt_cache = rand(self.N_cache+1)
        dt_cache = cumsum(dt_cache)
        dt_cache = dt_cache * self.N_cache*self.dt/amax(dt_cache)
        # self.t_cache = self.t + arange(self.N_cache+1)*self.dt
        self.t_cache = self.t + dt_cache 
        self.dt_cache = self.t_cache[1:]-self.t_cache[:-1]
        self.phi1_cache = self.phi1_0.reshape(1,self.n_par) + self.t_cache.reshape((self.N_cache+1,1)) * self.omega1 
        self.phi2_cache = self.phi2_0.reshape(1,self.n_par) + self.t_cache.reshape((self.N_cache+1,1)) * self.omega2 
        
        # self.k_cache =   swapaxes(wl.get_A(self.omega1*(self.t_cache),self.omega2*(self.t_cache)).T,0,1)+self.k 
        self.k_cache =   swapaxes(wl.get_A(self.phi1_cache.T,self.phi2_cache.T),0,2)+self.k.reshape((1,1,3))
        self.h_vec_cache = wl.get_h_vec(self.k_cache) 
    
        
        # do rotating frame interpolation. 
        theta_1_cache,theta_2_cache = so3.rotating_frame_interpolator(self.h_vec_cache,self.dt_cache)
        self.rhoeq_cache = wl.get_rhoeq_vec(self.k_cache.reshape(((self.N_cache+1)*self.n_par,3)),mu=self.Mu).reshape(self.N_cache+1,self.n_par,3)
        
        
        theta_1_norm = norm(theta_1_cache,axis=2,keepdims=1)
        theta_2_norm = norm(theta_2_cache,axis=2,keepdims=1)
        self.cosnorm_1_cache = cos(theta_1_norm)
        self.cosnorm_2_cache = cos(theta_2_norm)
        self.sinnorm_1_cache = sin(theta_1_norm)
        self.sinnorm_2_cache = sin(theta_2_norm)
        self.rvd_1_cache = theta_1_cache/theta_1_norm
        self.rvd_2_cache = theta_2_cache/theta_2_norm
        # Counter measuringh how far in the cache we are (?)
        self.ns_cache = 0
        
        # raise ValueError 
        
    def evolve(self):    
        """ 
        Main iteration.
        
        Evolves through one step dt. Updates t and rho.
        
        Also updates ns and cache 
        
        """

        
        # Load elements from cache
        # self.theta_1 = theta_1_cache[self.ns_cache]
        # self.theta_2 = self.theta_2_cache[self.ns_cache]
        self.rhoeq1  = self.rhoeq_cache[self.ns_cache]
        self.rhoeq2 = self.rhoeq_cache[self.ns_cache+1]
        
        self.cosnorm_1= self.cosnorm_1_cache[self.ns_cache]
        self.cosnorm_2= self.cosnorm_2_cache[self.ns_cache]
        self.sinnorm_1= self.sinnorm_1_cache[self.ns_cache]
        self.sinnorm_2= self.sinnorm_2_cache[self.ns_cache]
        self.rvd_1= self.rvd_1_cache[self.ns_cache]
        self.rvd_2= self.rvd_2_cache[self.ns_cache]

        # Update time, iteration step, and cache index
        self.t   += self.dt
        self.ns  += 1
        self.ns_cache+=1 
        

        # Generate new cache if cache is empty
        if self.ns_cache==self.N_cache:
            self.ns_cache=0
            self.generate_cache()


        # Compute rho_1 (used as an intermediate step in computation of steady state)
        # self.rho_1   = self.rho*exp(-self.dt/self.tau)+0.5*(1-exp(-self.dt/self.tau))*(self.rhoeq1)
        self.rho_1   = self.rho*self.tau_exp+self.tau_factor_2*self.rhoeq1

 
        # Update rho
        # self.rho   = so3.rotate(self.theta_2,so3.rotate(self.theta_1,self.rho_1))

        self.rho   = so3.efficient_rotate(self.rvd_2, self.cosnorm_2, self.sinnorm_2,so3.efficient_rotate(self.rvd_1,self.cosnorm_1,self.sinnorm_1,self.rho_1))
        
        # self.rho   += 0.5*(1-exp(-self.dt/self.tau))*self.rhoeq2
        self.rho   += self.tau_factor_2*self.rhoeq2


    def initialize_steady_state(self):
        """
        Compute steady state at phases (phi1_0[z],phi2_0[z]). Saves t and rho in self.t and self.rho 

        """
        
        B.tic()
        # Set times t_relax back in time. 
        self.t = -1*self.t_relax

        self.generate_cache()

        # Set steady state to zero
        self.rho = zeros((self.n_par,3))
        
        ### Counters to monitor progress

        # (-1) times the number of steps to evolve before steady state is reached (just used for printing progress)
        NS = int((self.t)/self.dt)+1
 
        # iteration step (just used for printing progress )
        self.ns_ss=0
        B.tic(n=18)
        
        print(f"Computing steady state. Number of iterations : {-NS}");B.tic(n=12)

        # Iterate until t reaches t0
        while self.t <-1e-10: #self.t<-1e-12:
            
            # Evolve rho
            self.evolve()
                    
            # Print progress
            self.ns_ss +=1 
            if self.ns_ss % (NS//10)==0:
                print(f"    progress: {-int(self.ns_ss/NS*100)} %. Time spent: {B.toc(n=18,disp=False):.4} s")
                sys.stdout.flush()
        
        
        print(f"done. Time spent: {B.toc(n=12,disp=0):.4}s")
        print("")
                    
    def record(self):
        """
        Record raw data to phase_array

        Returns
        -------
        None.

        """
        # print("")
        # print("Recording")
        B.tic()
        # global cumulated_data
        phi1_out = mod(self.phi1_list[:self.n_record,:],2*pi).flatten()
        phi2_out = mod(self.phi2_list[:self.n_record,:],2*pi).flatten()
        
        # assert amax(phi1_out)<2*pi
        # assert amax(phi2_out)<2*pi
        
        # bin_edges = arange(0,PHASE_RESOLUTION+1)/PHASE_RESOLUTION * 2*pi
        
        self.H = (0.5+histogram2d(phi1_out,phi2_out,self.bin_edges)[0]).astype(int)
        
        self.n_array += self.H
        
        Ind = where(self.H>0)
        
        for d in range(0,3):
            rho_out = self.rho_list[:self.n_record,:,d].flatten()
            
   
            cumulated_data = griddata((phi1_out,phi2_out),rho_out,self.interpolation_grid,method="nearest",fill_value = nan)
            
            self.phase_mat[:,:,d]+= 1*cumulated_data*self.H
            
        
        self.n_record = 0
        
        B.toc()
            
        
        
        
    def solve(self):
        

        """
        Core method. Solve time evolution until tmax and extract frequency 
        components in freqlist as output
        
        runs in paralellel by evolving NM systems in parallel. 
        each system is started at a given intial time (an integer multiple of  and evolved for N_T1 periods of T1

        Parameters
        ----------
        freqlist : ndarray or list, (NF), float
            Frequencies at which to evaluate the fourier transform
            
        tmax : float
            

        Returns
        -------
        fourier_transform : ndarray(NF,3)
            fourier_transform[nf,:] gives the fourier transform of the bloch vector of rho at frequncy freqlist[nf]. 
            Specifically,

            F[nf] = \sum_z \frac{1}{NT_1*T_1}\int_{t0_array[z]}^{t0_array[z]+NT_1*T_1} dt e^{i freqlist[nf] * t} rho(t)
        
            where rho(t) denotes the bloch vector of the steady state at time t
        """
        
        B.tic()
        # initialize rho in steady state at time 0.
        self.initialize_steady_state()

        ### Initialize counters

        # Total number of iteration steps (just used to print progress)
        self.NS_ft = -int(((self.t)-self.tmax)/self.dt)
        self.ns0 = 1*self.ns
        self.ns_ft=0
        self.counter = 0
        B.tic(n=19)
        
        print(f"Computing Fourier transform. Number of iterations : {self.NS_ft}");B.tic(n=11)
       
        
        self.Nsteps = int(self.tmax/self.dt + 100)
        
        
        if self.save_evolution:
            self.evolution_record = zeros((self.Nsteps,self.n_par,3),dtype=float)
            self.sampling_phases            = zeros((self.Nsteps,self.n_par,2),dtype=float)
   
        # Iterate until time exceeds T1*N_T1
        while self.t < self.tmax:
        
    

                
            # Evolve
            self.evolve()
            
            phi1 = self.phi1_cache[self.ns_cache]
            phi2 = self.phi2_cache[self.ns_cache]
            self.dphi = array([self.omega1*self.t,self.omega2*self.t])
            self.phi = array([phi1,phi2]).reshape((1,2,self.n_par,1))
            
            self.phi1_list[self.n_record] = 1*phi1
            self.phi2_list[self.n_record] = 1*phi2            
            self.rho_list[self.n_record]  = 1*self.rho
            self.n_record += 1 
            
            if self.n_record % self.N_record == 0:
                print("Recording");B.tic(n=68)
                self.record()
                print(f"Time spent: {B.toc(n=68,disp=False):.4} s")
            
            # # Do "manual" fourier transform, using time-difference (phase from initial time added later)
            # DT = self.t
            # # phase_arg = ind_list
            
            # self.phasemat = exp(1j*(sum(ind_list*self.phi,axis=1)))
            # self.Out += self.phasemat*self.rho
                
            if self.save_evolution:
                self.evolution_record[self.ns_ft,:,:]=1*self.rho
                self.sampling_phases[self.ns_ft] = self.phi[0,:,:,0].T
                
            
            # print progress
            self.ns_ft+=1
            if self.ns_ft-1 >self.NS_ft/10*(1+self.counter):
                print(f"    progress: {int(self.ns_ft/self.NS_ft*100)} %. Time spent: {B.toc(n=11,disp=False):.4} s")
                sys.stdout.flush()
                self.counter+=1 
        
        self.record()
        self.rho_mat = self.get_rho_mat()
        print(f"done. Time spent: {B.toc(n=11,disp=0):.4}s")
        print("")
        
        
        
        if self.save_evolution:
            self.evolution_record = self.evolution_record[:self.ns_ft]
            self.sampling_phases  = self.sampling_phases[:self.ns_ft]
            
            self.save_evolution_record()
            
        self.solved = True
        # # Add together contributions from all initializations 
        # self.fourier_transform = sum(self.Out,axis=1)/(self.n_par*(self.ns-self.ns0))
        
        # return self.fourier_transform

    def get_rho_mat(self):
        A0 = amin(self.n_array)
        global x,y,yo,p1,p1o,Ind_0,phi1,Ind,X
        if A0>0:
            return self.phase_mat/self.n_array.reshape((PHASE_RESOLUTION,PHASE_RESOLUTION,1))
        
        else:
            print("getting rhomat")
            n_depth = 5
            B.tic()
            
            Ind_0 = where(self.n_array==0)
            
            Ind = where(self.n_array>0)
            X = zeros((PHASE_RESOLUTION,PHASE_RESOLUTION,3))
            X[Ind] = self.phase_mat[Ind]/self.n_array.reshape((PHASE_RESOLUTION,PHASE_RESOLUTION,1))[Ind]
            y = X[Ind]
            (phi1,phi2) = self.interpolation_grid
            
            p1 = phi1[Ind]
            p2 = phi2[Ind]
            
            np = len(p1)
            p1o = zeros((9*np))
            p2o = zeros((9*np))
            yo  = zeros((9*np,3))
            
            nit = 0 
            for n1 in [-1,0,1]:
                for n2 in [-1,0,1]:
                   p1o[nit*np:(nit+1)*np]=p1+n1*2*pi
                   p2o[nit*np:(nit+1)*np]=p2+n2*2*pi
                   yo[nit*np:(nit+1)*np,:]=y
                   
                   nit += 1 
                   
            p1o,p2o,yo =[array(x) for x in [p1o,p2o,yo]]
            
                        
            Ind = where((abs(p1o-pi)<1.2*pi)*(abs(p2o-pi)<1.2*pi))
                        
            p1o =  p1o[Ind]
            p2o =  p2o[Ind]
            yo =  yo[Ind]    
            
            
            g_out = zeros((PHASE_RESOLUTION,PHASE_RESOLUTION,3),dtype=float)
            # Ind = where(self.n_array==0)'*
            print("Filling out empty spaces")
            
            # global f            
            for d in range(0,3):
                
                f = LinearNDInterpolator(array([p1o,p2o]).T,yo[:,d])
                # X[Ind_0][:,d] = 1*f(phi1[Ind_0],phi2[Ind_0])
                X[:,:,d] = 1*f(phi1,phi2)
            g_out = X
            # for d in range(0,3):
            #     g_out[:,:,d] = griddata((p1o,p2o),yo[:,d],self.interpolation_grid,method="linear")
            
            return g_out
            B.toc()
        
        
    def get_fourier_component(self,m,n):
    
        if not (type(m)==int and type(n)==int):
            
            raise ValueError("m and n must be integers")
    
        if not self.solved:
            self.solve()   
            
        phasemat = exp(1j*(m*self.interpolation_grid[0]+n*self.interpolation_grid[1])).reshape((PHASE_RESOLUTION,PHASE_RESOLUTION,1))
        
        nphi =  prod(shape(phasemat))
        out      = sum(phasemat*self.rho_mat,axis=(0,1))/nphi
        
        
        return out 
    
    def get_ft(self,freqlist):
        
        no = 0
        out = zeros((len(freqlist),3),dtype=complex)
        for ind in freqlist:
            (m,n) = ind
            out[no] = 1*self.get_fourier_component(m, n)
            
            no+=1 
            
        return out 
            
            
            
        
        
    def save_evolution_record(self):
        datadir = "../Time_domain_solutions/"
        filename = datadir + self.evolution_file
        
        savez(filename,k=self.k,parameters = self.parameters,phases =self.sampling_phases,evolution_record = self.evolution_record)
        

if __name__=="__main__":
    omega2 = 20*THz
    # omega1 = 0.61803398875*omega2
    omega1 = 1.500015* omega2 
    tau    = 10*picosecond
    vF     = 1e6*meter/second
    
    EF1 = 0.6*1.25*1.2e6*Volt/meter
    EF2 = 0.6*1.5*2e6*Volt/meter
    
    T1 = 2*pi/omega1
    
    Mu =115
    mu = Mu
    Temp  = 20*Kelvin;
    V0 = array([0,0,0.8*vF])
    [V0x,V0y,V0z] = V0
    parameters = 1*array([omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp])
    # set_parameters(parameters[0])
    k= array([[0,0,0.05  ]])
    
    integration_time = 1000
    
    try:
            
        S = time_domain_solver(k,parameters,integration_time,evolution_file="test")#"test_")
        B = S.get_fourier_component(0,1)
    except:
        pass
    
    from matplotlib.pyplot import *
    

    
    plot(S.sinnorm_2_cache[:,:,0])
    
    
    
   
    # t0 = array([0])
    # a = [(0,0),(1,2),(3,2),(4,5)]
    # A=S.get_phase_mat()
    phi1= mod(S.sampling_phases[:,:,0].flatten(),2*pi)
    phi2= mod(S.sampling_phases[:,:,1].flatten(),2*pi)
    
    print("Plotting")
    X,Y = meshgrid(S.bin_edges,S.bin_edges)
    figure(1)
    pcolormesh(X,Y,S.n_array.T)
    ylim((0,amax(S.n_array)))
    plot(phi1,phi2,'.w',markersize = 0.2)
    title("Number of data points")
    xlabel("$\phi_1$")
    ylabel("$\phi_2$")
    xlim(0,2*pi)
    ylim(0,2*pi)
    colorbar()
    figure(2)
    pcolormesh(X,Y,S.phase_mat[:,:,2])
    # plot(phi1,phi2,'.w',markersize = 0.1)    
    title("Accumulated data")
    xlabel("$\phi_1$")
    ylabel("$\phi_2$")
    colorbar()   
    figure(3)
    pcolormesh(X,Y,S.phase_mat[:,:,2]/S.n_array)
    # plot(phi1,phi2,'.w',markersize = 0.4)
    title("Acc. Data/Number of data points")
    xlabel("$\phi_1$")
    ylabel("$\phi_2$")
    colorbar()
    
    figure(4)
    pcolormesh(X,Y,S.rho_mat[:,:,2]);colorbar()
    # plot(phi1,phi2,'.w',markersize = 0.4)
    title("Interpolated")
    xlabel("$\phi_1$")
    ylabel("$\phi_2$")