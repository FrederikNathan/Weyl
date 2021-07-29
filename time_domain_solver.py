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
 
T_RELAX          = 11 # time-interval used for relaxing to steady state, in units of tau.
                                      # i.e. relative uncertainty of steady state = e^{-STEADY_STATE_RELATIVE_TMAX}
NMAT_MAX         = 100
T_RES            = 10   # time resolution that enters. 
print("WARNING - SET T_RES BACK TO 1000 BEFORE USING")
CACHE_ELEMENTS   = 1e6  # Number of entries in cached quantities
# N_CONTOURS       = 200 # NUMBER Of contours in the phase brillouin zone 

import os 
from scipy import *
import sys 

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
    def __init__(self,k,parameters,integration_time,evolution_file=None):
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
        
        self.t_relax = self.get_t_relax()  # time-interval used for relaxing to steady state
        
        self.is_commensurate,self.frequency_fraction,self.ext_period = self.is_commensurate()
        self.integration_window_T1 = self.get_optimal_integration_time_in_periods_of_mode_1()
        self.NM = self.find_optimal_NM()
        self.tmax = self.integration_window_T1/self.runs_per_contour*self.T1 
        self.phi1_0,self.phi2_0 = self.get_initial_phases()
            


        self.time_per_sample = self.integration_time/self.NM 

        # Set time integration parameters


        # self.tmax = self.tmax_T1 * self.T1 
        self.res = self.get_res()
        self.dt  = self.T1/self.res 
        self.tau_factor = 1-exp(-self.dt/self.tau)

        
        # Initialize running variables
        self.ns  = 0
        self.rho = None
        self.t   = None
        self.theta_1 = None
        self.theta_2 = None
        
        self.N_cache = max(1,int(CACHE_ELEMENTS/NMAT_MAX))   # number of steps to cache at a time

        # Find optimal paralellization scheme
    
        # self.phase_difference_array = self.get_initial_phases()# linspace(0,2*pi,self.NM+1)[:-1]
        #self.N_T1*self.T1 * (arange(self.NM)+1000*npr.rand(self.NM))
  
        # self.phi1_0  = zeros(self.NM)# 0*ones(self.NM)*self.omega1
        # self.phi2_0  = self.phi1_0 + self.phase_difference_array
    
    def get_t_relax(self):
        a = T_RELAX * self.tau
        b = a/self.T1 
         
        out = int(b+0.9)*self.T1 
        
        return out 
    
    def is_commensurate(self):
        """ 
        check if frequencies are commensurate within 1.2 * integration_time.
        """
        self.r = arange(1,self.integration_time*1.2/self.T1)*self.T1/self.T2
        self.r = mod(self.r+0.5,1)-0.5 
        
        vec = where(abs(self.r)<1e-10)[0]+1
        
        if len(vec)>0:
  
            q = amin(vec)
            p = int((q*self.T1)/self.T2 +0.5)
            
            frequency_fraction = (q,p)
            
            ext_period = q*self.T1
        

            return True,frequency_fraction,ext_period
        
        else:
            
            return False,None,None
        
    def get_optimal_integration_time_in_periods_of_mode_1(self):
        """
        Find optimal integration window in periods of mode 1 such that 
        the final phases are close to the initial phases. 


        Returns
        -------
        tmax_T1 : int
            tmax, in units of T1. (i.e., tmax = tmax_T1 * T1)

        """
        
        if self.is_commensurate:
            
            tmax_T1 = self.frequency_fraction[0]
            return tmax_T1 
        
            
        else :
        
        
            # self.a = int(self.integration_time/self.T1)
        
            self.b = int(self.integration_time/self.T1*1.2)
            # First determine tmax 
            
            self.v = arange(1,self.b+1)
            self.d = mod(self.v*self.T1+0.5*self.T2,self.T2)-0.5*self.T2
            self.d = self.d/(self.v**0.9)
            a0 = argmin(abs(self.d))
        
            tmax_T1 =  self.v[a0]
            
            p = int((tmax_T1*self.T1)/self.T2 +0.5)
            
        
            return tmax_T1         
    
    
    
    def find_optimal_NM(self):
        # raise NotImplementedError("Obsolete method")
        """
        Find optimal way of parallelizing time evolution

        Parameters
        ----------
        tmax : float
            evolution time to be probed.

        Returns
        -------
        NM : int
            number of parallel systems to evolve.
        N_T1 : int
            number of periods of T1 to evolve over(?).

        """

        # Determine number of separate (quasi-)closed contours
        self.n_contours = max(1,int(0.5+self.integration_time /(self.T1*self.integration_window_T1)))
        self.n_contours = min(NMAT_MAX,self.n_contours)
        
        
        # if self.is_commensurate:
            
            
            
        nml= arange(1,max(2,NMAT_MAX//self.n_contours))*self.n_contours
        self.cost = self.t_relax * sqrt(100**2+nml**2) + self.T1*self.integration_window_T1*self.n_contours/nml* sqrt(100**2+nml**2)
        a0 = argmin(self.cost)
        NM = nml[a0]
        
        self.runs_per_contour = NM//self.n_contours

        return NM
        

    def get_initial_phases(self):
        
        phi1list = []
        phi2list = []
        
        
        
        contour_distance = 2*pi/self.integration_window_T1
        
        for nc in range(0,self.n_contours):
            phi10 = 0
            phi20  = contour_distance * nc/self.n_contours     
            for nr in range(0,self.runs_per_contour):
                
                phi1 = mod(phi10 + nr*self.tmax*self.omega1,2*pi)
                phi2 = mod(phi20 + nr*self.tmax*self.omega2,2*pi)
                
                phi1list.append(1*phi1)
                phi2list.append(1*phi2)
                
        
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
        
        self.t_cache = self.t + arange(self.N_cache+1)
        
        
        self.phi1_cache = self.phi1_0.reshape(1,self.NM) + self.t_cache.reshape((self.N_cache+1,1)) * omega1 
        self.phi2_cache = self.phi2_0.reshape(1,self.NM) + self.t_cache.reshape((self.N_cache+1,1)) * omega2 
        
        # self.k_cache =   swapaxes(wl.get_A(self.omega1*(self.t_cache),self.omega2*(self.t_cache)).T,0,1)+self.k 
        self.k_cache =   swapaxes(wl.get_A(self.phi1_cache.T,self.phi2_cache.T),0,2)+self.k.reshape((1,1,3))
        self.h_vec_cache = wl.get_h_vec(self.k_cache) 
    
        
        # do rotating frame interpolation. 
        self.theta_1_cache,self.theta_2_cache = so3.rotating_frame_interpolator(self.h_vec_cache,self.dt)
        self.rhoeq_cache = wl.get_rhoeq_vec(self.k_cache.reshape(((self.N_cache+1)*self.NM,3)),mu=self.Mu).reshape(self.N_cache+1,self.NM,3)
        
        # Counter measuringh how far in the cache we are (?)
        self.ns_cache = 0
        
    def evolve(self):    
        """ 
        Main iteration.
        
        Evolves through one step dt. Updates t and rho.
        
        Also updates ns and cache 
        
        """
        
        # Load elements from cache
        self.theta_1 = self.theta_1_cache[self.ns_cache]
        self.theta_2 = self.theta_2_cache[self.ns_cache]
        self.rhoeq1  = self.rhoeq_cache[self.ns_cache]
        self.rhoeq2 = self.rhoeq_cache[self.ns_cache+1]
        
        # Update time, iteration step, and cache index
        self.t   += self.dt
        self.ns  += 1
        self.ns_cache+=1 
        
        # Generate new cache if cache is empty
        if self.ns_cache==self.N_cache:
            self.ns_cache=0
            self.generate_cache()

        # Compute rho_1 (used as an intermediate step in computation of steady state)
        self.rho_1   = self.rho*exp(-self.dt/self.tau)+0.5*(1-exp(-self.dt/self.tau))*(self.rhoeq1)
        
        # Update rho
        
        self.rho   = so3.rotate(self.theta_2,so3.rotate(self.theta_1,self.rho_1))
        self.rho   += 0.5*(1-exp(-self.dt/self.tau))*self.rhoeq2


    def initialize_steady_state(self):
        """
        Compute steady state at phases (phi1[z],phi2[z]). Saves t and rho in self.t and self.rho 

        Parameters
        ----------
        t0 : ndarray(NT0), float
            times at which to evaluate steady state.

        Returns
        -------
        None.

        """
        # assert len(phi1)==len(phi2),"Length of phase arguments must be identical"

        # NT0 = len(phi1)
        # NS = len(phi1)
        # NT0 = self.NM
        
        # Set times t_relax back in time. 
        self.t = -1*self.t_relax

        self.generate_cache()

        # Set steady state to zero
        self.rho = zeros((self.NM,3))
        
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
            


    # def find_optimal_NM(self):
    #     A = self.tmax_T1
        
        
    def get_ft(self,freqlist):
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
        
        # Reshape freqlist to the right dimensionality
        N_freqs= len(freqlist)
        freqlist = array(freqlist).reshape((N_freqs,1,1))


        
        # Set initial time of each parallel system to be evolved
        # self.phase_difference_array = arange(0,self.NM)/self.NM*2*pi# linspace(0,2*pi,self.NM+1)[:-1]
        #self.N_T1*self.T1 * (arange(self.NM)+1000*npr.rand(self.NM))
  
        # self.phi1_0  = 0*ones(self.NM)
        # self.phi2_0  = self.phi1_0 + self.phase_difference_array
        
        
        
        
        # initialize rho in steady state at time 0.
        self.initialize_steady_state()
        
        # Initialize output array
        self.Out = zeros((N_freqs,self.NM,3),dtype=complex)

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
            self.evolution_record = zeros((self.Nsteps,self.NM,3),dtype=float)
            self.sampling_times            = zeros((self.Nsteps,self.NM),dtype=float)
        # Iterate until time exceeds T1*N_T1
        while self.t < self.tmax:
            
            if self.save_evolution:
                self.evolution_record[self.ns_ft,:,:]=self.rho
                self.sampling_times[self.ns_ft] = self.t
                
                
            # Evolve
            self.evolve()
            
            
            # Do "manual" fourier transform, using time-difference (phase from initial time added later)
            DT = self.t
            self.Out += exp(1j*freqlist*DT)*self.rho
            
            # print progress
            self.ns_ft+=1
            if self.ns_ft-1 >self.NS_ft/10*(1+self.counter):
                print(f"    progress: {int(self.ns_ft/self.NS_ft*100)} %. Time spent: {B.toc(n=11,disp=False):.4} s")
                sys.stdout.flush()
                self.counter+=1 
        
        print(f"done. Time spent: {B.toc(n=11,disp=0):.4}s")
        print("")
        
        
        if self.save_evolution:
            self.evolution_record = self.evolution_record[:self.ns_ft]
            self.sampling_times   = self.sampling_times[:self.ns_ft]
            
            self.save_evolution_record()
            
            
        # Modify initial phases of fourier transform 
        print("Modify this to reflect the new phase differences.")
        self.Out = self.Out * exp(1j*freqlist*self.t0_array.reshape((1,len(self.t0_array),1)))
        
        # Add together contributions from all initializations 
        self.fourier_transform = sum(self.Out,axis=1)/(self.NM*(self.ns-self.ns0))
        
        return self.fourier_transform

    def save_evolution_record(self):
        datadir = "../Time_domain_solutions/"
        filename = datadir + self.evolution_file
        
        savez(filename,k=self.k,parameters = self.parameters,times =self.sampling_times,evolution_record = self.evolution_record,phi1_0 = self.phi1_0,phi2_0=self.phi2_0)
        

if __name__=="__main__":
    omega2 = 20*THz
    omega1 = 0.61803398875*omega2
    omega1 = 1.5000* omega2 
    tau    = 1*picosecond
    vF     = 1e5*meter/second
    
    EF2 = 0.6*1.5*2e6*Volt/meter
    EF1 = 0.6*1.25*1.2e6*Volt/meter*1e-10
    
    T1 = 2*pi/omega1
    
    Mu =115*0.1
    mu = Mu
    Temp  = 20*Kelvin*0.1;
    V0 = array([0,0,0.8*vF])
    [V0x,V0y,V0z] = V0
    parameters = 1*array([omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp])
    # set_parameters(parameters[0])
    k= array([[ 0.,  0.        , 0      ]])
    
    integration_time = 10000
    S = time_domain_solver(k,parameters,integration_time,evolution_file="test")
    
    
    
    
    
    
    
    
    # t0 = array([0])
    A=S.get_ft([0])
