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
"""
 
T_RELAX          = 11 # time-interval used for relaxing to steady state, in units of tau.
                                      # i.e. relative uncertainty of steady state = e^{-STEADY_STATE_RELATIVE_TMAX}
NMAT_MAX         = 100
T_RES            = 1000   # time resolution that enters. 
CACHE_ELEMENTS   = 1e6  # Number of entries in cached quantities

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
    def __init__(self,k,parameters,evolution_file=None):
        self.k = k
        self.parameters = parameters
        
        assert (evolution_file is None) or type(evolution_file)==str,"save_evolution must be None or str"
        if type(evolution_file)==str:
            
            self.save_evolution = True
            self.evolution_file = evolution_file
        else:
            self.save_evolution = False
            self.evolution_file = ""
        
        print(evolution_file)
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
            
        # Set time integration parameters
        self.dt = self.get_dt()        
        self.tau_factor = 1-exp(-self.dt/self.tau)
        self.t_relax = T_RELAX * self.tau # time-interval used for relaxing to steady state
        
        # Initialize running variables
        self.ns  = 0
        self.rho = None
        self.t   = None
        self.theta_1 = None
        self.theta_2 = None
        
        self.N_cache = int(CACHE_ELEMENTS/NMAT_MAX)+1   # number of steps to cache at a time

    def get_dt(self):
        """
        determine time integration increment. 
        Increment is an iarrational factor of order 1 times the smallest of 
        
        T1/T_RES, T2/T_RES, tau/T_RES

        Returns
        -------
        dt, float. 
            time integration increment.

        """
  
        return amin(abs(array([self.T1/T_RES,self.T2/T_RES,self.tau/T_RES])))*sqrt(0.5)
    
        
    def generate_cache(self):
        """ 
        t_cachce[nt,z] gives nt-th time step of realization z in the cache. nt runs from 0 to (including) self.N_cache
        t_cache[nt,z] = t[z] + nt*self.dt
        
        self.k_cache[nt,z] gives the self.k+A(t_cache[nt,z])
        self.hvec_cache[nt,z] gives the hvec(self.k_cache[nt,z])
    
        """
        self.t_cache = self.t.reshape((1,len(self.t))) + arange(self.N_cache+1).reshape((self.N_cache+1,1))*self.dt
        self.k_cache =   swapaxes(wl.get_A(self.omega1*(self.t_cache),self.omega2*(self.t_cache)).T,0,1)+self.k 
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


    def set_steady_state(self,t0):
        """
        Compute steady state at times in t0. Saves t and rho in self.t and self.rho 

        Parameters
        ----------
        t0 : ndarray(NT0), float
            times at which to evaluate steady state.

        Returns
        -------
        None.

        """
        # Number of different initial times to be probed
        NT0 = len(t0)
        
        # Set times t_relax back in time. 
        self.t = t0-1*self.t_relax
        self.generate_cache()

        # Set steady state to zero
        self.rho = zeros((NT0,3))
        
        ### Counters to monitor progress

        # (-1) times the number of steps to evolve before steady state is reached (just used for printing progress)
        NS = int((self.t[0]-t0[0])/self.dt)+1
        # iteration step (just used for printing progress )
        self.ns_ss=0
        B.tic(n=18)
        
        print(f"Computing steady state. Number of iterations : {-NS}");B.tic(n=12)

        # Iterate until t reaches t0
        while self.t[0]-t0[0]<-1e-12:
            
            # Evolve rho
            self.evolve()
                    
            # Print progress
            self.ns_ss +=1 
            if self.ns_ss % (NS//10)==0:
                print(f"    progress: {-int(self.ns_ss/NS*100)} %. Time spent: {B.toc(n=18,disp=False):.4} s")
                sys.stdout.flush()
            
        print(f"done. Time spent: {B.toc(n=12,disp=0):.4}s")
        print("")
            
    def find_optimal_NM(self,tmax):
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
        nml= arange(1,NMAT_MAX)
        
        self.cost = T_RELAX*self.tau * sqrt(100**2+nml**2) + tmax/nml* sqrt(100**2+nml**2)
        
        a0 = argmin(self.cost)
        NM = nml[a0]
        
        if NM > tmax/self.T1:
            N_T1 = 1
            
        else:
            N_T1 = int(tmax/(self.T1*NM))
            
        return NM,N_T1
        
    def get_ft(self,freqlist,tmax):
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

        # Find optimal paralellization scheme
        self.NM,self.N_T1 = self.find_optimal_NM(tmax)
        
        # Set initial time of each parallel system to be evolved
        npr.seed(0)
        self.t0_array = self.N_T1*self.T1 * (arange(self.NM)+1000*npr.rand(self.NM))
  
        # initialize rho in steady state at times in t0_array
        self.set_steady_state(self.t0_array)
        # """
        # Begin bypass code
        # """
        # print("WARNING: bypassed steady state for testing purposes. Uncomment code before using")
        # self.rho = array([[ 0.04844703,  0.00797926,  0.18033738],
        # [ 0.04783463, -0.1675469 ,  0.06433975],
        # [ 0.0482473 , -0.17545244, -0.03456465],
        # [ 0.04821787,  0.06624495,  0.16833097],
        # [ 0.0489186 ,  0.11059038, -0.14374385],
        # [ 0.04862403,  0.17031926, -0.06484554],
        # [ 0.04803467, -0.10900376, -0.14273384],
        # [ 0.04810566, -0.04853653, -0.17350298],
        # [ 0.04897943,  0.15416626,  0.09543941],
        # [ 0.04895121,  0.15204543,  0.0987585 ],
        # [ 0.0478431 , -0.16601233,  0.06825548],
        # [ 0.04861574,  0.16833781, -0.0697778 ],
        # [ 0.04835562, -0.16061988, -0.07855107],
        # [ 0.04911005,  0.16557859,  0.074119  ],
        # [ 0.04828069,  0.09046061,  0.15680233],
        # [ 0.04840674, -0.01064955, -0.18009798],
        # [ 0.04868758, -0.03377451,  0.17703148],
        # [ 0.04815143, -0.14019981,  0.11239497],
        # [ 0.04862892, -0.09761597,  0.15107842],
        # [ 0.048726  ,  0.02554999, -0.17881801],
        # [ 0.04841289,  0.01344577,  0.18005101],
        # [ 0.04805615, -0.05743951, -0.1706918 ]])
        # self.ns_cache = 9514
        # self.t = 1*self.t0_array
        
        # self.generate_cache()
        # self.ns = 49518
        # """End of bypass code
        # """

        # Initialize output array
        self.Out = zeros((N_freqs,self.NM,3),dtype=complex)

        ### Initialize counters

        # Total number of iteration steps (just used to print progress)
        self.NS_ft = -int(((self.t[0])-(self.t0_array[0]+self.T1*self.N_T1))/self.dt)
        self.ns0 = 1*self.ns
        self.ns_ft=0
        self.counter = 0
        B.tic(n=19)
        
        print(f"Computing Fourier transform. Number of iterations : {self.NS_ft}");B.tic(n=11)
       
        
        self.Nsteps = int(self.T1*self.N_T1/self.dt + 100)
        
        
        if self.save_evolution:
            self.evolution_record = zeros((self.Nsteps,self.NM,3),dtype=float)
            self.sampling_times            = zeros((self.Nsteps,self.NM),dtype=float)
        # Iterate until time exceeds T1*N_T1
        while self.t[0]-self.t0_array[0] < self.T1*self.N_T1:
            
            if self.save_evolution:
                self.evolution_record[self.ns_ft,:,:]=self.rho
                self.sampling_times[self.ns_ft,:] = self.t
                
            # Evolve
            self.evolve()
            
            
            # Do "manual" fourier transform, using time-difference (phase from initial time added later)
            DT = self.t[0]-self.t0_array[0]
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
        self.Out = self.Out * exp(1j*freqlist*self.t0_array.reshape((1,len(self.t0_array),1)))
        
        # Add together contributions from all initializations 
        self.fourier_transform = sum(self.Out,axis=1)/(self.NM*(self.ns-self.ns0))
        
        return self.fourier_transform

    def save_evolution_record(self):
        datadir = "../Time_domain_solutions/"
        filename = datadir + self.evolution_file
        
        savez(filename,k=self.k,parameters = self.parameters,times =self.sampling_times,evolution_record = self.evolution_record)
        

if __name__=="__main__":
    omega2 = 20*THz
    omega1 = 0.61803398875*omega2
    # omega1 = 3/2 * omega2 
    tau    = 10*picosecond
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
    
    S = time_domain_solver(k,parameters,evolution_file="test")
    t0 = array([0])
    A=S.get_ft([0],500)
