#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 08:51:26 2020

@author: frederik

Solution of master equation in time domain

v1: speedup computing of steady states and fourier transform. 
v4: with linear interpolation in iteration to reduce correction from O(dt) to O(dt^2)
v9: using SO(3) representation. Using rotating frame interpolator
"""
import os 
RELATIVE_DT_MAX = 1    # Maximum dt relative to 1/|H|.
T_RELAX         = 11 # time-interval used for relaxing to steady state, in units of tau.
                                      # i.e. relative uncertainty of steady state = e^{-STEADY_STATE_RELATIVE_TMAX}
NMAT_MAX        = 100
T_RES           = 100
CACHE_ELEMENTS   = 1e6  # Number of entries in cached quantities

from scipy import *
from Units import *
import weyl_liouvillian_v1 as wl
import so3 as so3
import sys 

I3 = eye(3)
Generator = zeros((3,3,3))

Generator[0,1,2],Generator[0,2,1]=1,-1
Generator[1,2,0],Generator[1,0,2]=1,-1
Generator[2,0,1],Generator[2,1,0]=1,-1


class time_domain_solver():
    def __init__(self,k):
        self.k = k
     
        self.dt = self.get_dt()
        
        
           
        self.T1 = 2*pi/omega1
        self.T2 = 2*pi/omega2
        
        self.tau_factor = 1-exp(-self.dt/tau)
        self.t_relax = T_RELAX * tau # time-interval used for relaxing to steady state
        
        self.ns  = 0
        
        self.rho = None
        self.t   = None
        
        self.theta_1 = None
        self.theta_2 = None
        
        self.N_cache = int(CACHE_ELEMENTS/NMAT_MAX)+1   # number of steps to cache at a time
        B.tic(n=7)
    def get_dt(self):
  
        return amin(abs(array([T1/T_RES,T2/T_RES,tau/T_RES])))*0.87234987261346789236
        
    def __rotation_iteration(self,vector):
        return so3.rotate(self.theta_2,so3.rotate(self.theta_1,vector))
    

    def set_t(self,t):
        self.t = t
        self.generate_cache()
        self.ns_cache = 0
        
        
    def generate_cache(self):
        t_cache = self.t.reshape((1,len(self.t))) + arange(self.N_cache+1).reshape((self.N_cache+1,1))*self.dt
        k_cache =   swapaxes(wl.get_A(omega1*(t_cache),omega2*(t_cache)).T,0,1)+self.k #,ndmin=2)
        h_vec_cache = wl.get_h_vec(k_cache) 
        h1 = h_vec_cache[:-1]
        h2 = h_vec_cache[1:]
        self.theta_1_cache,self.theta_2_cache = so3.rotating_frame_interpolator(h1*self.dt,h2*self.dt)
        self.rhoeq_cache = wl.get_rhoeq_vec(k_cache.reshape(((self.N_cache+1)*self.N_mat,3)),mu=Mu).reshape(self.N_cache+1,self.N_mat,3)
        self.ns_cache = 0
        

    def iterate_nb(self):
        ##1
        self.theta_1 = self.theta_1_cache[self.ns_cache]
        self.theta_2 = self.theta_2_cache[self.ns_cache]
        self.rhoeq1  = self.rhoeq_cache[self.ns_cache]
        self.rhoeq2 = self.rhoeq_cache[self.ns_cache+1]
        
        self.t   += self.dt
        self.ns  += 1
        self.ns_cache+=1 
        
        if self.ns_cache==self.N_cache:
            self.ns_cache=0
            self.generate_cache()

    def evolve(self):    

        
        
        self.iterate_nb()
        self.rho_1   = self.rho*exp(-self.dt/tau)+0.5*(1-exp(-self.dt/tau))*(self.rhoeq1)
        self.rho   = self.__rotation_iteration(self.rho_1) + 0.5*(1-exp(-self.dt/tau))*self.rhoeq2


    def set_steady_state(self,t0):
        """
        Compute steady state at times in t0 t=0
        """
        NT0 = len(t0)
        self.set_t(t0-1*self.t_relax)
        self.rho = zeros((NT0,3))
        
        ## Counters to monitor progress
        
        NS = int((self.t[0]-t0[0])/self.dt)+1
        self.ns_ss=0
        B.tic(n=18)
        print(f"Computing steady state. Number of iterations : {-NS}");B.tic(n=12)

#        print(f"    number of iterations      : {-NS}")
#        print("")
        while self.t[0]-t0[0]<-1e-12:
            
            self.evolve()
            
            self.ns_ss +=1 
            
            if self.ns_ss % (NS//10)==0:
                print(f"    progress: {-int(self.ns_ss/NS*100)} %. Time spent: {B.toc(n=18,disp=False):.4} s")
                sys.stdout.flush()
            
        print(f"done. Time spent: {B.toc(n=12,disp=0):.4}s")
        print("")
            
    def find_optimal_nmat(self,tmax):
        nml= arange(1,NMAT_MAX)
        
        self.cost = T_RELAX*tau * sqrt(100**2+nml**2) + tmax/nml* sqrt(100**2+nml**2)
        
        a0 = argmin(self.cost)
        N_mat = nml[a0]
        
        if N_mat > tmax/self.T1:
            N_T1 = 1
            
        else:
            N_T1 = int(tmax/(self.T1*N_mat))
            
        return N_mat,N_T1
        
    def get_ft(self,freqlist,tmax):
        """ Return fourier transform over effective time-interval of length tmax"""
        
        global t0_array,fl
        N_freqs= len(freqlist)
      
        self.N_mat,N_T1 = self.find_optimal_nmat(tmax)
        


        npr.seed(0)
        t0_array = N_T1*self.T1 * (arange(self.N_mat)+1000*npr.rand(self.N_mat))
        freqlist = array(freqlist).reshape((N_freqs,1,1))
  
        self.set_steady_state(t0_array)

        
        self.N_T1 = N_T1

        self.Out = zeros((N_freqs,self.N_mat,3),dtype=complex)


        self.ns0 = 1*self.ns
        
        self.NS_ft = -int(((self.t[0])-(t0_array[0]+self.T1*N_T1))/self.dt)
        print(f"Computing Fourier transform. Number of iterations : {self.NS_ft}");B.tic(n=11)
        self.ns_ft=0
        self.counter = 0
        B.tic(n=19)
        while self.t[0]-t0_array[0] < self.T1*N_T1:
            
            self.evolve()
            DT = self.t[0]-t0_array[0]

            self.Out += exp(1j*freqlist*DT)*self.rho
            
            
            if self.ns_ft >self.NS_ft/10*(1+self.counter):
                
                print(f"    progress: {int(self.ns_ft/self.NS_ft*100)} %. Time spent: {B.toc(n=11,disp=False):.4} s")
                sys.stdout.flush()
                self.counter+=1 
                
            self.ns_ft+=1

        self.Out = self.Out * exp(1j*freqlist*t0_array.reshape((1,len(t0_array),1)))
        self.Out = sum(self.Out,axis=1)/(self.N_mat*(self.ns-self.ns0))#(N_T1*self.T1*self.N_mat)*self.dt       
        print(f"done. Time spent: {B.toc(n=11,disp=0):.4}s")
        print("")
        return self.Out

  
    
def set_parameters(parameters):
    global omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp,T1,T2
    wl.set_parameters(parameters)

    [omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp]=parameters

    T1 = 2*pi/omega1
    T2 = 2*pi/omega2
 
    global V0,P0,A1,A2
    V0 = array([V0x,V0y,V0z])
    P0 = omega1*omega2/(2*pi)
    
    
    ## vector field amplitude 
    A1 = EF1/omega1
    A2 = EF2/omega2

    
    global SX,SY,SZ,I2,sx,sy,sz,i2,ZM,Ind,I6,IP,IPP,IF,NPP,DH

    
    [SX,SY,SZ,I2] = [ve.SX,ve.SY,ve.SZ,ve.I2]
    [sx,sy,sz,i2] = [q.flatten() for q in [SX,SY,SZ,I2]]
    
    ZM = zeros((4,4),dtype=complex)
    
    # Indices in vectorized matrix space, where \rho may be nonzero. (we restrict ourselves to this subspace)
    Ind = array([5,6,9,10])


if __name__=="__main__":
    omega2 = 20*THz
    omega1 = 0.61803398875*omega2
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
    parameters = 1*array([[omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp]])
    set_parameters(parameters[0])
    k= array([[ 0.,  0.        , 0      ]])
    
    S = time_domain_solver(k)
    t0 = array([0])
    A=S.get_ft([0],500)
