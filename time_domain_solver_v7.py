#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 08:51:26 2020

@author: frederik

Solution of master equation in time domain

v1: speedup computing of steady states and fourier transform. 
v4: with linear interpolation in iteration to reduce correction from O(dt) to O(dt^2)
"""
import os 
RELATIVE_DT_MAX = 20    # Maximum dt relative to 1/|H|.
T_RELAX = 30 # time-interval used for relaxing to steady state, in units of tau.
                                      # i.e. relative uncertainty of steady state = e^{-STEADY_STATE_RELATIVE_TMAX}

NMAT_MAX = 50
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
        

    
#        dt = amin(self.dtmin_array)
#        self.N_mat = None
    def get_dt(self):
        kgrid = wl.get_kgrid(100,k0=self.k)

        h  =wl.get_h(kgrid)
        Amax = amax(norm(h,axis=(1,2)))
        return amin(abs(array([RELATIVE_DT_MAX/Amax,tau/300])))


    def iterate(self):       
        global h_vec,rho0_vec,h
        self.k_new     = array(self.k + wl.get_A(omega1*self.t,omega2*self.t).T,ndmin=2)


        self.h_vec     = wl.get_h_vec(self.k_new)
#        self.h = wl.get_h(self.k_new)
        self.rho0_vec  = wl.get_rhoeq_vec(self.k_new)


        self.rho_1   = self.rho*exp(-self.dt/tau)
        self.rho_2   = so3.rotate(self.h_vec*self.dt,self.rho_1)
        self.drho_1  = self.dt/tau*self.rho0_vec
        self.drho_2  = 0.5*(self.drho_1+ so3.rotate(self.h_vec*self.dt,self.drho_1))
        
        self.rho = self.rho_2 + self.drho_2
#        self.p   = self.p * exp(-self.dt/tau)
#        self.p   = self.p + self.dt/tau * 
#        self.rho_1   = self.rho*exp(-self.dt/tau)
#        self.rho_2   =  so3.rotate(h_vec*self.dt,self.rho)  - self.tau_factor*(self.rho-rho0_vec)
#        self.rho = 
        
        self.t   += self.dt
        self.ns  += 1
#        raise ValueError
        
    def set_steady_state(self,t0):
        """
        Compute steady state at times in t0 t=0
        """
        NT0 = len(t0)
        self.t = t0-1*self.t_relax
        self.rho = zeros((NT0,3))
        
        while self.t[0]-t0[0]<-1e-12:
            
            self.iterate()
            if self.ns%1000 == 0 :
                
                print(self.t[0]-t0[0])
    def get_ft(self,freqlist,tmax):
        """ Return fourier transform over effective time-interval of length tmax"""
        
        global t0_array,fl
        N_freqs= len(freqlist)

        # Parallelize
        
        # N_T1 Number of periods of mode 1 to evolve over
        if tmax/self.T1 < NMAT_MAX:
            self.N_mat = int(tmax/self.T1)+1
            N_T1 = 1 
            
        else:
            self.N_mat= NMAT_MAX
            N_T1 = int(tmax/(self.T1*self.N_mat))+1
            
        
        # format freqlist

        npr.seed(0)
        t0_array = N_T1*self.T1 * (arange(self.N_mat)+1000*npr.rand(self.N_mat))
        freqlist = array(freqlist).reshape((N_freqs,1,1))
  
        print("Getting to steady state")
        self.set_steady_state(t0_array)
        
        self.Out = zeros((N_freqs,self.N_mat,3),dtype=complex)

        while self.t[0]-t0_array[0] < self.T1*N_T1:
            
            self.iterate()
            DT = self.t[0]-t0_array[0]

            self.Out += exp(1j*freqlist*DT)*self.rho
            if self.ns%1000==0:
                print(self.t[0]-t0_array[0] - self.T1*N_T1)
#            print(DT)

        self.Out = self.Out * exp(1j*freqlist*t0_array.reshape((1,len(t0_array),1)))
        self.Out = sum(self.Out,axis=1)/(N_T1*self.T1*self.N_mat)*self.dt       
        
        return self.Out
#        self.t0_array = self.t0_array.reshape((1,self.nmat,1,1))
#        
#        self.Out = zeros((nfreqs,self.nmat,2,2),dtype=complex)



#        # list used to monitor convergence, for diagnostics
#        self.convergence_list= []
#        
#        self.nt=0
#        self.delta_t_list = arange(0,self.n_T1*self.T1,self.dt)
#        
#        self.NDT = len(self.delta_t_list)
#        print(f"Number of iterations to compute fourier transform: {self.NDT*1e-6:.4} million")
#        print(f"Total time effectively integrated over: {self.n_T1*self.nmat} * T_1")
#        for self.delta_t  in self.delta_t_list:
#            self.t = self.t0_array + self.delta_t
#
#            self.Out += exp(1j*self.freqlist*self.t)*self.R     
#
#            self.__iterate(self.t)
#
#            self.nt+=1
#
#            if self.nt%(self.NDT//10) ==0:
#                print(f"    progress: {int(self.nt/self.NDT*100+0.1)} %. Time spent: {B.toc(n=18,disp=False):.4} s")
#                print(f"    current value : {sum(abs(self.Out))/(self.delta_t):.4}")
#                sys.stdout.flush()
#            if self.nt%(self.NDT//100) ==0:
#                self.convergence_list.append(self.Out/(self.delta_t))
#   
#        self.Out = sum(self.Out,axis=1)/(self.n_T1*self.T1*self.nmat)*self.dt
#        print(f"Done. Time spent: {B.toc(n=17,disp=0):.4} s")
#
#        self.t_out  =  self.T1 * self.nmat * self.n_T1
#        
#        return self.Out,self.t_out
#    
#        return self.R0        
        
        
  
    
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
    tau    = 0.05*picosecond
    vF     = 1e5*meter/second
    EF2 = 0.6*1.5*2e6*Volt/meter
    EF1 = 0.6*1.25*1.2e6*Volt/meter
    T1 = 2*pi/omega1
    Mu =110*0.1
    Temp  = 20*Kelvin*0.1;
    
    V0 = array([0,0,0.8*vF])
    [V0x,V0y,V0z] = V0
    parameters = 1*array([[omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp]]*len(klist))
    set_parameters(parameters[0])

    k = array([0.2,0,0])
    S = time_domain_solver(k)
    t0 = array([0])
    S.set_steady_state(t0)
    
#    A=S.get_ft([1,2,3,4],10)
    
    
    print("\nSpin components using v7:")
    print(S.rho)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#    S.solve()