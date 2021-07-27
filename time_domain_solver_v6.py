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
RELATIVE_DT_MAX = 0.1    # Maximum dt relative to 1/|H|.
STEADY_STATE_RELATIVE_TMAX = 14 # upper bound of integration for steady state, in units of tau.
                                      # i.e. relative uncertainty of steady state = e^{-STEADY_STATE_RELATIVE_TMAX}

from scipy import *
from Units import *
import weyl_liouvillian_v1 as wl
import sys 


class time_domain_solver():
    def __init__(self,k,Nphi=100,dtmax=inf):
        self.k = k
        self.Nphi = Nphi

        self.T1 = 2*pi/omega1
        self.T2 = 2*pi/omega2
        
        self.kgrid = wl.get_kgrid(self.Nphi,k0=self.k)
        self.dtmax = dtmax


        self.U_array,self.dt = self.get_uarray()
        self.Rgrid = wl.get_rhoeq(self.kgrid,mu=Mu)[1].reshape((self.Nphi,self.Nphi,2,2))

        self.NMAT_MAX = 50
#        
##        self.dhdt_mode_1 =  wl.
#    def get_uarray(self):
#        """
#        get array of e^{-ih(\phi_1,\phi_2)dt} for \phi_1,\phi_2 = (0,1,...N)*2pi/N
#        
#        also computes the crucial dt variable
#        """
#        
#        print("Initializing solver");B.tic(n=211)
#        S = self.Nphi**2
#        
#        Out = array([eye(2,dtype=complex) for n in range(0,S)])# zeros((S,4,4),dtype=complex)
#
#        h  =wl.get_h(self.kgrid)
#        Amax = amax(norm(h,axis=(1,2)))
#        
#        self.dtmin_array = abs(array([RELATIVE_DT_MAX/Amax,
#                 self.dtmax,
#                 self.T2/((sqrt(2)-0.2)*self.Nphi),
#                 tau/300]))
#    
#        methodstr = [f"{RELATIVE_DT_MAX}/sup_t|h(t)|","dtmax set by caller","T2/Nphi","tau/300"]
#        dt = amin(self.dtmin_array)
#        modeindex = argmin(self.dtmin_array)
#        print(f"    using method {modeindex} to set dt : dt = {methodstr[modeindex]}")
##        dt = DT
#    
#        # rescale by random number to avoid incommensuratin effects
#        
#        self.dt =(0.17+log(2))*dt
#        
#        generator = -1j*h*dt
#    
#        X = 1*generator 
#        
#        n=1
#        while True:
#            Out += X
#            
#            XN = norm(X)
#            if XN<1e-30:
#                break
#    
#            X = (generator @ X)
#            X= X/(n+1)
#            n=n+1
#            
#        print(f"    done with solver initalization. Time spent: {B.toc(n=211,disp=0):.4} s")
#        print("")
#        return Out.reshape((self.Nphi,self.Nphi,2,2)),dt
        
    
    def get_steadystate(self,t0_array):
        """
        Compute steady states for t in t0list
        """
        
        
        ### Format input
        t0_array = array(t0_array)
        
        if len(shape(t0_array))==0:
            t0_vec = array([t0_array])
           
        elif len(shape(t0_array))==1:
            pass
        
        elif len(shape(t0_array))==4:
            t0_vec = t0_array.reshape(shape(t0_array)[1])
            
        else:
            raise ValueError("t0_array must be 1-d array with ndim 0,1, or 4")
            
        NT0 = len(t0_vec)

            
        ### Declare matrices 
        self.U0 = array([eye(2,dtype=complex) for x in range(0,NT0)])
        self.U0conj = 1*self.U0
        self.R0  =zeros((NT0,2,2),dtype=complex)
        self.W = 0

        self.s0_vec =arange(0,-log(1e6)*tau,-self.dt)

        self.ns=0
        self.NS=len(self.s0_vec)
        
        B.tic(n=20)
        print(f"Number of iterations to compute steady state : {self.NS*1e-6} million")
        for self.s0 in self.s0_vec:
            self.ns+=1
            
            self.s = t0_vec + self.s0
            
            if self.ns%(self.NS//10)==0:
                print(f"    progress: {int(self.ns/self.NS*100+0.1)} %. Time spent: {B.toc(n=20,disp=0):.4} s")
                self.Rnew = self.R0/self.W

                print(f"    current value : {abs(sum(self.Rnew)):.5}")
                sys.stdout.flush()            
            ind1,ind2 = self.get_ind(self.s-self.dt)
    
            self.R1 = self.U0@self.Rgrid[ind1,ind2]@self.U0conj* exp(self.s0/tau)*self.dt/tau
            self.R0 = self.R0 + self.R1
            
            self.U0 = self.U0@self.U_array[ind1,ind2]
            self.W += exp(self.s0/tau)*self.dt/tau
            self.U0conj = self.U0.conj().swapaxes(-1,-2)

        if isnan(sum(self.R0)):
            raise ValueError("Nan value encountered")
            
        return self.R0
       
        
    def get_ind(self,t):
        ind1 = (self.Nphi*t/T1+0.1*pi).astype(int)%self.Nphi
        ind2 = (self.Nphi*t/T2+0.1*pi).astype(int)%self.Nphi
        return (ind1.flatten(),ind2.flatten())
    
    def __iterate(self,t):
        """
        Evolve R from t to t+self.dt
        """
        ind1,ind2 = self.get_ind(t)

        self.dU = 1*self.U_array[ind1,ind2]
        
        self.R1 = exp(-self.dt/tau)*self.R 
        self.R2 = self.dU@self.R1@(self.dU.conj().swapaxes(-1,-2))
        self.dR =  self.dt/tau*self.Rgrid[ind1,ind2]
        self.dR1 = 0.5*(self.dR + self.dU@self.dR@(self.dU.conj().swapaxes(-1,-2)))
        self.R = self.R2+ self.dR1 #self.dt/tau*self.Rgrid[ind1,ind2]       
        
        
    def get_ft(self,freqlist,tmax,t0=0):
        """
        fourier transform $1/(tmax-t0)\int_t0^tmax \rho(t) e^{i\omega t}$ for 
        omega in freqlist
        """
        nfreqs = len(freqlist)
        
        if array(freqlist).dtype!=float:
            print("%"*80)
            print("WARNING: Frequencies given are not float values of meV. Make sure that this is intended")
            print("%"*80)
            

        # Parallelize
        if tmax/self.T1 < self.NMAT_MAX:
            self.nmat = int((tmax-t0)/self.T1)+1
            self.n_T1 = 1
            
        else:
            self.nmat = self.NMAT_MAX
            self.n_T1 = int((tmax-t0)/(self.T1*self.nmat)+1)

        self.freqlist = array(freqlist).reshape((nfreqs,1,1,1))
        
        
        npr.seed(0)
        self.t0_array = t0 + self.n_T1*self.T1 * (arange(self.nmat)+1000*npr.rand(self.nmat))
        self.t0_array = self.t0_array.reshape((1,self.nmat,1,1))
        
        self.Out = zeros((nfreqs,self.nmat,2,2),dtype=complex)
        print("Computing intial steady state");B.tic(n=123)
        self.R = self.get_steadystate(self.t0_array)
        print(f"Time domain solver done. Time spent: {B.toc(n=123,disp=0):.4}s")
        print("")
        
            
        print("Computing Fourier transform")
        B.tic(n=17);B.tic(n=18)
        
        # list used to monitor convergence, for diagnostics
        self.convergence_list= []
        
        self.nt=0
        self.delta_t_list = arange(0,self.n_T1*self.T1,self.dt)
        
        self.NDT = len(self.delta_t_list)
        print(f"Number of iterations to compute fourier transform: {self.NDT*1e-6:.4} million")
        print(f"Total time effectively integrated over: {self.n_T1*self.nmat} * T_1")
        for self.delta_t  in self.delta_t_list:
            self.t = self.t0_array + self.delta_t

            self.Out += exp(1j*self.freqlist*self.t)*self.R     

            self.__iterate(self.t)

            self.nt+=1

            if self.nt%(self.NDT//10) ==0:
                print(f"    progress: {int(self.nt/self.NDT*100+0.1)} %. Time spent: {B.toc(n=18,disp=False):.4} s")
                print(f"    current value : {sum(abs(self.Out))/(self.delta_t):.4}")
                sys.stdout.flush()
            if self.nt%(self.NDT//100) ==0:
                self.convergence_list.append(self.Out/(self.delta_t))
   
        self.Out = sum(self.Out,axis=1)/(self.n_T1*self.T1*self.nmat)*self.dt
        print(f"Done. Time spent: {B.toc(n=17,disp=0):.4} s")

        self.t_out  =  self.T1 * self.nmat * self.n_T1
        
        return self.Out,self.t_out
    

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
    vF     = 1e6*meter/second
    
    EF2 = 0.6*1.5*2e6*Volt/meter
    EF1 = 0.6*1.25*1.2e6*Volt/meter
    
    T1 = 2*pi/omega1
    
    Mu =0
    Temp  = 20*Kelvin;
    
    V0 = array([0,0,0.8*vF])
    [V0x,V0y,V0z] = V0
    
    ky=0;kx=0
    kzlist = linspace(-0.5,-0.3,5)
    kzlist = [-0.4427672955974843]

    klist = array([[kx,ky,kz] for kz in kzlist])

    parameters = 1*array([[omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp]]*len(klist))
    set_parameters(parameters[0])
    
    k =klist[0]
    DT = 0.0005
    S = time_domain_solver(k,Nphi=50)
    
    Out = S.get_ft([0,omega2,omega1],1000*T1)
    

    r1  = Out[0]
    
    [h00,h01,h10] = [wl.get_h_fourier_component(m,n,k) for (m,n) in [(0,0),(0,1),(1,0)]]

    Ess = 0
    
    Ess += real(trace(h00@r1[0]))
    Ess += 2*real(trace(h01.conj().T@r1[1]))
    Ess += 2*real(trace(h10.conj().T@r1[2]))
    
    [h00,h01,h10] = [wl.get_h_fourier_component(m,n,k) for (m,n) in [(0,0),(0,1),(1,0)]]
        

    [dh00,dh01,dh10] = [wl.get_dhdt_fourier_component(m,n,k) for (m,n) in [(0,0),(0,1),(1,0)]]
        

                
#    mode1_power_0    = 0
    mode1_power   = 2*real(trace(dh10.conj().T @ r1[2]))
#    mode1_power_2    = 2*real(trace(dh10.conj())*r2[2])
    
#    mode2_power_0    = 0
    mode2_power    = 2*real(trace(dh01.conj().T @ r1[1]))
#    mode2_power_2    = 2*real(trace(dh01.conj()) * r2[2])
    
#    mode1_power = mode1_power_0+mode1_power_1+mode1_power_2
#    mode2_power = mode2_power_0+mode2_power_1+mode2_power_2
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    