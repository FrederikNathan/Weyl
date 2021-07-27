#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 08:51:26 2020

@author: frederik

Solution of master equation in time domain

v1: speedup computing of steady states and fourier transform. 
"""
import os 
NUM_THREADS = 1
RELATIVE_DT_MAX = 1    # Maximum dt relative to 1/|H|.
STEADY_STATE_RELATIVE_TMAX = 14 # upper bound of integration for steady state, in units of tau.
                                      # i.e. relative uncertainty of steady state = e^{-STEADY_STATE_RELATIVE_TMAX}

os.environ["OMP_NUM_THREADS"]        = str(NUM_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(NUM_THREADS)
os.environ["NUMEXPR_NUM_THREADS"]    = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"]        = str(NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"]   = str(NUM_THREADS)

from scipy import *
from Units import *
import weyl_liouvillian_v1 as wl

class time_domain_solver():
    def __init__(self,k,Nphi=600,dtmax=inf):
        self.k = k
        self.Nphi = Nphi

        self.T1 = 2*pi/omega1
        self.T2 = 2*pi/omega2
        
        self.kgrid = wl.get_kgrid(self.Nphi,k0=self.k)
        self.dtmax = dtmax


        self.U_array,self.dt = self.get_uarray()
        self.Rgrid = wl.get_rhoeq(self.kgrid,mu=Mu)[1].reshape((self.Nphi,self.Nphi,2,2))

        self.NMAT_MAX = 50
        
    def get_uarray(self):
        """
        get array of e^{-ih(\phi_1,\phi_2)dt} for \phi_1,\phi_2 = (0,1,...N)*2pi/N
        
        also computes the crucial dt variable
        """
                
        S = self.Nphi**2
        
        Out = array([eye(2,dtype=complex) for n in range(0,S)])# zeros((S,4,4),dtype=complex)

        h  =wl.get_h(self.kgrid)
        Amax = amax(norm(h,axis=(1,2)))
        
        dt = min(RELATIVE_DT_MAX/Amax,self.dtmax,self.T1/self.Nphi,self.T2/((sqrt(2)-0.2)*self.Nphi),log(1e6)*tau/600)
    
        generator = -1j*h*dt
    
        X = 1*generator 
        
        n=1
        while True:
            Out += X
            
            XN = norm(X)
            if XN<1e-30:
                break
    
            X = (generator @ X)
            X= X/(n+1)
            n=n+1
            

        return Out.reshape((self.Nphi,self.Nphi,2,2)),dt
        
    
    def get_steadystate(self,t0_array):
        """
        Compute steady states for t in t0list
        """
#        global NT0,U0,s,tt,R0,ind1,ind2,s0,TA,TB
        
        t0_array = array(t0_array)
        
        if len(shape(t0_array))==0:
            t0_array = array([t0_array])
        
        NT0 = len(t0_array)
        t0_array = t0_array.reshape((NT0,1,1))

            
        
        tt =t0_array
        self.U0 = array([eye(2,dtype=complex) for x in range(0,NT0)])
        self.U0conj = 1*self.U0
    
        R0  =zeros((NT0,2,2),dtype=complex)
        
        Rt = 1*R0

        s0_vec =arange(0,-log(1e6)*tau,-self.dt)

        self.ns=0
        self.NS=len(s0_vec)
        B.tic(n=20)
        W = 0
#        TA=0
#        TB=0
        for s0 in s0_vec:
            self.ns+=1
            
            s = t0_array + s0
            
            if self.ns%(self.NS//10)==0:
                print(f"    progress: {int(self.ns/self.NS*100)+1} %. Time spent: {B.toc(n=20,disp=0):.4} s")
                Rnew = R0/W
                print(f"    current value : {abs(sum(Rnew)):.5}, {mean(W):.5}")
#                print("")
#                print(exp(s0/tau))
#                
#                print(real(trace(self.Rgrid[ind1,ind2].reshape((2,2))))* exp(s0/tau)*self.dt/tau)
#                print(real(trace(self.R1.reshape((2,2)))))
                
            ind1,ind2 = self.get_ind(s-self.dt)
    
            if isnan(R0[0,0,0]):
                raise ValueError
#            print("")
#            B.tic();B.tic(n=71)
#            print(norm(self.Rgrid[ind1,ind2]))
            self.R1 = self.U0@self.Rgrid[ind1,ind2]@self.U0conj* exp(s0/tau)*self.dt/tau
            R0 = R0 + self.R1
            
            
#            T+=trace(self.R1.reshape((2,2)))#trace(R0.reshape((2,2)))#trace(self.Rgrid[ind1,ind2].reshape((2,2)))* exp(s0/tau)*self.dt/tau
            
#            TA+=trace(self.Rgrid[ind1,ind2].reshape((2,2)))* exp(s0/tau)*self.dt/tau
#            TB+=trace(self.R1.reshape((2,2)))
#            B.toc();B.tic()
            W+=self.dt/tau*exp(s0/tau)

#            B.toc();B.tic()
            self.U0 = self.U0@self.U_array[ind1,ind2]
            
#            B.toc();B.tic()
            self.U0conj = self.U0.conj().swapaxes(-1,-2)
#            B.toc()
#            print(f"Total time: {B.toc(n=71,disp=0)*1e3:.4} ms")
            
        return R0
       
        
    def get_ind(self,t):
        ind1 = (self.Nphi*t/T1+0.1*pi).astype(int)%self.Nphi
        ind2 = (self.Nphi*t/T2+0.1*pi).astype(int)%self.Nphi
        return (ind1.flatten(),ind2.flatten())
    
    def evolve(self,R0,t0,t1):
        R=1*R0
        for t in arange(t0,t1,self.dt):
            R = self.__iterate(R,t)


        return R,t
    def __iterate(self,t):
        ind1,ind2 = self.get_ind(t)

        self.dU = 1*self.U_array[ind1,ind2]
        
        self.R1 = exp(-self.dt/tau)*self.R 
        self.R2 = self.dU@self.R1@(self.dU.conj().swapaxes(-1,-2))
        self.R = self.R2+ self.dt/tau*self.Rgrid[ind1,ind2]
    
#        return self.R3
    
    
    def get_ft(self,freqlist,tmax,t0=0):
        """
        fourier transform $1/(tmax-t0)\int_t0^tmax \rho(t) e^{i\omega t}$ for 
        omega in freqlist
        """
        global t0_array,ff,t
        nfreqs = len(freqlist)
        
        self.Out = zeros((nfreqs,2,2),dtype=complex)
        

        if tmax/self.T1 < self.NMAT_MAX:
            nmat = int((tmax-t0)/self.T1)+1
            n_T1 = 1
            
        else:
            nmat = self.NMAT_MAX
            n_T1 = int((tmax-t0)/(self.T1*nmat)+1)
            
            
        freqlist = array(freqlist).reshape((nfreqs,1,1,1))
        ff = freqlist
        t0_array = n_T1*self.T1 * arange(0,nmat)+t0
        
        
        print("Computing intial steady state");B.tic(n=123)
        self.R = self.get_steadystate(t0_array)
        print(f"Time domain solver done. Time spent: {B.toc(n=123,disp=0):.4}s")
        print("")
        print("Computing Fourier transform")



        B.tic(n=17);B.tic(n=18)
        Outlist= []
        nt=0
        delta_t_list = arange(0,n_T1*self.T1,self.dt)
        
        N_delta_t = len(delta_t_list)
        for delta_t  in delta_t_list:
            t = t0_array + delta_t
            
            t = t.reshape((1,nmat,1,1))
            self.Out += sum(exp(1j*freqlist*t)*self.R,axis=1)*self.dt/nmat
            
            self.__iterate(t)
            self.t = t 
            nt+=1
            if nt%(N_delta_t//10) ==0:
                print(f"    progress: {int(nt/N_delta_t*100+1)} %. Time spent: {B.toc(n=18,disp=False):.4} s")
                print(f"    current value : {sum(abs(self.Out))/(delta_t):.4}")
#                print("")
                Outlist.append(self.Out/(delta_t))
   
        self.Out = self.Out/(n_T1*self.T1)
        print(f"Done. Time spent: {B.toc(n=17,disp=0):.4} s")
        
        t_out  =  self.T1 * nmat * n_T1
        
        return self.Out,t_out
    

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
        
#B.tic()
#Q=time_domain_solver(k[0],dtmax=T1/100)      
#F1,Outlist = Q.get_ft([omega1,-omega1],1000*T1)
#B.toc()

#print("With pyx")  
#
#import time_domain_pyx as tdp
#
#B.tic()
#Q=tdp.time_domain_solver(k[0],dtmax=T1/100)      
#F1,Outlist = Q.get_ft([omega1,-omega1],1000*T1)
#B.toc()


if __name__=="__main__":
    omega2 = 20*THz
    omega1 = 0.61803398875*omega2
    tau    = 5*picosecond
    vF     = 1e6*meter/second
    
    T1 = 2*pi/omega1
    
    EF2 = 0.6*1.5*2e6*Volt/meter
    EF1 = 0.6*1.25*1.2e6*Volt/meter
    
    Mu =300
    Temp  = 0.1*Mu;
    
    V0 = array([0,0,0.8*vF])*1
    [V0x,V0y,V0z] = V0
    
    
    k=array([ 0.        ,  0.        , 0.06086957])
    
    parameters = array([omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp])

    set_parameters(parameters)
    S = time_domain_solver(k,Nphi=30)
#    R0 = S.get_steadystate([0,1,2,3,4])
    R = S.get_steadystate(0)
#    S.get_ft([0,1,2],10*T1)
    
#    print(R0)
#    R0,Outlist = S.get_ft([0],1000*T1)
    
    import time_domain_solver as tds_old
    
    tds_old.set_parameters(parameters)
    S1 = tds_old.time_domain_solver(k,Nphi=300)
#    R1,Outlist1 = S1.get_ft([0,1,2],1*T1)
#    R1 = S1.get_steadystate(0)
    