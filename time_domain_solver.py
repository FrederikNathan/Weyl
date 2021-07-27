#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 08:51:26 2020

@author: frederik

Solution of master equation in time domain
"""

from scipy import *
from Units import *
import weyl_liouvillian as wl

#omega1 = 20*THz
##omega1 = 2*pi
#omega2 = 0.61803398875*omega1
##tau    = 50*picosecond
#tau =  5*picosecond
#vF   = 1e5*meter/second
#T1 = 2*pi/omega1
#T2 = 2*pi/omega2
#EF1 = 0.6*1.5*2e6*Volt/meter
#EF2 = 0.6*1.25*1.2e6*Volt/meter
#
#Mu =15*meV*0
#Temp  = 25*Kelvin;
#
#V0 = array([0,0,0.8*vF])*1
#[V0x,V0y,V0z] = V0
#
#
#k = array([[0,0,0.01]])
#parameters = [omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp]
#
#wl.set_parameters(parameters)
#
#h = wl.get_h(k)
#hc = wl.get_hc(k)
#r0 = wl.get_rhoeq(k)[1]
#r0 = ve.mat_to_vec(r0)

class time_domain_solver():
    def __init__(self,k,Nphi=1000,dtmax=inf):
        self.k = k
        self.Nphi = Nphi
        
    
        self.T1 = 2*pi/omega1
        self.T2 = 2*pi/omega2

        self.kgrid = wl.get_kgrid(self.Nphi,k0=self.k)
        self.dtmax = dtmax


        self.U_array,self.dt = self.get_uarray()
        self.Rgrid = wl.get_rhoeq(self.kgrid,mu=Mu)[1].reshape((self.Nphi,self.Nphi,2,2))

    def get_uarray(self):
        """
        get array of e^{-ih(\phi_1,\phi_2)dt} for \phi_1,\phi_2 = (0,1,...N)*2pi/N
        """
                
        S = self.Nphi**2
        
        Out = array([eye(2,dtype=complex) for n in range(0,S)])# zeros((S,4,4),dtype=complex)

        h  =wl.get_h(self.kgrid)
        Amax = amax(norm(h,axis=(1,2)))
        

#        dt =
        dt = min(0.1/Amax,self.dtmax,self.T1/self.Nphi,self.T2/self.Nphi)

    
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
        
    
    def get_steadystate(self,t0):
        print("Computing steady state")

        U0 = eye(2,dtype=complex)
        
    
        R0  =zeros((2,2),dtype=complex)
        
        Rt = 1*R0

        svec =arange(t0,t0-log(1e3)*tau,-self.dt)
        global ns,NS
        ns=0
        NS=len(svec)
        B.tic(n=20)
        W = 0
        for s in svec:
            ns+=1
            if ns%(NS//100)==0:
                print(f"    at step {ns}/{NS}. Time spent: {B.toc(n=20,disp=0):.4} s")
                Rnew = R0/W
                print(f"   current value:")
                print(Rnew,W)
                print("")
                
            ind1,ind2 = self.get_ind(s-self.dt)
    
            if isnan(R0[0,0]):
                raise ValueError
                
#            print("")
#            B.tic()
            R0  += U0@self.Rgrid[ind1,ind2]@(U0.conj().T)*self.dt*exp(-(t0-s)/tau)/tau
#            B.toc()
            W+=self.dt/tau*exp(-(t0-s)/tau)
#            B.toc()
            dU = 1*self.U_array[ind1,ind2]
#            B.toc()
            U0 = U0@dU
#            B.toc()
            
            
        return R0
       
        
    def get_ind(self,t):
        ind1 = int(self.Nphi*t/T1+0.1*pi)%self.Nphi
        ind2 = int(self.Nphi*t/T2+0.1*pi)%self.Nphi
        return (ind1,ind2)
    
    def evolve(self,R0,t0,t1):
        R=1*R0
        for t in arange(t0,t1,self.dt):
            R = self.__iterate(R,t)


        return R,t
    def __iterate(self,t):
        ind1,ind2 = self.get_ind(t)

        self.dU = 1*self.U_array[ind1,ind2]
        
        self.R1 = exp(-self.dt/tau)*self.R 
        self.R2 = self.dU@self.R1@(self.dU.conj().T)
        self.R = self.R2+ self.dt/tau*self.Rgrid[ind1,ind2]
    
#        return self.R3
    
    
    def get_ft(self,freqlist,tmax,t0=0):
        """
        fourier transform $1/(tmax-t0)\int_t0^tmax \rho(t) e^{i\omega t}$ for 
        omega in freqlist
        """
        B.tic(n=17)
        nfreqs = len(freqlist)
        self.Out = zeros((nfreqs,2,2),dtype=complex)
        
        freqlist = array(freqlist).reshape((nfreqs,1,1))
        ff=freqlist
        self.R = self.get_steadystate(t0)
        print("Initialization complete")
        B.toc(n=17)
        B.tic(n=17);B.tic(n=18)
        Outlist= []
        nt=0
        tlist = arange(t0,tmax,self.dt)
        NT = len(tlist)
        for t  in tlist:
           self.xx = self.dt*exp(1j*freqlist*t)*self.R
           self.Out += self.xx#self.dt*exp(1j*freqlist*t)*R
           self.__iterate(t)
           self.t = t 
           nt+=1
           if nt%(NT//10) ==0:
               print(f"    at time {t:.4}/{tmax:.1}. Time spent: {B.toc(n=18,disp=False):.4} s")
               print(f"    Current value of output: {sum(abs(self.Out))/(t-t0):.4}")
               Outlist.append(self.Out/(t-t0))
           
        self.Out = self.Out/(t-t0)
        B.toc(n=17)
        t_out = t
        return self.Out,t
    

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
    tau    = 50*picosecond
    vF     = 1e6*meter/second
    
    T1 = 2*pi/omega1
    
    EF2 = 0.6*1.5*2e6*Volt/meter
    EF1 = 0.6*1.25*1.2e6*Volt/meter
    
    Mu =20*meV*10
    Temp  = 0.1*Mu;
    
    V0 = array([0,0,0.8*vF])*1
    [V0x,V0y,V0z] = V0
    
    
    k=array([ 0.        ,  0.        , 0.06086957])
    
    parameters = array([omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp])

    set_parameters(parameters)
    S = time_domain_solver(k)
    
    R0,Outlist = S.get_ft([0],1*T1)