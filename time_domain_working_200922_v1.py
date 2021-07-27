#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 08:51:26 2020

@author: frederik

Solution of master equation in time domain
"""

from scipy import *
from matplotlib.pyplot import *
from Units import *
import weyl_liouvillian as wl
import Cython

omega1 = 20*THz
#omega1 = 2*pi
omega2 = 0.61803398875*omega1
#tau    = 50*picosecond
tau =  5*picosecond
vF   = 1e5*meter/second
T1 = 2*pi/omega1
T2 = 2*pi/omega2
EF1 = 0.6*1.5*2e6*Volt/meter
EF2 = 0.6*1.25*1.2e6*Volt/meter

Mu =15*meV*0
Temp  = 25*Kelvin;

V0 = array([0,0,0.8*vF])*1
[V0x,V0y,V0z] = V0


k = array([[0,0,0.01]])
parameters = [omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp]

wl.set_parameters(parameters)

h = wl.get_h(k)
hc = wl.get_hc(k)
r0 = wl.get_rhoeq(k)[1]
r0 = ve.mat_to_vec(r0)

class time_domain_solver():
    def __init__(self,k,Nphi=300,dtmax=inf):
        self.k = k
        self.Nphi = Nphi

        self.kgrid = wl.get_kgrid(self.Nphi,k0=self.k)
        self.dtmax = dtmax


        self.U_array,self.dt = self.get_uarray()
        self.Rgrid = wl.get_rhoeq(self.kgrid)[1].reshape((self.Nphi,self.Nphi,2,2))

    def get_uarray(self):
        """
        get array of e^{-ih(\phi_1,\phi_2)dt} for \phi_1,\phi_2 = (0,1,...N)*2pi/N
        """
                
        S = self.Nphi**2
        
        Out = array([eye(2,dtype=complex) for n in range(0,S)])# zeros((S,4,4),dtype=complex)

        h  =wl.get_h(self.kgrid)
        Amax = amax(norm(h,axis=(1,2)))
        

        dt =0.2/Amax
        dt = min(dt,self.dtmax)
#        Ratio = 
#        dt = T1/4*self.Nphi
    
    
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
    

        U0 = eye(2,dtype=complex)
        
    
        R0  =zeros((2,2),dtype=complex)
        

        svec =arange(t0,t0-32*tau,-self.dt)
        global ns,NS
        ns=0
        NS=len(svec)
        for s in svec:
            ns+=1
#            if ns%1e5==0:
#                print(f"{ns}/{NS}")
            ind1,ind2 = self.get_ind(s-self.dt)
    
            if isnan(R0[0,0]):
                raise ValueError
            R0  += U0@self.Rgrid[ind1,ind2]@(U0.conj().T)*self.dt*exp(-(t0-s)/tau)/tau
             
            dU = 1*self.U_array[ind1,ind2]
            U0 = U0@dU
            
            
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
        B.toc(n=17)
        Outlist= []
        nt=0
        tlist = arange(t0,tmax,self.dt)
        NT = len(tlist)
        for t  in tlist:
           self.xx = self.dt*exp(1j*freqlist*t)*self.R
           self.Out += self.xx#self.dt*exp(1j*freqlist*t)*R
           self.__iterate(t)
           
           nt+=1
           if nt%(NT//20) ==0:
               print(f"at time {t}/{tmax}")
               print(sum(abs(self.Out))/(t-t0))
               Outlist.append(self.Out/(t-t0))
           
        self.Out = self.Out/(t-t0)
        B.toc(n=17)
        Outlist= array(Outlist)
        return Out,Outlist 
    
        
B.tic()
Q=time_domain_solver(k[0],dtmax=T1/100)      
F1,Outlist = Q.get_ft([omega1,-omega1],1000*T1)
B.toc()

#print("With pyx")  
#
#import time_domain_pyx as tdp
#
#B.tic()
#Q=tdp.time_domain_solver(k[0],dtmax=T1/100)      
#F1,Outlist = Q.get_ft([omega1,-omega1],1000*T1)
#B.toc()
