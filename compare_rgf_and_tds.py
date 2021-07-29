#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 18:17:46 2021

@author: frederiknathan

Script comparing time domain solutions with frequency domain solutions
"""

import os 
import sys 
from scipy.linalg import * 
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import * 
from numpy.fft import *
import numpy.random as npr
import scipy.optimize as optimize
import gc

import basic as B
from units import *
import weyl_liouvillian as wl
import recursive_greens_function as rgf
import time_domain_solver as tds
from matplotlib.pyplot import *


data_tds = array([   0.3658    ,    2.68029025,    0.        , -220.17980461,
       -221.04790799,   -3.04609025])

data_fds = array([-3.37792409e-03, -1.48520376e-02,  5.55111512e-17, -2.20179805e+02,
       -2.20172265e+02,  1.82299616e-02])

Ess = data_tds[4]
Eeq = data_tds[3]
"""
data = array([P1,P2,density,Eeq,Ess,work])
"""

fds_dir = "../Frequency_domain_solutions/"
tds_dir = "../Time_domain_solutions/"

filename_fds = "_1_210729_1141-49.414_0.npz"
filename_tds = "_1_210729_1113-42.674_0.npz"
fds_data = load(fds_dir+filename_fds)
tds_data = load(tds_dir+filename_tds)

pt = tds_data["parameters"]
pf = fds_data["parameters"]

kt = tds_data["k"]
kf = tds_data["k"]
k = kf
assert amax(abs(pt-pf))<1e-10,"parameters of two solutions dont match"
assert amax(abs(kt-kf))<1e-10,"momenta of two solutions dont match"

omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,mu,Temp = pt
V0=array([V0x,V0y,V0z])
wl.set_parameters(pt)


times = tds_data["times"]
evolution = tds_data["evolution_record"]
(nt,nr,Null)=shape(evolution)
f1,f2 = [fds_data[x] for x in ["freq_1","freq_2"]]
nf1 = len(f1)
nf2 = len(f2)
fourier_coeffs = fds_data["fourier_coefficients_1p"]
D0 = fds_data["fourier_coefficients_0p"]
D2 = fds_data["fourier_coefficients_2p"]
# extract Bloch vector from fourier coeffs

SX = 0.5*(fourier_coeffs[:,:,0,1]+fourier_coeffs[:,:,1,0])
SY = 0.5*1j*(fourier_coeffs[:,:,0,1]-fourier_coeffs[:,:,1,0])
SZ = 0.5*(fourier_coeffs[:,:,0,0]-fourier_coeffs[:,:,1,1])
D1 = (fourier_coeffs[:,:,0,0]+fourier_coeffs[:,:,1,1])

rho_f = array([SX,SY,SZ]).swapaxes(0,1).swapaxes(1,2)



def get_S_exact(t):

# t = times[:,0]
    """
    Get exact bloch vector from frequency domain solutoin at times specified in t
    """
    nt = len(t)
    phase1_array = exp(-1j*outer(t,f1)*omega1)
    phase2_array = exp(-1j*outer(t,f2)*omega2)
    
    out = sum(rho_f.reshape((1,nf1,nf2,3)) * phase1_array.reshape((nt,nf1,1,1))*phase2_array.reshape((nt,1,nf2,1)),axis=(1,2))
    assert amax(abs(imag(out)))<1e-10,"Imaginary value of output too large"

    return real(out)

def get_particle_density(t):
    """
    Get exact particle density from frequency domain solutoin at times specified in t
    """
    nt = len(t)
    phase1_array = exp(-1j*outer(t,f1)*omega1)
    phase2_array = exp(-1j*outer(t,f2)*omega2)
    
    Dlist = [D0,D1,D2]
    out = 0
    
    for np in [1,2]:
        D = Dlist[np]
        
        out += np* sum(D.reshape((1,nf1,nf2)) * phase1_array.reshape((nt,nf1,1))*phase2_array.reshape((nt,1,nf2)),axis=(1,2))

    return real(out)

def get_A(t):
    return wl.get_A(omega1*t,omega2*t).T

def get_energy(bv,density,times):
    
    assert len(bv)==len(density)and len(bv)==len(times),"arguments must have the same length"
    k_eff = get_A(times) + k 
    E = 2*(vF * sum(k_eff*bv,axis=1)+V0@k_eff.T*density)
    assert amax(abs(imag(E)))<1e-10,"Imaginary value of E too large"
        # return E 1
    return real(E)

NT,NR = shape(times)
skip = 5 

Esslist = []
Eeqlist = []
t=  []
# for ind in range(0,NR):
    # print(ind)
ind = array([1])
t = times[::skip,ind]
nt = len(t)
t = t.reshape((nt*len(ind)),order="F")
bv_tds = evolution[::skip,ind,:].reshape((nt*len(ind),3),order="F")

bv_fds = get_S_exact(t)
density = get_particle_density(t)
diff = norm(bv_fds-bv_tds,axis=1)


### Get steady-state energy

# ns = shape()




kgrid = get_A(t) + k 
R0eq,R1eq,R2eq = wl.get_rhoeq(kgrid,mu=mu)
H           = wl.get_h(kgrid)

E0 = 0
E1 = trace(H@R1eq,axis1=1,axis2=2)
E2 = R2eq*trace(H,axis1=1,axis2=2)

Eeq = real((E0+E1+E2))

# def get_equilibrium_energy(times):
    
    
E_tds = get_energy(bv_tds,density,t)
E_fds = get_energy(bv_fds,density,t)# E2 = 

Eeqlist.append(E_fds)
Esslist.append(Eeq)

    # timelist.append(t)

    

# =============================================================================
# Plotting
# =============================================================================

figure(1)
plot(t,bv_fds,'-')
plot(t,bv_tds,'-')  
title("Evolution of Bloch vector vs. time",fontsize=8)
legend(["x, exact","y, exact","z,exact","x, tds","y,tds","z,tds"],fontsize=6)
xlabel("time")

figure(2)
plot(t,diff)
title("Difference in bloch vector between tds and exact result",fontsize = 6)
xlabel("Time")
ylabel("$|v_{\\rm f}-v_{\\rm t}|$")
# diff = 

figure(3)
plot(t,density)
title("Particle density vs time")
ylim(0,2.1)
xlabel("time")

figure(4)
plot(t,E_fds)
plot(t,E_tds)
plot(t,Eeq)
title("Energy vs. time")
legend(["Steadystate energy, fds","Steadystate energy, tds","Equilibrium energy"],fontsize=6)
# energy_difference = 
xlabel("time")



















