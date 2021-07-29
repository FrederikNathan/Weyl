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

"""
data = array([P1,P2,density,Eeq,Ess,work])
"""

fds_dir = "../Frequency_domain_solutions/"
tds_dir = "../Time_domain_solutions/"

filename_fds = "_1_210729_1114-06.037_0.npz"
filename_tds = "_1_210729_1113-42.674_0.npz"
fds_data = load(fds_dir+filename_fds)
tds_data = load(tds_dir+filename_tds)

pt = tds_data["parameters"]
pf = fds_data["parameters"]

kt = tds_data["k"]
kf = tds_data["k"]
assert amax(abs(pt-pf))<1e-10,"parameters of two solutions dont match"
assert amax(abs(kt-kf))<1e-10,"momenta of two solutions dont match"

omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,mu,Temp = pt



times = tds_data["times"]
evolution = tds_data["evolution_record"]
(nt,nr,Null)=shape(evolution)
f1,f2 = [fds_data[x] for x in ["freq_1","freq_2"]]
nf1 = len(f1)
nf2 = len(f2)
fourier_coeffs = fds_data["fourier_coefficients"]

# extract Bloch vector from fourier coeffs

SX = 0.5*(fourier_coeffs[:,:,0,1]+fourier_coeffs[:,:,1,0])
SY = 0.5*1j*(fourier_coeffs[:,:,0,1]-fourier_coeffs[:,:,1,0])
SZ = 0.5*(fourier_coeffs[:,:,0,0]-fourier_coeffs[:,:,1,1])

rho_f = array([SX,SY,SZ]).swapaxes(0,1).swapaxes(1,2)

ind = array([0])
t = times[:,ind].reshape((nt*len(ind)),order="F")
out_tds = evolution[:,ind,:].reshape((nt*len(ind),3),order="F")
def get_S_exact(t):

# t = times[:,0]
    """
    Get exact bloch vector from frequency domain solutoin at times specified in t
    """
    nt = len(t)
    phase1_array = exp(-1j*outer(t,f1)*omega1)
    phase2_array = exp(-1j*outer(t,f2)*omega2)
    
    out = sum(rho_f.reshape((1,nf1,nf2,3)) * phase1_array.reshape((nt,nf1,1,1))*phase2_array.reshape((nt,1,nf2,1)),axis=(1,2))

    return out 


out = get_S_exact(t)

diff = norm(out-out_tds,axis=1)

figure(1)
plot(t,real(out),'-')
plot(t,out_tds,'-')  
title("Evolution of Bloch vector vs. time",fontsize=8)
legend(["x, exact","y, exact","z,exact","x, tds","y,tds","z,tds"],fontsize=6)

figure(2)
plot(t,diff)
title("Difference in bloch vector between tds and exact result",fontsize = 6)
xlabel("Time")
ylabel("$|v_{\\rm f}-v_{\\rm t}|$")
# diff = 























