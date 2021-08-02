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
from scipy.interpolate import griddata

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

filename_fds = "_1_210730_1611-07.994_0.npz"
filename_tds = "_1_210730_1611-15.161_0.npz"
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


phases    = tds_data["phases"]
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



def get_S_exact(phi):

# t = times[:,0]
    """
    Get exact bloch vector from frequency domain solutoin at phases specified in phi
    """
    nt = len(phi)
    phi1 = phi[:,0]
    phi2 = phi[:,1]
    
    phase1_array = exp(-1j*outer(phi1,f1))
    phase2_array = exp(-1j*outer(phi2,f2))
    
    out = sum(rho_f.reshape((1,nf1,nf2,3)) * phase1_array.reshape((nt,nf1,1,1))*phase2_array.reshape((nt,1,nf2,1)),axis=(1,2))
    assert amax(abs(imag(out)))<1e-5,f"Imaginary value of output too large: {amax(abs(imag(out)))}"

    return real(out)

def get_particle_density(t):
    """
    Get exact particle density from frequency domain solutoin at times specified in t
    """
    nt = len(phi)

    phi1 = phi[:,0]
    phi2 = phi[:,1]
    
    phase1_array = exp(-1j*outer(phi1,f1))
    phase2_array = exp(-1j*outer(phi2,f2))
    
    Dlist = [D0,D1,D2]
    out = 0
    
    for np in [1,2]:
        D = Dlist[np]
        
        out += np* sum(D.reshape((1,nf1,nf2)) * phase1_array.reshape((nt,nf1,1))*phase2_array.reshape((nt,1,nf2)),axis=(1,2))

    return real(out)

def get_A(phi):
    phi1 = phi[:,0]
    phi2 = phi[:,1]
    return wl.get_A(phi1,phi2).T

def get_energy(bv,density,phi):
    
    assert len(bv)==len(density)and len(bv)==len(phi),"arguments must have the same length"
    k_eff = get_A(phi) + k 
    E = 2*(vF * sum(k_eff*bv,axis=1)+V0@k_eff.T*density)
    assert amax(abs(imag(E)))<1e-10,"Imaginary value of E too large"
        # return E 1
    return real(E)

NT,NR = shape(phases)[:2]
skip = 5 

Esslist = []
Eeqlist = []

ind = 2

phi = phases[::skip,ind,:]

# t = t.reshape((nt*len(ind)),order="F")
bv_tds = evolution[:,ind,:].reshape((nt,3),order="F")[::skip,:]
bv_fds = get_S_exact(phi)
density = get_particle_density(phi)
diff = norm(bv_fds-bv_tds,axis=1)


### Get steady-state energy

# ns = shape()




kgrid = get_A(phi) + k 
R0eq,R1eq,R2eq = wl.get_rhoeq(kgrid,mu=mu)
H           = wl.get_h(kgrid)

E0 = 0
E1 = trace(H@R1eq,axis1=1,axis2=2)
E2 = R2eq*trace(H,axis1=1,axis2=2)

Eeq = real((E0+E1+E2))

# def get_equilibrium_energy(times):
    
    
E_tds = get_energy(bv_tds,density,phi)
E_fds = get_energy(bv_fds,density,phi)# E2 = 

Eeqlist.append(E_fds)
Esslist.append(Eeq)

    # timelist.append(t)

    
skip = 10
# Interpolate values to grid
bv   = evolution.reshape((nr*nt,3))
phi1 = mod(phases[:,:,0].flatten(),2*pi)
phi2 = mod(phases[:,:,1].flatten(),2*pi)

# raise ValueError
nphi = 200

ext_indices = where(logical_or(phi1>2*pi*(1-4/nphi),phi2>2*pi*(1-4/nphi)))[0]
phi1 = concatenate((phi1,phi1[ext_indices]-2*pi))
phi2 = concatenate((phi2,phi2[ext_indices]))
phi1 = concatenate((phi1,phi1[ext_indices]))
phi2 = concatenate((phi2,phi2[ext_indices]-2*pi))
phi1 = concatenate((phi1,phi1[ext_indices]-2*pi))
phi2 = concatenate((phi2,phi2[ext_indices]-2*pi))
bv   = concatenate((bv,bv[ext_indices],bv[ext_indices],bv[ext_indices]))
phirange=  arange(0,nphi)/nphi * 2*pi
phi1_g,phi2_g = meshgrid(phirange,phirange)


dphi = phirange[1]-phirange[0]
#%%
figure(1)
outgrid = griddata((phi1[::skip],phi2[::skip]),bv[::skip,:],(phi1_g,phi2_g),fill_value=0,method="linear")
pcolormesh(phi1_g,phi2_g,outgrid[:,:,2])
# plot(phi1,phi2,'.w',markersize=0.1)
xlim((0,2*pi))
ylim((0,2*pi))
colorbar()



figure(2)
out = get_S_exact(array((phi1_g.flatten(),phi2_g.flatten())).T).reshape((nphi,nphi,3))
pcolormesh(phi1_g,phi2_g,out[:,:,2])
colorbar()

figure(3)
pcolormesh(phi1_g,phi2_g,out[:,:,0]-outgrid[:,:,0])
colorbar()

figure(4)
plot(phi1,phi2,'.',markersize=0.3)
xlim(0,2*pi)
ylim(0,2*pi)
def get_fourier_component(values,m,n):
    
    phasemat = exp(1j*m*phi1_g+1j*n*phi2_g).reshape((nphi,nphi,1))
    fourier_coeff = sum(phasemat * values,axis=(0,1))*dphi**2/(4*pi**2)
    
    return fourier_coeff

a = get_fourier_component(out, 0,0)
b = get_fourier_component(outgrid, 0,0)    
    

def check_fourier_match(m,n):
    # a = get_fourier_component(out, m,n) 
    a = B.get_bloch_vector(fourier_coeffs[f1==m,f2==n][0])
    b = get_fourier_component(outgrid, m,n)
    
    
    
    print(f"Exact                : {around(a,4)}")
    print(f"TDS                  : {around(b,4)}")
    print(f"Difference           : {norm(a-b):.4}")
    print(f"Relative difference  : {norm((a-b))/norm(a):.4} ")
    
check_fourier_match(0,0)
    
# plot(phi1,phi2,'.')
# # =============================================================================
# # Plotting
# # =============================================================================
# close("all")
# phi1 = phi[:,0]
# phi2 = phi[:,1]
# figure(1)
# plot(phi1,bv_fds,'.-')
# plot(phi1,bv_tds,'.-')  
# title("Evolution of Bloch vector vs. time",fontsize=8)
# legend(["x, exact","y, exact","z,exact","x, tds","y,tds","z,tds"],fontsize=6)
# xlabel("\phi1")

# figure(2)
# plot(phi1,diff,".-")
# title("Difference in bloch vector between tds and exact result",fontsize = 6)
# xlabel("Time")
# ylabel("$|v_{\\rm f}-v_{\\rm t}|$")
# # diff = 

# figure(3)
# plot(phi1,density)
# title("Particle density vs time")
# ylim(0,2.1)
# xlabel("time")

# figure(4)
# plot(phi1,E_fds)
# plot(phi1,E_tds)
# plot(phi1,Eeq)
# title("Energy vs. time")
# legend(["Steadystate energy, fds","Steadystate energy, tds","Equilibrium energy"],fontsize=6)
# # energy_difference = 
# xlabel("time")

# # figure(5)
# # plot(mod(phases[:,:10,0].flatten(),2*pi),mod(phases[:,:10,1].flatten(),2*pi),'.')



















