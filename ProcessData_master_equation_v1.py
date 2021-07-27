#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Obsolete plotting script
"""

raise NotImplementedError("Script is obsolete")
plot = 1 
savefig = 0

import os 


from scipy import *
from scipy.linalg import * 
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy import * 

from numpy.fft import *
import vectorization as ve
import numpy.random as npr
import scipy.optimize as optimize
import basic as B
import Master_equation_multimode_v5 as MA

import time as time 
from matplotlib.pyplot import *

import scipy.interpolate as scint
# import pandas as pan
from Units import *
import kgrid as kgrid 
from DataProcessingMasterEquation import *



def compute_power(nparm,disp=True):
    global kpoints,P1_vec,P2_vec,Dens_vec
    parameters,kpoints,P1_vec,P2_vec,Dens_vec,Eeq,Ess,TDS = get_data(nparm,disp=disp)
    
    NP = len(P1_vec)
    
    [omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp]=parameters
    
    
    P0=omega1*omega2/(2*pi)
    
    center = (0,0,0)
    cubewidth = 0.2/Å
    ncubes = (1,1,1)
    order=6
    
    Data = array([P1_vec,P2_vec,Dens_vec]).T
    (kx,ky,kz),Ng,Data_grid = kgrid.compute_grid(kpoints,Data,center,cubewidth,ncubes,order=order)
    
    P1g = Data_grid[:,:,:,0]
    P2g = Data_grid[:,:,:,1]
    Dg = Data_grid[:,:,:,2]
    
    #raise ValueError
    #(kx,ky,kz),Ng,P2g = kgrid.compute_grid(kpoints,P2,center,cubewidth,ncubes,order=order)
    #(kx,ky,kz),Ng,Dg = kgrid.compute_grid(kpoints,Dens,center,cubewidth,ncubes,order=order)
    
    (Nx,Ny,Nz) = (len(q) for q in (kx,ky,kz))
    (dkx,dky,dkz)=(q[1]-q[0] for q in (kx,ky,kz)) 
    
    
    xg = kron(kron(kx,ones(Ny)),ones(Nz)).reshape(Nx,Ny,Nz)
    yg = kron(kron(ones(Nx),ky),ones(Nz)).reshape(Nx,Ny,Nz)
    zg = kron(kron(ones(Nx),ones(Ny)),kz).reshape(Nx,Ny,Nz)
    
    Power_1=sum(P1g)*dkx*dky*dkz
    Power_2=sum(P2g)*dkx*dky*dkz
    Density = sum(Dg)*dkx*dky*dkz
    
    if disp:
#        print("-"*80+"\n")
        print(f"Density           : {Density/(centimeter**(-3)):<10.4} /cm**3\n")
        print(f"Energy pumping: \n    Mode 1        : {Power_1/(Joule/second/centimeter**3):<10.2} W/cm**3\n    Mode 2        : {Power_2/(Joule/second/centimeter**3):<10.2}")
        print("\n"+"="*80)
    
    
    return Power_1,Power_2,Density


#    print("Power in xz plane:")
#    print(sum(P1g[:,A0,:])*dkx*dkz)
#    print(sum(P2g[:,A0,:])*dkx*dkz)

    
# =============================================================================
# Plot 
# =============================================================================



def plotdata(nparm):
    
    parameters,kpoints,P1,P2,Dens = get_data(nparm)
    
    NP = len(P1)
    
    [omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp]=parameters
    
    
    P0=omega1*omega2/(2*pi)
    
    center = (0,0,0)
    cubewidth = 0.2/Å
    ncubes = (1,1,1)
    order=6
    
    Data = array([P1,P2,Dens]).T
    (kx,ky,kz),Ng,Data_grid = kgrid.compute_grid(kpoints,Data,center,cubewidth,ncubes,order=order)
    
    P1g = Data_grid[:,:,:,0]
    P2g = Data_grid[:,:,:,1]
    Dg = Data_grid[:,:,:,2]
    
    #raise ValueError
    #(kx,ky,kz),Ng,P2g = kgrid.compute_grid(kpoints,P2,center,cubewidth,ncubes,order=order)
    #(kx,ky,kz),Ng,Dg = kgrid.compute_grid(kpoints,Dens,center,cubewidth,ncubes,order=order)
    
    (Nx,Ny,Nz) = (len(q) for q in (kx,ky,kz))
    (dkx,dky,dkz)=(q[1]-q[0] for q in (kx,ky,kz)) 
    
    
    xg = kron(kron(kx,ones(Ny)),ones(Nz)).reshape(Nx,Ny,Nz)
    yg = kron(kron(ones(Nx),ky),ones(Nz)).reshape(Nx,Ny,Nz)
    zg = kron(kron(ones(Nx),ones(Ny)),kz).reshape(Nx,Ny,Nz)
    
    FigID = B.ID_gen()
    A0  = argmin(abs(ky))
    ky0 = ky[A0]
    
    
    figure(1,figsize=(7,7))
    Vmax = amax(abs(P1g[:,A0,:])/P0)   
    Vmax=max(Vmax,1)
    pcolormesh(xg[:,A0,:],zg[:,A0,:],P1g[:,A0,:]/P0,cmap="bwr",vmin=-Vmax,vmax=Vmax)          
    colorbar()
    ax=gca()
    ax.set_aspect(1)
    title(f"Energy pumped into mode 1, in units of $\hbar\omega_1\omega_2/2\pi$.\n$k_y$ = {ky0:.3}"+" ${\\rm Å}^{-1}$")
    xlabel("kx [${\\rm Å}^{-1}$]")
    ylabel("kz [${\\rm Å}^{-1}$]")
    plot([0],[0],'xk')
    
    if savefig:
        
        savefig(f"../Figures/Parm{nparm}_{FigID}_P1.png",dpi=200)
    
    
    
    figure(2,figsize=(7,7))
    Vmax = amax(abs(P2g[:,A0,:])/P0)   
    Vmax=max(Vmax,1)
    pcolormesh(xg[:,A0,:],zg[:,A0,:],P2g[:,A0,:]/P0,cmap="bwr",vmin=-Vmax,vmax=Vmax)          
    colorbar()
    ax=gca()
    ax.set_aspect(1)
    title(f"Energy pumped into mode 2, in units of $\hbar\omega_1\omega_2/2\pi$.\n$k_y$ = {ky0:.3}"+" ${\\rm Å}^{-1}$")# ({P0/(meV/picosecond):.2} meV/ps)")
    xlabel("kx [${\\rm Å}^{-1}$]")
    ylabel("kz [${\\rm Å}^{-1}$]")
    plot([0],[0],'xk')
    if savefig:
        
        savefig(f"../Figures/Parm{nparm}_{FigID}_P2.png",dpi=300)
    
    
    
    
    figure(3,figsize=(7,7))
    Vmax = amax(abs(Dg[:,A0,:])/P0)         
    
    pcolormesh(xg[:,A0,:],zg[:,A0,:],Dg[:,A0,:]/P0,cmap="bwr",vmin=-Vmax,vmax=Vmax)          
    colorbar()
    ax=gca()
    ax.set_aspect(1)
    title(f"Energy being dissipated, in units of $\hbar\omega_1\omega_2/2\pi$.\n$k_y$ = {ky0:.3}"+" ${\\rm Å}^{-1}$")# ({P0/(meV/picosecond):.2} meV/ps)")
    xlabel("kx [${\\rm Å}^{-1}$]")
    ylabel("kz [${\\rm Å}^{-1}$]")
    plot([0],[0],'xk')
    if savefig:
        
        savefig(f"../Figures/Parm{nparm}_{FigID}_Dis.png",dpi=300)
    
    
    
    
    figure(4,figsize=(7,7))
    Vmax = amax(abs(P2g[:,A0,:])/P0)         
    title("Density in steady-state")
    pcolormesh(xg[:,A0,:],zg[:,A0,:],-Dg[:,A0,:],vmin=-2,vmax=0,cmap="PuOr")
    colorbar()
    ax=gca()
    ax.set_aspect(1)
    
    if savefig:
        
        savefig(f"../Figures/Parm{nparm}_{FigID}_Dens.png",dpi=300)
    
    show()
