#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 10:09:24 2020

@author: frederik
"""

import os 
import sys
sys.path.append("/Users/frederik/Dropbox/Academics/Active/WeylNumerics/Code")
from scipy import *
import time as time 
from matplotlib.pyplot import *


import basic as B
from units import *
from data_refiner import *


def angle_plot(angle_data,energy_angle_data,tau,nfig=1):
    (P1_phi,P2_phi,Rho_phi)=angle_data
    (Eeq,Ess)=energy_angle_data
    Nphi = len(P1_phi)
    phivec = arange(0,2*pi,2*pi/Nphi)
    dphi = 2*pi/Nphi    

    figure(nfig)    
    Dis_1_phi = - P1_phi-P2_phi
    Dis_2_phi = -1/tau*(Eeq-Ess)
    Amax=max(amax(P1_phi),amax(P2_phi),amax(Dis_1_phi),amax(Dis_2_phi))
    Rmax = amax(Rho_phi)
    
    
    P0 = Joule/(second*micrometer**2)
    title("Contribution to pumping vs. xy angle ($\phi$)")
    plot(phivec,P1_phi/P0,'.-')
    plot(phivec,P2_phi/P0,'.-')
    plot(phivec,Dis_1_phi/P0,'.-')
    plot(phivec,Dis_2_phi/P0,'.-')
    # plot(phivec,Rho_phi*Amax/Rmax,'.-')
        
    xlabel("$\phi$")
    ylabel("Power, $W/\mu m^3$")
    legend(("$dP_1/d\phi$","$dP_2/d\phi$","$-dP_1/d\phi-dP_2/d\phi$","$dP_{\\rm dis}/d\phi$"))
    
    plt = gcf()
    plt.set_size_inches(7,7)
    
    
def power_plot(Data,P0,XLIM=(-0.25,0.25),YLIM=(-0.45,0.2),angle=0,nfig=3,vmax=None):
    ((P1,P2,Rho),grid)=Data
    
    P1  =  P1/P0
    P2  =  P2/P0
    rg,zg = grid
        
    if angle>pi/2 or angle <0:
        raise ValueError("Angle must be in the interval [0,pi/2]")
          
    if vmax==None:
        
        Pmax = amax(abs(P2))*1.05
    else:
        Pmax=vmax
    S = shape(P2)[0]
    nphi = int((angle/(pi/2))*S)
    
    figure(nfig)
    title(f"Energy transfer to mode 1 vs. $k_r$ and $k_z$, at xy-angle $\phi = {around(angle,1)}$, in units of $P_0$")
    
    pcolormesh(rg,zg,P2[nphi,:,:],cmap="bwr",vmin=-Pmax,vmax=Pmax,shading="auto")
    pcolormesh(-rg,zg,P2[nphi,:,:],cmap="bwr",vmin=-Pmax,vmax=Pmax,shading="auto")
    ylim(YLIM)
    xlim(XLIM)
    xlabel("$k_r$")
    ylabel("$k_z$")
    
    ax  =gca()
    ax.set_aspect("equal")
    plt = gcf()
    plt.set_size_inches(11,8)
    colorbar()
    
    figure(nfig+1)
    title(f"Energy transfer to mode 2 vs. $k_r$ and $k_z$, at xy-angle $\phi = {around(angle,1)}$, in units of $P_0$")
    
    pcolormesh(rg,zg,P1[nphi,:,:],cmap="bwr",vmin=-Pmax,vmax=Pmax,shading="auto")
    pcolormesh(-rg,zg,P1[nphi,:,:],cmap="bwr",vmin=-Pmax,vmax=Pmax,shading="auto")
    ylim(YLIM)
    xlim(XLIM)
    xlabel("$k_r$")
    ylabel("$k_z$")
    
    
    ax  =gca()
    ax.set_aspect("equal")
    plt = gcf()
    plt.set_size_inches(11,8)
    colorbar()

    
    
    # ax  =gca()
    # ax.set_aspect("equal")
    # plt = gcf()
    # plt.set_size_inches(11,8)    
    # colorbar()
      
def energy_plot(DataP,DataE,P0,tau,XLIM=(-0.25,0.25),YLIM=(-0.45,0.2),angle=0,nfig=3,vmax=None):
    ((P1,P2,Rho),grid)=DataP
    ((Eeq,Ess),grid)=DataE
    
    Nphi = shape(Eeq)[0]
    
    rg,zg = grid
        
    if angle>pi/2 or angle <0:
        raise ValueError("Angle must be in the interval [0,pi/2]")
          
    if vmax==None:
        
        Pmax = max(amax(abs(Eeq)),amax(abs(Ess)))*1.01
    else:
        Pmax=vmax
    S = shape(Eeq)[0]
    nphi = int((angle/(pi/2))*S)
    
    Emax = max(amax((Eeq)),amax((Ess)))*1.01
    Emin = min(amin((Eeq)),amin(Ess))*1.01
    # figure(nfig)  
    # title(f"Average equilibrium energy vs. $k_r$ and $k_z$, at xy-angle $\phi = {around(angle,1)}$, in units of $P_0$")
    
    
    # pcolormesh(rg,zg,Eeq[nphi,:,:],cmap="hsv",vmax=Emax,vmin=Emin)
    # pcolormesh(-rg,zg,Eeq[nphi,:,:],cmap="hsv",vmax=Emax,vmin=Emin)
    # ylim(YLIM)
    # xlim(XLIM)
    # xlabel("$k_r$")
    # ylabel("$k_z$")
    
    # ax  =gca()
    # ax.set_aspect("equal")
    # plt = gcf()
    # plt.set_size_inches(11,8)
    # colorbar()
    
    # figure(nfig+1)
    # title(f"Average steady-state energy vs. $k_r$ and $k_z$, at xy-angle $\phi = {around(angle,1)}$, in units of $P_0$")
    
    # pcolormesh(rg,zg,Ess[nphi,:,:],cmap="hsv",vmax=Emax,vmin=Emin)
    # pcolormesh(-rg,zg,Ess[nphi,:,:],cmap="hsv",vmax=Emax,vmin=Emin)
    # ylim(YLIM)
    # xlim(XLIM)
    # xlabel("$k_r$")
    # ylabel("$k_z$")
    
    
    # ax  =gca()
    # ax.set_aspect("equal")
    # plt = gcf()
    # plt.set_size_inches(11,8)
    # colorbar()
    global Diss
    figure(nfig)
    title(f"Dissipation (computed by average energy difference to equilibrium)\nvs. $k_r$ and $k_z$, at xy-angle $\phi = {around(angle,1)}$, in units of $P_0$")
    
    Diss = 1/tau*(Ess[nphi,:,:]-Eeq[nphi,:,:])/P0
    pcolormesh(rg,zg,Diss,cmap="bwr",vmin=-Pmax,vmax=Pmax,shading="auto")
    pcolormesh(-rg,zg,Diss,cmap="bwr",vmin=-Pmax,vmax=Pmax,shading="auto")
    ylim(YLIM)
    xlim(XLIM)
    xlabel("$k_r$")
    ylabel("$k_z$")
    
    
    ax  =gca()
    ax.set_aspect("equal")
    plt = gcf()
    plt.set_size_inches(11,8)    
    colorbar()
    
    figure(nfig+1)
    title(f"Work done by driving on system\nvs. $k_r$ and $k_z$, at xy-angle $\phi = {around(angle,1)}$, in units of $P_0$")
    
    pcolormesh(rg,zg,(-P1[nphi,:,:]-P2[nphi,:,:])/P0,cmap="bwr",vmin=-Pmax,vmax=Pmax,shading="auto")
    pcolormesh(-rg,zg,(-P1[nphi,:,:]-P2[nphi,:,:])/P0,cmap="bwr",vmin=-Pmax,vmax=Pmax,shading="auto")
    ylim(YLIM)
    xlim(XLIM)
    xlabel("$k_r$")
    ylabel("$k_z$")
    
    ax  =gca()
    ax.set_aspect("equal")
    plt = gcf()
    plt.set_size_inches(11,8)    
    colorbar()
    
    figure(nfig+2)
    title("Absorbed energy (work - disspated energy) [$P_0$] \n(i.e. energy absorbed by the electrons which is not dissipated yet)",fontsize=12)
    Diff =(-( P1+P2) -1/tau * (Ess-Eeq))/P0
    pcolormesh(rg,zg,Diff[nphi,:,:],cmap="bwr",vmin=-Pmax,vmax=Pmax,shading="auto")
    pcolormesh(-rg,zg,Diff[nphi,:,:],cmap="bwr",vmin=-Pmax,vmax=Pmax,shading="auto")
    
    XLIM=(-0.25,0.25)
    YLIM=(-0.45,0.2)
    
    ylim(YLIM)
    xlim(XLIM)
    xlabel("$k_r$")
    ylabel("$k_z$")
    
    ax  =gca()
    ax.set_aspect("equal")
    plt = gcf()
    plt.set_size_inches(11,8)
    colorbar()
      
    # ND = sum(abs(Diff))*0.001**2*(pi/(2*Nphi))
    # print(f"Norm of difference between two energy absorptions/(): {ND:.4}")
      
def density_plot(Data,XLIM=(-0.25,0.25),YLIM=(-0.45,0.2),angle=0):
    ((P1,P2,Rho),grid)=Data
    
    rg,zg = grid
        
    if angle>pi/2 or angle <0:
        raise ValueError("Angle must be in the interval [0,pi/2]")
          
    Pmax = amax(abs(P2))
    
    S = shape(P2)[0]
    nphi = int((angle/(pi/2))*S)

    figure(4)
    title(f"Particle density vs. $k_r$ and $k_z$, at xy-angle $\phi = {around(angle,1)}$, in units of $P_0$")
    
    pcolormesh(rg,zg,-Rho[nphi,:,:],cmap="Greens",vmin=-2,vmax=-1,shading="auto")
    pcolormesh(-rg,zg,-Rho[nphi,:,:],cmap="Greens",vmin=-2,vmax=-1,shading="auto")
    ylim(YLIM)
    xlim(XLIM)
    xlabel("$k_r$")
    ylabel("$k_z$")
    
    ax  =gca()
    ax.set_aspect("equal")
    plt = gcf()
    plt.set_size_inches(11,8)
    
    # colorbar()
def data_point_plot(klist_0,TDS,phi0,dphi=pi/(2*6),XLIM=(-0.25,0.25),YLIM=(-0.45,0.2),nfig=8):
    # global kr
    cstrlist=[".k",".r"]
    legendlist=["Time-domain solver","Frequency domain solver"]
    for z in [1,0]:
        klist = klist_0[TDS==z,:]
        cstr = cstrlist[z]
        kx,ky,kz = [klist[:,n] for n in (0,1,2)]
        r        = sqrt(kx**2+ky**2)
        phi      = arcsin(ky/(1e-14+r))+1e-9
    
        phimin = phi0-dphi/2
        phimax = phi0+dphi/2
        
        Ind = where((phi<phimax)*(phi>=phimin))[0]
         
        kz  = kz[Ind]
        kr  = r[Ind]
        kz = concatenate((kz,kz))
        kr = concatenate((kr,-kr))
        figure(nfig)
        plot(kr,kz,cstr,markersize=1.5)    
        # plot(-kr,kz,cstr,markersize=1)
        # 
        ylim(YLIM)
        xlim(XLIM)
        xlabel("$k_r$")
        ylabel("$k_z$")
        title(f"Data points in angle interval $\phi \in [{around(phi0,2)},{around(phi0+dphi,2)}]$",fontsize=8)
 
        
    ax  =gca()
    ax.set_aspect("equal")
    plt = gcf()
    plt.set_size_inches(11,8)
    legend(legendlist)
#    plt = gcf()
#    plt.set_size_inches(11,8)
    
