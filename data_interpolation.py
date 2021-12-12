#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 10:09:24 2020

@author: frederik
Functions for interpolating quanties from raw data 
"""

import os 
import sys
sys.path.append("/Users/frederik/Dropbox/Academics/Active/WeylNumerics/Code")
from scipy import *
import time as time 
from matplotlib.pyplot import *
from scipy.interpolate import griddata 

import basic as B
from units import *
from data_refiner import *
from weyl_liouvillian import * 
N_SAMPLES_ESTIMATE_ERROR = 5 #30


def extend_angle(x):
    
    out = concatenate((x,x[-2:0:-1]))
    out= concatenate((out,out))
    return out

def compute_energy(klist,Eeq,Ess,phi_res=6):
    """
    
    Interpolate average energy and steady state energy at k-points to array of k-points with even spacing in cylindrical coordinates. The polar angle resolution set by phi_res.
    The radial and z resolution of k is set to 0.001.
    
    Input and output is analogous to compute_power, see doc of that method. 
    
    Parameters
    ----------
    klist : ndarray(NP,3) float
        k-points of data points. klist[np] gives k-position of data point np
    Eeq : ndarray(NP) float
        average equilibrium energies at k-points in klist. Eeq[z] gives the average equilibrium energy at k=klist[z]
    Ess : ndarray(NP) float
        average steady-state energy. Ess[z] gives the average steady sstate energy at k=klist[z]
    phi_res : int, optional
        polar angle resolution of interpolated array. The default is 6.

    Returns
    -------
    (E_equilibrium, E_steadystate): ndarray,,, equilibrim
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    tuple
        DESCRIPTION.
    tuple
        DESCRIPTION.
    Nlist : TYPE
        DESCRIPTION.

    """
    
    kx,ky,kz = [klist[:,n] for n in (0,1,2)]
    r        = sqrt(kx**2+ky**2)
    phi      = arcsin(ky/(1e-14+r))+1e-9
    
    kpoints  = array([phi,r,kz]).T

    dphi = pi/(2*phi_res)
    
    rmax = amax(r)
    kzmax=amax(kz)
    kzmin = amin(kz)
    
    kz_int = kzmax-kzmin

    #kz_box_cen = int(kz_n_cen)
    dk = 0.001
    rvec = arange(0,rmax+dk,dk)
    kzvec = arange(kzmin-dk,kzmax+dk,dk)
    out_grid = meshgrid(rvec,kzvec,indexing="ij")
    
    N_r = len(rvec)
    N_z = len(kzvec)
    N_phi=phi_res
    
    Out = zeros((4,N_phi,N_r,N_z))
    Nlist = zeros(N_phi)
    for nphi in range(0,phi_res):
        phimin = nphi*dphi
        phimax = (nphi+1)*dphi
        
        Ind = where((kpoints[:,0]<phimax)*(kpoints[:,0]>=phimin))[0]
        if len(Ind)>0:
            
            eeq_in,ess_in,k2d_in,r_in   = [x[Ind] for x in (Eeq,Ess,kpoints[:,1:],r)]
            
            x= griddata(k2d_in,array([eeq_in,ess_in,eeq_in*r_in,ess_in*r_in]).T,(out_grid[0],out_grid[1]),fill_value=0,method="nearest")
            Out[:,nphi,:,:]=x.swapaxes(0,2).swapaxes(1,2)
            Nlist[nphi] = 1*len(Ind)
    
    Eeq = Out[0,:,:,:]
    Ess = Out[1,:,:,:]
    Eeqw = Out[2,:,:,:]
    Essw = Out[3,:,:,:]
    
    Eeq_phi = extend_angle(sum(Eeqw,axis=(1,2))*dk**2)
    Ess_phi = extend_angle(sum(Essw,axis=(1,2))*dk**2)

    Nlist  = extend_angle(Nlist)
    
    Nphi = len(Eeq_phi)
    dphi = 2*pi/Nphi
    
    E_equilibrium = sum(Eeq_phi)*dphi/(8*pi**3)
    E_steadystate = sum(Ess_phi)*dphi/(8*pi**3)
    
    

    return (E_equilibrium,E_steadystate),(Eeq_phi,Ess_phi),((Eeq,Ess),out_grid),Nlist


def compute_power(klist,p1,p2,rho,phi_res=6):
    """
    Compute pumping power at data set n0, using grid interpolator.
    
    input:
        klist : array of shape (N,3): k-points of data
        p1,p2 : arrays of shape (N) computed power at data points in klist
        k0    : float. momentum bound at which the fine-resolution is used
    
    returns 
    
        Power_1,Power_2,Data_crude,Data_fine, where 
        
        Power_1 Power_2         :  the rate of energy transfer into mode 1 and 2 (in units of meV**2/Å**3)
        Data_crude,Data_fine    :  momentum-resolved data for the pumping power (useful e.g. for plotting), see below:
                    
    Here Data_crude = ((Phi,Kr,Kz),(P1,P2)), where Phi,Kr,Kz gives the samping grid for the crude interpolation in cylindrical coordinates, and P1,P2 gives the corresponding computed powers vs. momentum
    Similarly for Data_fine
        
    
    BZ is divided into two parts, where a fine and crude resolution is used for the 
    interpolation scheme, respectively
    
    The fine region is the region |k_z|<k0, |k_r|<k0, 
    
    In the crude region, the momentum resolution is pi/32 (0.1 rad) for the angle  coordinate, and  and k0/32 for the z and r coordinates
    In the fine region, the momentum resolution is pi/128 (0.025 rad) for the angle  coordinate, and  and k0/128 for the z and r coordinates
    
    
    """
        
    global ind,kpoints_reduced,rho_reduced,rhho,rho_in_reduced,Ind_reduced,phimax,phimin,x,y
    kx,ky,kz = [klist[:,n] for n in (0,1,2)]
    r        = sqrt(kx**2+ky**2)
    phi      = arcsin(ky/(1e-14+r))+1e-9
    
    kpoints  = array([phi,r,kz]).T

    dphi = pi/(2*phi_res)
    
    rhho = 1*rho
    rmax = amax(r)
    kzmax=amax(kz)
    kzmin = amin(kz)
    
    kz_int = kzmax-kzmin
    
    ind = where(isnan(rho)^1)[0]
    [kpoints_reduced,r_reduced,rho_reduced] =[x[ind] for x in [kpoints,r,rho]]


    #kz_box_cen = int(kz_n_cen)
    dk = 0.001
    rvec = arange(0,rmax+dk,dk)
    kzvec = arange(kzmin-dk,kzmax+dk,dk)
    out_grid = meshgrid(rvec,kzvec,indexing="ij")
    
    N_r = len(rvec)
    N_z = len(kzvec)
    N_phi=phi_res
    
    Out = zeros((6,N_phi,N_r,N_z))
    Nlist = zeros(N_phi)
    for nphi in range(0,phi_res):
        phimin = nphi*dphi
        phimax = (nphi+1)*dphi
        
        Ind = where((kpoints[:,0]<phimax)*(kpoints[:,0]>=phimin))[0]
        Ind_reduced = where((kpoints_reduced[:,0]<phimax)*(kpoints_reduced[:,0]>=phimin))[0]
        if len(Ind)>0:
            
            p1_in,p2_in,k2d_in,r_in    = [x[Ind] for x in (p1,p2,kpoints[:,1:],r)]
            k2d_in_reduced,r_in_reduced,rho_in_reduced    = [x[Ind_reduced] for x in (kpoints_reduced[:,1:],r_reduced,rho_reduced)]
            
            x= griddata(k2d_in,array([p1_in,p2_in,p1_in*r_in,p2_in*r_in]).T,(out_grid[0],out_grid[1]),fill_value=0,method="nearest")
            
            if len(Ind_reduced)>0:
                
                y= griddata(k2d_in_reduced,array([rho_in_reduced,rho_in_reduced*r_in_reduced]).T,(out_grid[0],out_grid[1]),fill_value=0,method="nearest")
            else:
                y = nan*ones((N_r,N_z,2))
                

            Out[array([0,1,3,4]),nphi,:,:]=x.swapaxes(0,2).swapaxes(1,2)
            Out[array([2,5]),nphi,:,:] =y.swapaxes(0,2).swapaxes(1,2)

            Nlist[nphi] = 1*len(Ind)
    
    P1   = Out[0,:,:,:]
    P2   = Out[1,:,:,:]
    Rho  = Out[2,:,:,:]
    P1w  = Out[3,:,:,:]
    P2w  = Out[4,:,:,:]
    Rhow = Out[5,:,:,:]
    
    P1_phi = extend_angle(sum(P1w,axis=(1,2))*dk**2)
    P2_phi = extend_angle(sum(P2w,axis=(1,2))*dk**2)
    Rho_phi = extend_angle(sum(Rhow,axis=(1,2)))*dk**2
    Nlist  = extend_angle(Nlist)
    
    Nphi = len(P1_phi)
    dphi = 2*pi/Nphi
    
    Power_1 = sum(P1_phi)*dphi/(8*pi**3)
    Power_2 = sum(P2_phi)*dphi/(8*pi**3)
    Density = sum(Rho_phi)*dphi/(8*pi**3)
    
    

    return (Power_1,Power_2,Density),(P1_phi,P2_phi,Rho_phi),((P1,P2,Rho),out_grid),Nlist

def estimate_errorbars(klist,p1,p2,rho):
    N_SAMPLES = N_SAMPLES_ESTIMATE_ERROR#30
    FRACTION  = 0.5
    
    NK = shape(klist)[0]
    
    P1list = zeros(N_SAMPLES)
    P2list = zeros(N_SAMPLES)

    for n in range(0,N_SAMPLES):

        P = npr.permutation(NK)
        N1 = int(NK*FRACTION)
        
        Ind = P[:N1]
        p1_t,p2_t,klist_t,rho_t = [x[Ind] for x in (p1,p2,klist,rho)]
        
        S,V,M,N = compute_power(klist_t,p1_t,p2_t,rho_t)
        P1,P2,R = S
        P1list[n] = P1 
        P2list[n] = P2
        
        
    P1_std,P2_std = (std(x)*sqrt(FRACTION) for x in (P1list,P2list))
    
    return P1_std,P2_std


def power_sweep(nlist):
    NS = len(nlist)
    
    P1_out=zeros(NS)
    P2_out=zeros(NS)
    Rho_out=zeros(NS)
    Std1_out=zeros(NS)
    Std2_out= zeros(NS)
    
    Trustworthy_out =  zeros(NS,dtype=bool)
    parameters_out = zeros((NS,11))
    n = 0
    B.tic(n=8)
    global Power_1
    print("Refining data.")
    for n0 in nlist:
        B.tic(n=7)
        print(f"At parameter set {n0} (set {n+1}/{NS})")
        print("     Computing power")
        parameters,klist,p1,p2,rho,Eeq,Ess,TDS              = get_data(n0,disp=0)
        (Power_1,Power_2,Density),(P1_phi,P2_phi,Rho_phi),((P1,P2,Rho),out_grid),Nlist = compute_power(klist,p1,p2,rho)
    
        kx,ky,kz = [klist[:,n] for n in (0,1,2)]
        r        = sqrt(kx**2+ky**2)
        
        utw = amax(Rho_phi/mean(Rho_phi)-1)>0.1 or amin(kz)>-0.35 or amax(kz)<0.09 or amax(r)<0.12
        print("     Estimating error")
        std1,std2 = estimate_errorbars(klist,p1,p2,rho)
        print(f"     done. Time spent: {B.toc(n=7,disp=0):.4} s")
        
        P1_out[n]  = 1*Power_1
        P2_out[n]  = 1*Power_2
        Rho_out[n] = 1*Density
        Std1_out[n],Std2_out[n] = std1,std2
        Trustworthy_out[n] = not utw
        
        parameters_out[n,:]=parameters


        n+=1
    print(f"done. Total time spent: {B.toc(n=8,disp=0):.4} s ")
    return (P1_out,P2_out),(Std1_out,Std2_out),Rho_out,Trustworthy_out,parameters_out
    
        
        
        
    omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp    = parameters
if __name__=="__main__":
    import data_plotting as PL 
#    n0=142
#    parameters,klist,p1,p2,rho,Eeq,Ess,TDS              = get_data(n0)
#    omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp    = parameters
#    P0=omega1*omega2/(2*pi)
#
#    (Power_1,Power_2,Density),(P1_phi,P2_phi,Rho_phi),((P1,P2,Rho),out_grid),Nlist = compute_power(klist,p1,p2,rho)
#    A=estimate_errorbars(klist,p1,p2,rho)
    # =============================================================================
    # Estimate standard deviation
#    # =============================================================================
#    
#
#    print("-"*80)
#    print(f"Total Power:")
#    print(f"   Mode 1:  {Power_1:>10.4}")# +/- {P1_std:>5.4} meV**2/Å^3")
#    print(f"   Mode 2:  {Power_2:>10.4}")# +/- {P2_std:>5.4} meV**2/Å^3")
#    print("="*80)

    

    X=power_sweep([108,139,140])
