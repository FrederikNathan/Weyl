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
import basic as B
import time as time 
from matplotlib.pyplot import *
from Units import *
import kgrid as kgrid 
from DataProcessingMasterEquation import *



def extend_angle(x):
    
    out = concatenate((x,x[-2:0:-1]))
    out= concatenate((out,out))
    return out

def compute_power(klist,p1,p2,k0):
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
    

    
    kx,ky,kz = [klist[:,n] for n in (0,1,2)]
    r        = sqrt(kx**2+ky**2)
    phi      = arcsin(ky/(1e-14+r))
    
    # rescaled coordinates to feed to interpolator
    phi_n    = phi/(pi/2+1e-6)
    r_n      = r/k0
    kz_n     = kz/k0
    
    kpoints  = array([phi_n,r_n,kz_n]).T
    
    order_c = 4
    order_f = 6
    # Fine grid interpolator
    out_coords_f,nlist_f,data_f = kgrid.compute_grid(kpoints,array([p1,p2,p1*r,p2*r]).T,(0.5,0.5,0),1,(1,1,2),order=order_f)
    
    # Crude grid interpolator
    r_n_max  = amax(r_n)
    kz_n_max = amax(kz_n)
    kz_n_min = amin(kz_n)
    
    
    kz_n_int = kz_n_max-kz_n_min
    kz_n_cen = 0.5*(kz_n_max+kz_n_min)
    nz = 2*int(kz_n_int//2)+2
    nr = int(r_n_max)+1
    kz_box_cen = int(kz_n_cen)
    kr_box_cen = nr//2
    kz_box_min = min(kz_box_cen-nz//2,-1)
    kz_box_max = max(kz_box_cen+nz//2,1)
    nz = kz_box_max-kz_box_min
    
    ## dimension of cubic subregion grid used for fine interpolator
    center_c = (0.5,0.5*nr,kz_box_cen)
    out_coords_c,nlist_c,data_c = kgrid.compute_grid(kpoints,array([p1,p2,p1*r,p2*r]).T,center_c,1,(1,nr,nz),order=order_c)
    
    
    
    Phi_f,Kr_f,Kz_f = out_coords_f
    Phi_c,Kr_c,Kz_c = out_coords_c
    
    Kr_c,Kz_c = k0*Kr_c,k0*Kz_c
    Kr_f,Kz_f = k0*Kr_f,k0*Kz_f
    
    ## Bounds for fine interpolation grid
    dr_f   = Kr_f[1]  -  Kr_f[0]
    dz_f   = Kz_f[1]  -  Kz_f[0]
    dphi_f = Phi_f[1] -  Phi_f[0]
    dr_c   = Kr_c[1]  -  Kr_c[0]
    dz_c   = Kz_c[1]  -  Kz_c[0]
    dphi_c = Phi_c[1] -  Phi_c[0] 
    
    P1_f,P2_f,P1w_f,P2w_f = [data_f[:,:,:,n] for n in (0,1,2,3)]
    P1_c,P2_c,P1w_c,P2w_c = [data_c[:,:,:,n] for n in (0,1,2,3)]
    
    Nphi_f,Nr_f,Nz_f = (len(x) for x in out_coords_f)
    Nphi_c,Nr_c,Nz_c = (len(x) for x in out_coords_c)
    
    ez_min = 2**order_c * (-1-kz_box_min)
    ez_max = 2**order_c * (kz_box_max-1)
    
    er_max = 2**order_c * 1
    
    P1w_c[:,:er_max,ez_min:-ez_max]=0
    P2w_c[:,:er_max,ez_min:-ez_max]=0
    
    
    P1_f_phi = sum(P1w_f,axis=(1,2))*dr_f*dz_f
    P2_f_phi = sum(P2w_f,axis=(1,2))*dr_f*dz_f
    
    P1_c_phi = sum(P1w_c,axis=(1,2))*dr_c*dz_c
    P2_c_phi = sum(P2w_c,axis=(1,2))*dr_c*dz_c
    
    
    P1_c_phi,P2_c_phi = [(array([x]*(2**(order_f-order_c))).T).flatten() for x in  (P1_c_phi,P2_c_phi)]# sum(Rhow_c,axis=(1,2))*dr_c*dz_c
    
    P1_phi  = P1_c_phi  + P1_f_phi 
    P2_phi  = P2_c_phi  + P2_f_phi 

    
    #raise ValueError
    P1_phi,P2_phi = [extend_angle(x) for x in (P1_phi,P2_phi)]
    Nphi = len(P1_phi)
    phivec = arange(0,2*pi,2*pi/Nphi)
    dphi = 2*pi/Nphi
    
    Dis_phi = -P1_phi-P2_phi
    
    
    Power_1 = sum(P1_phi)*dphi
    Power_2 = sum(P2_phi)*dphi
    Dissipation = -Power_1-Power_2
    
    return Power_1,Power_2,P1_phi,P2_phi,((Phi_f,Kr_f,Kz_f),(P1_f,P2_f)),((Phi_c,Kr_c,Kz_c),(P1_c,P2_c)),nlist_f,nlist_c

def estimate_errorbars(klist,p1,p2,k0):
    N_SAMPLES = 30
    FRACTION  = 0.5
    
    NK = shape(klist)[0]
    
    P1list = zeros(N_SAMPLES)
    P2list = zeros(N_SAMPLES)
    print("Estimating uncertainty of computed power")
    for n in range(0,N_SAMPLES):
        if (n+1)%(max(1,N_SAMPLES//10))==0:    
            print(f"    At step {n+1}/{N_SAMPLES}")
        P = npr.permutation(NK)
        N1 = int(NK*FRACTION)
        
        Ind = P[:N1]
        
        p1_t,p2_t,klist_t = [x[Ind] for x in (p1,p2,klist)]
        
        P1,P2,A,B,C,D = compute_power(klist_t,p1_t,p2_t,k0)
        
        P1list[n] = P1 
        P2list[n] = P2
        
        
    P1_std,P2_std = (std(x)*sqrt(FRACTION) for x in (P1list,P2list))
    
    return P1_std,P2_std

if __name__=="__main__":
        
    n0=142
    parameters,klist,p1,p2,rho,Eeq,Ess,TDS              = get_data(n0)
    omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp    = parameters
    P0=omega1*omega2/(2*pi)
    A1,A2 = EF1/omega1,EF2/omega2
    k0 = max(A1,A2)/3 # characteristic k_scale used for interpolation - maximal vector potential
    Power_1,Power_2,P1_phi,P2_phi,((Phi_f,Kr_f,Kz_f),(P1_f,P2_f)),((Phi_c,Kr_c,Kz_c),(P1_c,P2_c)) = compute_power(klist,p1,p2,k0)
    
    P1_std,P2_std = estimate_errorbars(klist,p1,p2,k0)
    
    # =============================================================================
    # Estimate standard deviation
    # =============================================================================
    
    #print((P1_std,P2_std))
    
    print("-"*80)
    print(f"Total Power:")
    print(f"   Mode 1:  {Power_1:>10.4} +/- {P1_std:>5.4} meV**2/Å^3")
    print(f"   Mode 2:  {Power_2:>10.4} +/- {P2_std:>5.4} meV**2/Å^3")
    print("="*80)



