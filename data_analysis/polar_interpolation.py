#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 10:09:24 2020

@author: frederik
"""

import os 
import sys
sys.path.append("/Users/frederik/Dropbox/Academics/Active/WeylNumerics/Code")
os.chdir('../')
from scipy import *
import basic as B
import time as time 
from matplotlib.pyplot import *
from Units import *
import kgrid as kgrid 

from DataProcessingMasterEquation import *
import data_refinement_v1 as DR

#%%
import data_plotting as PL
    
  
n0=161
angle=0#pi/2-0.1
angle = 0
phi_res = 12
parameters,klist,p1,p2,rho,Eeq,Ess,TDS              = get_data(n0)
omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp    = parameters

tau = parameters[2]

scalar_data,angle_data,data_2d,nlist = DR.compute_power(klist,p1,p2,rho,phi_res=phi_res)
e_sd,e_ad,e_d2d,e_nlist = DR.compute_energy(klist,Eeq,Ess,phi_res=phi_res)

(Power_1,Power_2,Density)   = scalar_data
(P1_phi,P2_phi,Rho_phi)     = angle_data
((P1,P2,Rho),out_grid)      = data_2d
((Eeq,Ess),out_grid_e)      = e_d2d
rg,zg = out_grid
#(Power_1,Power_2,Density)   = scalar_data
#(P1_phi,P2_phi,Rho_phi)     = angle_data
#((P1,P2,Rho),out_grid)      = data_2d

#P1_std,P2_std = DR.estimate_errorbars(klist,p1,p2,rho)
#print(f"Total Power:")
#print(f"   Mode 1:  {Power_1:>10.4} meV**2/Å^3")
#print(f"   Mode 2:  {Power_2:>10.4} meV**2/Å^3")
#print("="*80)


P0=omega1*omega2/(2*pi)


PL.angle_plot(angle_data)
PL.power_plot(data_2d,P0,angle=angle,nfig=20,vmax=1.)
PL.energy_plot(e_d2d,P0,tau,angle=angle,nfig=75,vmax=1.)
PL.density_plot(data_2d,angle=angle)
#ylim((-0.1,0.1))
#xlim((-0.08,0.08))

#ylim((0,0.15))
#xlim((0,0.01))
Diff =(-( P1+P2) -1/tau * (Ess-Eeq))/P0

S = shape(P2)[0]
nphi = int((angle/(pi/2))*S)

figure(4)
title(f"Particle density vs. $k_r$ and $k_z$, at xy-angle $\phi = {around(angle,1)}$, in units of $P_0$")

Pd= amax(abs(Diff))
Pd = 0.1
pcolormesh(rg,zg,Diff[nphi,:,:],cmap="bwr",vmin=-Pd,vmax=Pd)
pcolormesh(-rg,zg,Diff[nphi,:,:],cmap="bwr",vmin=-Pd,vmax=Pd)
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
title("Difference between absorbed and disspated energy [$P_0$] \n(i.e. energy absorbed by the electrons which is not dissipated yet)",fontsize=12)
colorbar()
  
ND = sum(abs(Diff))*0.001**2*(pi/(2*phi_res))
print(f"Norm of difference: {ND}")

#%% 
# ind = where(TDS==1)

PL.data_point_plot(klist,TDS,angle-1e-7,dphi=pi/(2*phi_res),nfig=11)
ax  =gca()
ax.set_aspect("equal")
plt = gcf()
plt.set_size_inches(11,8)
# colorbar()