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
import time as time 
from matplotlib.pyplot import *

import basic as B
from units import *
from data_refiner import *
import data_interpolation as DR
import data_plotting as PL
    
  
n0=165
angle=0.01
# angle = 0
phi_res = 12
parameters,klist,p1,p2,rho,Eeq,Ess,TDS              = get_data(n0)
omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp    = parameters
P0=omega1*omega2/(2*pi)

tau = parameters[2]

scalar_data,angle_data,data_2d,nlist = DR.compute_power(klist,p1,p2,rho,phi_res=phi_res)
e_sd,e_ad,e_d2d,e_nlist = DR.compute_energy(klist,Eeq,Ess,phi_res=phi_res)
(E_eq,E_ss) = e_sd
Diss = 1/tau * (E_ss-E_eq)

vmax_energy = 1
PL.angle_plot(angle_data,e_ad,tau)
PL.power_plot(data_2d,P0,angle=angle,nfig=20,vmax=1.)
# raise ValueError
PL.energy_plot(data_2d,e_d2d,P0,tau,angle=angle,nfig=75,vmax=vmax_energy)
PL.density_plot(data_2d,angle=angle)
PL.data_point_plot(klist,TDS,angle,dphi=pi/(2*phi_res),nfig=50)

#%%
(P1,P2),(std1,std2),rho,tw,parameters = DR.power_sweep([n0])
print("-"*80)
# print("")
# 
print(f"Dissipation        : {Diss/(Joule/second/(micrometer**3)):>10.4} W/micrometer^2")
print(f"Unaccounted energy : {(Diss-P1[0]-P2[0])/(Joule/second/(micrometer**3)):>10.4} W/micrometer^2")
print("-"*80)
# (P1,P2),(std1,std2),rho,tw,parameters = DR.power_sweep([n0])
# print("") 
print(f"Conversion power   : {P1[0]/(Joule/second/(micrometer**3)):>10.4} W/micrometer^2 (+/- {abs(std1[0]/P1[0])*100:.0}%)")#/(Joule/second/(micrometer**3)):>0.4} W/micrometer^2")
print("="*80)