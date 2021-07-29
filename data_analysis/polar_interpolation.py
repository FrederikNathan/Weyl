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
    
  
n0=156
angle=0.01
# angle = 0
phi_res = 12
parameters,klist,p1,p2,rho,Eeq,Ess,TDS              = get_data(n0)
omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp    = parameters
P0=omega1*omega2/(2*pi)

tau = parameters[2]

scalar_data,angle_data,data_2d,nlist = DR.compute_power(klist,p1,p2,rho,phi_res=phi_res)
e_sd,e_ad,e_d2d,e_nlist = DR.compute_energy(klist,Eeq,Ess,phi_res=phi_res)

PL.angle_plot(angle_data)
PL.power_plot(data_2d,P0,angle=angle,nfig=20,vmax=1.)
PL.energy_plot(data_2d,e_d2d,P0,tau,angle=angle,nfig=75,vmax=.1)
PL.density_plot(data_2d,angle=angle)
PL.data_point_plot(klist,TDS,angle,dphi=pi/(2*phi_res),nfig=50)

