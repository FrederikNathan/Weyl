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
import data_plotting as PL


    
  
nlist = [108,139,140,141,142,143,144,145,146,147,148,149,150,151,152]
try:
    A=tw
except:
    
    (P1,P2),(std1,std2),rho,tw,parameters = DR.power_sweep(nlist)



    tau = parameters[:,2]
    
    AS = argsort(tau)
    vf  = parameters[:,3]
    P1,P2,std1,std2,vf,tau = (x[AS] for x in (P1,P2,std1,std2,vf,tau))
    


    


ind_2e5  = where(abs(vf-2e5*meter/second)<1e-8)[0]
ind_5e5  = where(abs(vf-5e5*meter/second)<1e-8)[0]
ind_1e6  = where(abs(vf-1e6*meter/second)<1e-8)[0]

P1_2e5 = P1[ind_2e5]/(Joule/(second*micrometer**3))
P2_2e5 = P2[ind_2e5]/(Joule/(second*micrometer**3))
S1_2e5 = std1[ind_2e5]/(Joule/(second*micrometer**3))
S2_2e5 = std2[ind_2e5]/(Joule/(second*micrometer**3))
tau_2e5 = tau[ind_2e5]

P1_5e5 = P1[ind_5e5]/(Joule/(second*micrometer**3))
P2_5e5 = P2[ind_5e5]/(Joule/(second*micrometer**3))
S1_5e5 = std1[ind_5e5]/(Joule/(second*micrometer**3))
S2_5e5 = std2[ind_5e5]/(Joule/(second*micrometer**3))
tau_5e5 = tau[ind_5e5]

P1_1e6= P1[ind_1e6] /(Joule/(second*micrometer**3))
P2_1e6= P2[ind_1e6] /(Joule/(second*micrometer**3))
S1_1e6 = std1[ind_1e6]/(Joule/(second*micrometer**3))
S2_1e6 = std2[ind_1e6]/(Joule/(second*micrometer**3))
tau_1e6 = tau[ind_1e6]

EF1,EF2,Mu,Temp    = parameters[0,-4:]
omega1,omega2 = parameters[0,:2]

figure(1)
errorbar(tau_2e5/picosecond,P2_2e5,yerr=S2_2e5,fmt='s-')
errorbar(tau_5e5/picosecond,P2_5e5,yerr=S2_5e5,fmt='s-')
errorbar(tau_1e6/picosecond,P2_1e6,yerr=S2_1e6,fmt='s-')
legend(("$v_{\\rm F} = 2\cdot 10^5$ m/s, $\mu=24$ meV",
        "$v_{\\rm F} = 5\cdot 10^5$ m/s, $\mu=60$ meV",
        "$v_{\\rm F} = 1\cdot 10^6$ m/s, $\mu=115$ meV"))
plot([0,1.1*amax(tau/picosecond)],[0,0],'--k')
xlabel("relaxation time, $\\tau$, picoseconds")
ylabel("Amplification power, $W/\mu m^3$")
title("Amplification  power vs. $\\tau$. $H(k) = v_{\\rm F}  \,\\vec{k}\cdot\\vec{\sigma} +0.8 v_{\\rm F}\, k_z$. \n"+f"$E_1 = {EF1/(10**6*Volt/meter):.2} \cdot 10^6$ V/m, $E_2 = {EF2/(10**6*Volt/meter):.2}\cdot 10^6$ V/m, "+
      f"$\omega_1 = {omega1/THz:2.3}$ THz,$\omega_2 = {omega2/THz:2.3}$ THz")
ylim((-0.1,0.075))
plt = gcf()
plt.set_size_inches(8,5)
savefig("../Figures/amplification_power_1.png",dpi=200)
# $v=(v_F,v_F,$E_1=
#      ")
figure(2)
errorbar(tau_2e5/picosecond,P1_2e5,yerr=S1_2e5,fmt='s-')
errorbar(tau_5e5/picosecond,P1_5e5,yerr=S1_5e5,fmt='s-')
errorbar(tau_1e6/picosecond,P1_1e6,yerr=S1_1e6,fmt='s-')
legend(("$v_{\\rm F} = 2\cdot 10^5$ m/s, $\mu=24$ meV",
        "$v_{\\rm F} = 5\cdot 10^5$ m/s, $\mu=60$ meV",
        "$v_{\\rm F} = 1\cdot 10^6$ m/s, $\mu=115$ meV"))
plot([0,1.1*amax(tau/picosecond)],[0,0],':k')
xlabel("relaxation time, $\\tau$, picoseconds")
ylabel("Amplification power, $W/\mu m^3$")
title("Amplification power vs. $\\tau$ (mode 2). $H(k) = v_{\\rm F}  \,\\vec{k}\cdot\\vec{\sigma} +0.8 v_{\\rm F}\, k_z$. \n"+f"$E_1 = {EF1/(10**6*Volt/meter):.2} \cdot 10^6$ V/m, $E_2 = {EF2/(10**6*Volt/meter):.2}\cdot 10^6$ V/m, "+
      f"$\omega_1 = {omega1/THz:2.3}$ THz,$\omega_2 = {omega2/THz:2.3}$ THz")
ylim((-0.2,0.04))
plt = gcf()
plt.set_size_inches(8,5)
savefig("../Figures/amplification_power_2.png",dpi=200)