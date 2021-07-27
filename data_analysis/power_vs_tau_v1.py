#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 10:09:24 2020

@author: frederik

WARNING: what is plotted is not the steady-state energy transffer, 
but the energy transferred over a finite time-window. See code for details of 
averaging window. At some k-points slow landau-zener tunneling 
(on the timescale averaging-window time) may cause a large absorption of 
energy from the electrons which is still not lost to dissipation 
(i.e., some energy may be reabsorbed). Thus, the energy absorption is sensitive
to the detials of the averaging window, even if the averaging window is very large.

here can also be k-points where there is a deficit of energy in the electrons 
due to the Landu zener tunneling (i.e. there is a net energy transfer from the electrons
to the driving modes. This is allowed by thermodynamics, as long as 
there are other momenta with a higher deposit. (following the principle of a heat pump)

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


    
print("WARNING: what is plotted is not the steady-state energy transffer, but the energy transferred over a finite time-window. See code for details of averaging window. At some k-points slow landau-zener tunneling (on the timescale averaging-window time) may cause a large absorption of energy from the electrons which is still not lost to dissipation (i.e., some energy may be reabsorbed). There can also be k-points where there is a deficit of energy in the electrons due to the Landu zener tunneling")
  
nlist = [108,139,140,141,142,143,149,152,153,154,156,158,160,161,162]

    
(P1,P2),(std1,std2),rho,tw,parameters = DR.power_sweep(nlist)



series_list = zeros((0,10))
series_pointer = zeros(len(parameters))

def find_index(X):
    Y = sum(abs(X-series_list),axis=1)
    Ind = where(Y<1e-8)[0]
    if len(Ind)==0 or len(series_list)==0:
        return None
    else:
        return Ind[0]
    
    
    
    
    
np=0
for p in parameters:
    X = concatenate((p[:2],p[3:]))
    Z=find_index(X)
    
    if Z==None:
#        print("Hej")
        series_list=concatenate((series_list,X.reshape((1,10))))
        series_pointer[np]=len(series_list)-1
    else:
        series_pointer[np]=1*Z
        
    np+=1
        
#%%

legendlist=[]
NS = len(series_list)
for ns in range(0,NS):
    ind = where(series_pointer==ns)[0]
    p1  = P1[ind]   
    p2  = P2[ind]   
    s1  = std1[ind] / (Joule/(second*micrometer**3))
    s2  = std2[ind] / (Joule/(second*micrometer**3))

    EF1,EF2,Mu,Temp    = parameters[ind[0],-4:]
    omega1,omega2 = parameters[ind[0],:2]
    
    tau = parameters[ind,2]
    AS  = argsort(tau)
    
    tau = tau[AS]
    p1,p2,s1,s2  = (x[AS] for x in (p1,p2,s1,s2))

    figure(1)
    errorbar(tau/picosecond,p2/(Joule/(second*micrometer**3)),yerr=s2,fmt=".-",capsize=3,lw=0.9)

    figure(2)
    errorbar(tau/picosecond,p1/(Joule/(second*micrometer**3)),yerr=s1,fmt=".-",capsize=3)
    legendlist.append("\t$v_{\\rm F} = "+f"{series_list[ns,2]/(1e6*meter/second):.2}$, \t$\mu={series_list[ns,-2]:.4}$, \t$\omega_2/\omega_1 = {omega2/omega1:.9}$")


figure(1)
legend(legendlist)
plot([0,800],[0,0],'--k')
xlabel("relaxation time, $\\tau$, picoseconds")
ylabel("Amplification power, $W/\mu m^3$")
title("Amplification of mode 1 vs. $\\tau$.\n$H(k) = v_{\\rm F}  \,\\vec{k}\cdot\\vec{\sigma} +0.8 v_{\\rm F}\, k_z$, "+f"$E_1 = {EF1/(10**6*Volt/meter):.2} \cdot 10^6$ V/m, $E_2 = {EF2/(10**6*Volt/meter):.2}\cdot 10^6$ V/m")
ylim((-0.15,0.15))
plt = gcf()
plt.set_size_inches(8,5)
savefig("../Figures/amplification_power_1.pdf")#,dpi=200)

figure(2)
legend(legendlist)
plot([0,800],[0,0],'--k')
xlabel("relaxation time, $\\tau$, picoseconds")
ylabel("Amplification power, $W/\mu m^3$")
title("Amplification of mode 2 vs. $\\tau$.\n$H(k) = v_{\\rm F}  \,\\vec{k}\cdot\\vec{\sigma} +0.8 v_{\\rm F}\, k_z$, "+f"$E_1 = {EF1/(10**6*Volt/meter):.2} \cdot 10^6$ V/m, $E_2 = {EF2/(10**6*Volt/meter):.2}\cdot 10^6$ V/m")
ylim((-0.2,0.04))
plt = gcf()
plt.set_size_inches(8,5)
savefig("../Figures/amplification_power_2.pdf")#,dpi=200)

## $v=(v_F,v_F,$E_1=
##      ")
#errorbar(tau_2e5/picosecond,P1_2e5,yerr=S1_2e5,fmt='s-')
#errorbar(tau_5e5/picosecond,P1_5e5,yerr=S1_5e5,fmt='s-')
#errorbar(tau_1e6/picosecond,P1_1e6,yerr=S1_1e6,fmt='s-')
#legend(("$v_{\\rm F} = 2\cdot 10^5$ m/s, $\mu=24$ meV",
#        "$v_{\\rm F} = 5\cdot 10^5$ m/s, $\mu=60$ meV",
#        "$v_{\\rm F} = 1\cdot 10^6$ m/s, $\mu=115$ meV"))
#plot([0,1.1*amax(tau/picosecond)],[0,0],':k')
#xlabel("relaxation time, $\\tau$, picoseconds")
#ylabel("Amplification power, $W/\mu m^3$")
#title("Amplification power vs. $\\tau$ (mode 2). $H(k) = v_{\\rm F}  \,\\vec{k}\cdot\\vec{\sigma} +0.8 v_{\\rm F}\, k_z$. \n"+f"$E_1 = {EF1/(10**6*Volt/meter):.2} \cdot 10^6$ V/m, $E_2 = {EF2/(10**6*Volt/meter):.2}\cdot 10^6$ V/m, "+
#      f"$\omega_1 = {omega1/THz:2.3}$ THz,$\omega_2 = {omega2/THz:2.3}$ THz")
#ylim((-0.2,0.04))
#plt = gcf()
#plt.set_size_inches(8,5)
#savefig("../Figures/amplification_power_2.png",dpi=200)



#    
#
#ind_2e5  = where(abs(vf-2e5*meter/second)<1e-8)[0]
#ind_5e5  = where(abs(vf-5e5*meter/second)<1e-8)[0]
#ind_1e6  = where(abs(vf-1e6*meter/second)<1e-8)[0]
#
#P1_2e5 = P1[ind_2e5]/(Joule/(second*micrometer**3))
#P2_2e5 = P2[ind_2e5]/(Joule/(second*micrometer**3))
#S1_2e5 = std1[ind_2e5]/(Joule/(second*micrometer**3))
#S2_2e5 = std2[ind_2e5]/(Joule/(second*micrometer**3))
#tau_2e5 = tau[ind_2e5]
#
#P1_5e5 = P1[ind_5e5]/(Joule/(second*micrometer**3))
#P2_5e5 = P2[ind_5e5]/(Joule/(second*micrometer**3))
#S1_5e5 = std1[ind_5e5]/(Joule/(second*micrometer**3))
#S2_5e5 = std2[ind_5e5]/(Joule/(second*micrometer**3))
#tau_5e5 = tau[ind_5e5]
#
#P1_1e6= P1[ind_1e6] /(Joule/(second*micrometer**3))
#P2_1e6= P2[ind_1e6] /(Joule/(second*micrometer**3))
#S1_1e6 = std1[ind_1e6]/(Joule/(second*micrometer**3))
#S2_1e6 = std2[ind_1e6]/(Joule/(second*micrometer**3))
#tau_1e6 = tau[ind_1e6]
