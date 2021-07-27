#!/usr/bin/env python3
# -*- coding: utf-8 -*-


plot = 1 
savefig = 0

import os 
import sys
sys.path.append("/Users/frederik/Dropbox/Academics/Active/WeylNumerics/Code")
os.chdir('../')
print(os.getcwd())
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
import pandas as pan
from Units import *
import kgrid as kgrid 
from DataProcessingMasterEquation import *
import ProcessData_master_equation_v1 as DP
import weyl_liouvillian_v1 as wl
n0 =137
## Parameters 3 and 4 are really good. 


parm,k,p1,p2,dens,Eeq,Ess,TDS = get_data(n0,disp=False)
omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp = parm
#pprint(n=n0)
P0 = omega1*omega2/(2*pi)
wl.set_parameters(parm)
Neq= []
#for k0 in k:
#    
#    Neq.append(wl.get_average_equlibrium_density(k0,300,Mu))
#Neq = array(Neq) 

#if amax(p1+p2)>0:
#    raise ValueError
kz = k [:,2]
AS = argsort(kz)
kz = sort(kz)
#pprint(n=n0)

figure(1)
dis =(-(p2[AS]+p1[AS])/P0)
sd = sign(dis)**2
plot(kz,p1[AS]/P0*sd,'.-b')
plot(kz,p2[AS]/P0*sd,'.-r')
plot(kz,(dis),'.-k')
xlabel("$k_z$")
ylabel("Power/$P_0$")
title("Power vs. kz")
legend(["P1","P2","Dis"])
#ylim((-1.1,1.1))
#xlim((-0.15,0.15))

figure(2)
plot(kz,dens[AS],'.-g')
#plot(kz,Neq[AS],'.-r')

#plot(kz,p2[AS]/P0,'.-b')
#plot(kz,-(p2[AS]+p1[AS])/P0,'.-k')
xlabel("$k_z$")
ylabel("Density")
title("Density vs. kz")
ylim((0,2.2))
xlim((-0.15,0.15))

#xlim((-0.2,0.2))
   
    
    
figure(3)
title("Energy in equlibribum and steady state, as a function of $k_z$")
plot(kz,Ess[AS],'.r-')
plot(kz,Eeq[AS],'-k')
ylabel("Energy [meV]")
xlabel("$k_z$")

figure(4)
title("Dissipation rate computed from two methods")
plot(kz,(Ess[AS]-Eeq[AS])/(tau*P0),".-m")
plot(kz,-(p1[AS]+p2[AS])/P0,".-g")
legend(["From steady state","from pumping power"])
ylabel("Dissipation rate [$P_0$]")
xlabel("$k_z$")
plot(kz,TDS[AS],'+k')
plot(kz,0*kz,'--k')
#figure(5)
#plot(kz,dens[AS],'.-g')
#ylim((-10,10))#plot(kz,Eeq[AS],'.g')

figure(5)
kmin= amin(k[:,2])
kmax = amax(k[:,2])

klist = zeros((300,3))
kzlist = linspace(kmin,kmax,300)
klist[:,2]=kzlist
wl.set_parameters(parm)
[E1list,E2list] = wl.get_E1(klist).T
Emax = max(amax(abs(E1list)),amax(abs(E2list)))
#
figure(6)
plot(kzlist*Å,E1list,'k')#/max(abs(E1list)))
plot(kzlist*Å,E2list,'k')#/max(abs(E1list)))
plot(kzlist*Å,Mu+0*kzlist,'r')
xlabel("$k_z$")
ylabel("Energy, meV")
title("Energy bands and chemical potential on z-axis")
show()

DP.compute_power(n0)
#DP.compute_power(n0)
#legend(["P1","P2","Dis"])
#p1list = []
#p1list = []
#denslist=[]
#p2list = []
#mulist = []
#P1list = []
#P2list = []
#Denslist=[]
#mupclist=[]
#n0  = 17
#p0 = pl[n0,:]
#refparm = p0[:-2]
#pprint(n=n0)
#
#for n in range(0,NPARMS):
#    p = pl[n,:-1]
#    
#    if norm(p[:-1]-refparm)<1e-9:
#        omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu = p
#
#        mulist.append(1*Mu)
#        
#        p1,p2,dens = DP.compute_power(n,disp=0)
#        p1list.append(p1)
#        p2list.append(p2)x
#        denslist.append(dens)
#        
#        if len(DP.kpoints)==70:
#            ASK = argsort(DP.kpoints[:,2])
#            P1list.append(1*DP.P1_vec[ASK])
#            P2list.append(1*DP.P2_vec[ASK])
#            Denslist.append(1*DP.Dens_vec[ASK])
#            mupclist.append(Mu)
#            kp= DP.kpoints[:,2]
#
#p1list,p2list,mulist,denslist = [array(x) for x in (p1list,p2list,mulist,denslist)]
#P1list = array(P1list)
#P2list = array(P2list)
#Denslist = array(Denslist)
#
#P0  =omega1*omega2/(2*pi)
#AS = argsort(mulist)
#p1list,p2list,mulist,denslist=  [x[AS] for x in [p1list,p2list,mulist,denslist]]
##try:
#mumpclist= array(mupclist)
#ASP = argsort(mupclist)
#try:
#    
#    P1list = P1list[ASP,:]
#    P2list = P2list[ASP,:]
#    Denslist  = Denslist[ASP,:]
#
#    kg,mg = meshgrid(sort(kp),sort(mupclist))
#except:
#    pass
#figure(1)
#
#plot(mulist,p1list/P0,'b.-');plot(mulist,p2list/P0,'r.-')
#plot(mulist,0*mulist,'--k')
#ylabel("Power/P0")
#xlabel("Mu, meV")
#title("Power vs mu, from states on z-axis")
#legend(["P1","P2","zero"])
#
#figure(2)
#plot(mulist,denslist,'g.-')
#ylabel("density")
#xlabel("Mu,meV")
#title("Densityvs mu")
#
#figure(3)
#pcolormesh(mg,kg,P1list/P0,cmap='bwr');colorbar()
#xlabel("$\mu$")
#ylabel("$k_z$")
#title("Pumping into mode 1, from states on z-axis")
#figure(4)
#pcolormesh(mg,kg,P2list/P0,cmap='bwr');colorbar()
#
#xlabel("$\mu$")
#ylabel("$k_z$")
#title("Pumping into mode 2, from states on z-axis")
#figure(5)
#pcolormesh(mg,kg,Denslist,cmap='jet');colorbar()
#
#xlabel("$\mu$")
#ylabel("$k_z$")
#title("Density on z-axis")
#show()