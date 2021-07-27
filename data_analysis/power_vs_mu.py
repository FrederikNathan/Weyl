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

p1list = []
denslist=[]
p2list = []
mulist = []
P1list = []
P2list = []
Denslist=[]
mupclist=[]
Muarray=[]
n0  = 70

p0 = pl[n0,:]
pprint(n=n0)


klist = []
nlist = []
ind = array([x for x in range(0,9)]+[10])
refparm = p0[ind]
outparmlist= []
for n in range(0,NPARMS):
    p = pl[n,ind]
    
    if norm(p-refparm)<1e-9 and NDP[n]<1000:
        p1,p2,dens = DP.compute_power(n,disp=0)

        if NDP[n]>10:
            omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp = pl[n,:]
            nlist.append(n)
            mulist.append(1*Mu)
            
            p1list.append(p1)
            p2list.append(p2)
            denslist.append(dens)
#            outparmlist.append(pl[n,:])
        klist = concatenate((klist,DP.kpoints[:,2]))
        P1list = concatenate((P1list,DP.P1_vec))
        P2list = concatenate((P2list,DP.P2_vec))      
        Denslist  = concatenate((Denslist,DP.Dens_vec))  
        Muarray = concatenate((Muarray,array([pl[n,9]]*NDP[n])))


     
p1list,p2list,mulist,denslist = [array(x) for x in (p1list,p2list,mulist,denslist)]
AS = argsort(mulist)
p1list,p2list,mulist,denslist=  [x[AS] for x in [p1list,p2list,mulist,denslist]]

    


xi_mu = linspace(amin(Muarray),amax(Muarray),200)
xi_k = linspace(-0.2,amax(klist),200)
Center = (mean(xi_mu),mean(xi_k))
mug,kg= meshgrid(xi_mu,xi_k)
Dmu = amax(Muarray)-amin(Muarray)
Dk = 0.4
xi = (mug/Dmu,kg/Dk)
points = array([Muarray/Dmu,klist/Dk]).T#,axis=1)

#Center = (mean(xi_mu)/Dmu,mean(xi_k)/Dk)
#
#Data = array([P1list,P2list,Denslist]).T
#(mug,kg,kz),Ng,Data_grid = kgrid.compute_grid(points,Data,Center,1,(1,1,1),order=4)

#p1g = Data_grid[:,:,:,0]
#p2g = Data_grid[:,:,:,1]
#dg = Data_grid[:,:,:,2]
p1g = scint.griddata(points,P1list,xi,fill_value=0,method="nearest")#,rescale=1)
p2g = scint.griddata(points,P2list,xi,fill_value=0,method="nearest")#,rescale=1)
dg = scint.griddata(points,Denslist,xi,fill_value=0,method="nearest")#,rescale=1)
#
P0  =omega1*omega2/(2*pi)


figure(1)

plot(mulist,p1list/P0,'b.-');plot(mulist,p2list/P0,'r.-')
plot(mulist,-(p1list+p2list)/P0,'k.-');
plot(mulist,0*mulist,'--k')
ylabel("Power/P0")
xlabel("Mu, meV")
title(f"Power vs mu, from states on z-axis\n $v_F$ = {vF/(meter/second):.2} m/s, $\\tau$={tau/picosecond:.2} ps")
legend(["P1","P2","Dissipation","zero"])

figure(2)
plot(mulist,denslist,'g.-')
ylabel("Carrier concentration [$Å^{-3}$]")
xlabel("Mu,meV")
title(f"Estimated carrier concentration vs mu\n $v_F$ = {vF/(meter/second):.2} m/s, $\\tau$={tau/picosecond:.2} ps")


figure(3)
VMAX= max(amax(abs(p1g/P0)),amax(abs(p2g/P0)))
pcolormesh(mug,kg,p1g/P0,cmap='bwr',vmin=-VMAX,vmax=VMAX);colorbar()
xlabel("$\mu$")
ylabel("$k_z$")
title(f"Pumping into mode 1, from states on z-axis\n $v_F$ = {vF/(meter/second):.2} m/s, $\\tau$={tau/picosecond:.2} ps")
figure(4)
pcolormesh(mug,kg,p2g/P0,cmap='bwr',vmin=-VMAX,vmax=VMAX);colorbar()
#plot(Muarray,klist,'.k')

xlabel("$\mu$")
ylabel("$k_z$")
title(f"Pumping into mode 2, from states on z-axis\n $v_F$ = {vF/(meter/second):.2} m/s, $\\tau$={tau/picosecond:.2} ps")
figure(5)
#pcolormesh(mg,kg,P2list/P0,cmap='bwr');colorbar()

xlabel("$\mu$")
ylabel("$k_z$")
title(f"Dissipation into mode 2, from states on z-axis\n v_F = {vF/(meter/second):.2} m/s, $\\tau$={tau/picosecond:.2} ps")

VV = amax(abs((p1g+p2g)/P0))
pcolormesh(mug,kg,-(p1g+p2g)/P0,cmap='bwr',vmin=-VV,vmax=VV);colorbar()


figure(6)
pcolormesh(mug,kg,dg,cmap='jet');colorbar()

#plot(Muarray,klist,'.w')
xlabel("$\mu$")
ylabel("$k_z$")
title(f"Density on z-axis\n $v_F$ = {vF/(meter/second):.2} m/s, $\\tau$={tau/picosecond:.2} ps")
show()

#except:
#    pass