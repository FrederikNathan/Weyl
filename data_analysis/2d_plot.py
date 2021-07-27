#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 10:09:24 2020

@author: frederikffsgsgwerqghtgfghdfgsafweqqwzxvcvbnm,.lkkkjlFQDDJOIEIFEÆDMMØØ
^Å
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
# import ProcessData_master_equation_v1 as DP

n0 = 156



parameters,klist,P1_list,P2_list,Dens_list,Eeq,Ess,TDS  = get_data(n0)
omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp = parameters
P0 = omega1*omega2/(2*pi)


kmin_z= -0.6
klim_z = (-0.4,0.2)
klim_x = (0,0.2)

ncubes = (1,1,3)
cubewidth = 0.201
order = 7
dk = cubewidth/(2**order)
center = (0.1,-dk/2,-0.1)

k_out,Nlist,Data = kgrid.compute_grid(klist,array([Dens_list,P1_list,P2_list]).T,center,cubewidth,ncubes,order=order)
#k_out = [k_out[n] for n in (0,1,2)]
kx_out, kz_out = k_out[0],k_out[2]
kx_out = concatenate((-kx_out[::-1],kx_out))
xg,zg = meshgrid(kz_out,kx_out)
#xg = concatenate((xg,-xg[::-1,:]))
#zg = concatenate((zg,zg[:,:]))
Data =  concatenate((Data[::-1,:,:,:],Data),axis=0)

S = shape(Data)
N0 = S[1]//2

print(f"Power in xz plane:")
print(f"   Mode 1:  {dk**2*sum(Data[:,N0,:,1]):.4} meV**2/Å^2")
print(f"   Mode 2:  {dk**2*sum(Data[:,N0,:,2]):.4} meV**2/Å^2")


figure(1)
clf()
title("Density of electrons in $k_y=0$ plane")
pcolormesh(xg,zg,Data[:,N0,:,0],cmap="jet",vmin=1);colorbar()
xlabel("$k_z$ [$Å^{-1}$]")
ylabel("$k_x$ [$Å^{-1}$]")
ax=gca()
ax.set_aspect("equal")
show()

figure(2)
clf()
title("Energy pumped into mode 2, from modes in $k_y=0$ plane")
pcolormesh(xg,zg,Data[:,N0,:,1]/P0,cmap="bwr",vmin=-1.0e-0,vmax=1.0e-0);colorbar()
xlabel("$k_z$ [$Å^{-1}$]")
ylabel("$k_x$ [$Å^{-1}$]")
#xlim((-0.1,0.1))
#ylim((-0.1,0.1))

ax=gca()
ax.set_aspect("equal")
figure(3)
clf()
title("Energy pumped into mode 2, from modes in $k_y=0$ plane")
pcolormesh(xg,zg,Data[:,N0,:,2]/P0,cmap="bwr",vmin=-1.1e-0,vmax=1.1e-0);colorbar()
xlabel("$k_z$ [$Å^{-1}$]")
ylabel("$k_x$ [$Å^{-1}$]")

ax=gca()
ax.set_aspect("equal")
figure(4)
clf()
title("Rate of dissipative absorption, from modes in $k_y=0$ plane")
pcolormesh(xg,zg,-(Data[:,N0,:,2]+Data[:,N0,:,1])/P0,cmap="bwr",vmin=-2e-1,vmax=2e-1);colorbar()
xlabel("$k_z$ [$Å^{-1}$]")
ylabel("$k_x$ [$Å^{-1}$]")
ax=gca()
ax.set_aspect("equal")


#ax=gca1()
#ax.set_aspect("equal")
#figure(5)
#title("Number of data points in $k_y=0$ plane")
#pcolormesh(xg,zg,Nlist[:,N0,:],cmap="jet");colorbar()
#xlabel("$k_z$ [$Å^{-1}$]")
#ylabel("$k_x$ [$Å^{-1}$]")
#ax=gca()
#ax.set_aspect("equal")

ax.set_aspect("equal")
figure(5)
title("Data points projected into $k_y=0$ plane")
plot(klist[:,2],klist[:,0],'.k',markersize=1)
plot(klist[:,2],-klist[:,0],'.k',markersize=1)
xlabel("$k_z$ [$Å^{-1}$]")
ylabel("$k_x$ [$Å^{-1}$]")
ax=gca()
ax.set_aspect("equal")

show()