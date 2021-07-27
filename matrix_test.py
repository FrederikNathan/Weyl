#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 08:37:33 2020

@author: frederik
"""

from scipy import *
import numpy.random as npr
import basic as B
from matplotlib.pyplot import *

Nlist  = [int(2**n) for n in arange(0,11,0.2)]
Outlist=[]
for N in Nlist:
    
    #N =10
    NM = 1000
    A = npr.rand(N,2,2)
    
    
    Out = 1*A
    
    
    B.tic()
    for n in range(0,NM):
        
        Out = A@(Out)
        
    Out = B.toc(disp=0)/(N*NM)
    
#    print(f"With N = {N}:")
#    print("time per multiplication")
#    print(f"{Out:.4} s")
    Outlist.append(Out)
    
    
Nlist = array(Nlist)
Outlist= array(Outlist)/1e-6
Outlist = Outlist/Outlist[0]

plot(Nlist,1/(Outlist))
ylim((0,amax(1/Outlist)))
dlOdN = (log(Outlist)[1:]-log(Outlist)[:-1])/(Nlist[1:]-Nlist[:-1])

figure(2)
nn = 0.5*(Nlist[:-1]+Nlist[1:])
plot(nn,dlOdN+1/(100+nn))
plot(nn,0*nn,'--k')

