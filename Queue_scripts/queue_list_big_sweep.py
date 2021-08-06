#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:31:50 2020

@author: frederik

Script for generating list of queues. Generating a bigsweep for each parameter 
set in the list.

Naming convention: _quantities indicate which things are varied 


"""
import os 
import sys

sys.path.append("../")
from scipy import *
from scipy.linalg import * 
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy import * 
import numpy.random as npr
import scipy.optimize as optimize

from units import * 
import weyl_queue as Q
import queue_generators as QG

QueueDir = "../Queues/"
QueueListDir = "../QueueLists/"
queue_list_name     = "exact_commensurate_tau"
# queue_list_name = "test"
os.chdir("..")

# =============================================================================
# Parameters
# =============================================================================

dk_i = 0.0015
dk_o = 0.003
phi_res = 48

omega2 = 20*THz
omega1 = omega2/1.5

taulist = [10*picosecond,20*picosecond,50*picosecond,100*picosecond,150*picosecond,200*picosecond,300*picosecond,500*picosecond]
tau    = 200*picosecond
vF     = 1e6*meter/second

EF2 = 0.6*1.5*2e6*Volt/meter
EF1 = 0.6*1.25*1.2e6*Volt/meter

A1 = EF1/omega1
A2 = EF2/omega2

Mu =115*meV
Temp  = 20*Kelvin;

V0 = array([0,0,0.8*vF])*1
[V0x,V0y,V0z] = V0

#NPmax = 30


# npr.seed(0)
klist = QG.get_kgrid(phi_res,phi_res,dk_i,dk_o)
NK = shape(klist)[0]
Perm = npr.permutation(NK)
klist=klist[Perm,:]



queue_names = []
parameter_list = []
ntau = 0
for tau in taulist:
    queue_name = queue_list_name+f"_{ntau}"
    
    

    parameters = 1*array([omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp])
    parameterlist = array([parameters]*NK)
    Q.add_to_queue(parameterlist,klist,Name=queue_name,prompt=False)
    
    queue_names.append(queue_name+".npz")
    parameter_list.append(parameters)
    ntau +=1 
    
    
queue_names = array(queue_names)
parameter_list=array(parameter_list)
savez(QueueListDir+queue_list_name,queue_names = queue_names,parameter_list=parameter_list)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    