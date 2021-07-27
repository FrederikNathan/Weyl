#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:31:50 2020

@author: frederik
"""
import os 
import sys

import queue_generators as QG
sys.path.append("../")
from scipy import *
from scipy.linalg import * 
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy import * 

import numpy.random as npr
import scipy.optimize as optimize
from Units import * 
import weyl_queue as Q

QueueDir = "../Queues/"
Name     = "bigsweep_outer_vf1e6_tau500ps"
os.chdir("..")

# =============================================================================
# Parameters
# =============================================================================


dk_o = 0.004
phi_res = 6

omega2 = 20*THz
omega1 = 0.61803398875*omega2
tau    = 500*picosecond
vF     = 1e6*meter/second

EF2 = 0.6*1.5*2e6*Volt/meter
EF1 = 0.6*1.25*1.2e6*Volt/meter

A1 = EF1/omega1
A2 = EF2/omega2

Mu =115*meV
Temp  = 20*Kelvin;

V0 = array([0,0,0.8*vF])*1
[V0x,V0y,V0z] = V0

NPmax = 30


npr.seed(0)
klist = QG.get_outer_kgrid(phi_res,dk_o)
NK = shape(klist)[0]
Perm = npr.permutation(NK)
klist=klist[Perm,:]


parameters = 1*array([omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp])
parameterlist = array([parameters]*NK)
Q.add_to_queue(parameterlist,klist,Name=Name)
