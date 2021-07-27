#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:31:50 2020

@author: frederik
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
from Units import * 
import weyl_queue as Q
import queue_generators as QG

QueueDir = "../Queues/"
Name     = "nearer_commensurate_2_3_bigsweep_vf1e6_tau200ps"
os.chdir("..")

# =============================================================================
# Parameters
# =============================================================================

dk_i = 0.0015
dk_o = 0.003
phi_res = 48



omega2 = 20*THz
omega1 = omega2/1.50001
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


parameters = 1*array([omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp])
parameterlist = array([parameters]*NK)
Q.add_to_queue(parameterlist,klist,Name=Name)
