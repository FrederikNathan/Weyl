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
from . import basic as B
from Units import * 
import Queue_generator as Q

QueueDir = "../../Queues/"

kzlist = linspace(-0.1,0.1,25)*(Å**-1)
kylist = [0]
kxlist = [0]


klist = array([[[[kx,ky,kz] for kz in kzlist] for ky in kylist] for kx in kxlist])


# =============================================================================
# Parameters
# =============================================================================

omega2 = 20*THz
omega1 = 0.61803398875*omega2
tau    = 50*picosecond
vF     = 1e5*meter/second

EF2 = 0.6*1.5*2e6*Volt/meter
EF1 = 0.6*1.25*1.2e6*Volt/meter

A1 = EF1/omega1
A2 = EF2/omega2




Mu =15*meV
Temp  = 25*Kelvin;

V0 = array([0,0,0.8*vF])*1
[V0x,V0y,V0z] = V0

NPmax = 30



# =============================================================================
# K grid
# =============================================================================

NK= prod(shape(klist)[:3])
klist=klist.reshape((NK,3))

Perm = npr.permutation(NK)
klist=klist[Perm,:]

#NK = len(klist)

parameters = 1*array([omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp,NPmax])
parameterlist = array([parameters]*NK)


#Q.add_to_queue(parameterlist,klist)
