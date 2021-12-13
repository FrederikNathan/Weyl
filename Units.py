#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:31:50 2020

@author: frederik

WARNING --- the frequency and time units I used were wrong. 
use picosecond_physical and second_physical. 

Parameters are in "fake" units.

"""
import os 
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

# =============================================================================
# Units
# =============================================================================
meV = 1
Å  = 1 
e_charge = 1 
hbar = 1 
#h = # 1/0.24180 meV/THz = 

Coulomb = e_charge/1.602e-19
Volt  = 1e3 * meV/e_charge

Joule = 1*Volt*Coulomb
THz   = 0.24180*meV #NB! 1 THz in the simulation corresponds to 
THz_physical = 0.6242*meV/hbar
nm     = 10
meter = 1e9*nm
centimeter = 1e-2 *meter
micrometer = 1e-6 *meter
millimeter = 1e-3 *meter
picosecond = 1/THz
second = 1e12*picosecond


picosecond_physical = 1/THz_physical
second_physical=10**12*picosecond_physical

Kelvin = 0.0862 *meV


SX = array([[0,1],[1,0]])
SY = array([[0,-1j],[1j,0]])
SZ = array([[1,0],[0,-1]])
I2 = eye(2)