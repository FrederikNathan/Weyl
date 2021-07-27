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



def get_outer_kgrid(phi_res,dk):

    # Use cylindrical coordinates
    kzlist = arange(-0.38,0.15,dk)*(Å**-1)
    krlist = arange(dk,0.15,dk)*(Å**-1)
    philist = linspace(0,pi/2,phi_res)
    
    NK = len(kzlist)*len(krlist)*len(philist)+len(kzlist)
    
    klist = zeros((NK,3))
    
    kzlist_1 = kzlist[abs(kzlist)>=0.1]
    
    counter = len(kzlist_1)
    
    klist[:len(kzlist_1),2] = kzlist_1
    
    for phi in philist:
        for kr in krlist:
            for kz in kzlist:
                
                if abs(kz)>=0.1 or kr>=0.1:
                    kx = kr*cos(phi)
                    ky = kr*sin(phi)
                    
                    klist[counter,:]=array([kx,ky,kz])
                    counter+=1
    
    NK = counter
    klist = klist[:NK,:]
    return klist

def get_inner_kgrid(phi_res,dk):
    kzlist = arange(-0.1,0.1,dk)*(Å**-1)
    krlist = arange(dk,0.1,dk)*(Å**-1)
    philist = linspace(0,pi/2,phi_res)
    
    NK = len(kzlist)*len(krlist)*len(philist)+len(kzlist)
    
    klist = zeros((NK,3))
    
    counter = len(kzlist)
    klist[:len(kzlist),2] = kzlist
    
    for phi in philist:
        for kr in krlist:
            for kz in kzlist:
                kx = kr*cos(phi)
                ky = kr*sin(phi)
                
                klist[counter,:]=array([kx,ky,kz])
                counter+=1
            
            
    return klist


def get_kgrid(phires_inner,phires_outer,dk_inner,dk_outer):
    ki = get_inner_kgrid(phires_inner,dk_inner)
    ko = get_outer_kgrid(phires_inner,dk_outer)
    
    return concatenate((ki,ko))

#Y = get_kgrid(16,16,0.002,0.008)

