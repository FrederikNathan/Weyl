#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 10:31:47 2018

@author: frederik
"""

import sys
from scipy.linalg import *
from scipy import *

from numpy.random import *
from time import *
import os as os 
import os.path as op
try:
    
    from scipy.misc import factorial as factorial
except ImportError:
    from scipy.special import factorial as factorial 
    
import datetime
import logging as l
#from matplotlib.pyplot import *
#from mpl_toolkits.mplot3d import Axes3D
import numpy.fft as fft

#import basic as B
"""
Master equation solver using vectorization
"""



# =============================================================================
# Basis:
#
# 1:  [[1,0], = |u><u|
#      [0,0]]
#
# 2:  [[0,1], = |u><d|
#      [0,0]]
#
# 3:  [[0,0], = |d><u|
#      [1,0]]
#
# 4:  [[0,0], = |d><d|
#      [0,1]]
#
# =============================================================================

SX = array(([[0,1],[1,0]]),dtype=complex)
SY = array(([[0,-1j],[1j,0]]),dtype=complex)
SZ = array(([[1,0],[0,-1]]),dtype=complex)
I2 = array(([[1,0],[0,1]]),dtype=complex)
    
def lm(Mat):
    # construct superoperator corresponding to left multiplication by Mat
    
    S = shape(Mat)   

        
    D = shape(Mat)[-1]
    I = eye(D)
    
    Out = kron(Mat,I)
    return Out
         


def rm(Mat):
    # construct superoperator corresponding to right multiplication by Mat
    D = shape(Mat)[-1]
    I = eye(D)
    
#    Newmat = 
    Out = kron(I,Mat.swapaxes(-1,-2))

    return Out                   
    
def com(Mat):
    # Construct superoperator corresponding to commutator with Mat ([Mat,*])
    
    return lm(Mat)-rm(Mat)


def mat_to_vec(M):  
    S = shape(M)
    if len(S)==2:
        
        return ravel(M)
    else:
        NS = S[0]
        return reshape(M,(NS,S[1]*S[2]))
    
def vec_to_mat(V):
    D=sqrt(len(V))
    D=int(D+0.1)
    return reshape(V,(D,D))

def get_lindblad(L):
        
    Self_energy= L.conj().T.dot(L)
    
    
    Dissipator = lm(L).dot(rm(L.conj().T))-0.5*(lm(Self_energy)+rm(Self_energy))
    
    return Dissipator
    
def get_trace_vec(dim):
    """ 
    Get vector v corresponding to trace, such that v.dot(X) = trace
    """
    M = eye(dim,dtype=bool)
    return mat_to_vec(M)

if __name__=="__main__":
    
    from basic import * 
    import jump_operator as JO 

    theta= pi/4
    Vz    =5 * GHz
    Temperature = 0.1 * K    
    Lambda = 50* GHz    
    H = Vz*SZ
    
    X = cos(theta)*SZ + sin(theta)*SX
    
    gamma =0.2*MHz 
    
       
    def J(x):
        
        omega_0 = 1 * GHz 
        
        return (x/omega_0)*exp(-x**2 /(2*Lambda**2)) / (1-exp(-x/Temperature))
    

    
    df = 5* MHz 
    freqvec = arange(-20*Lambda,20*Lambda,df)
    
    L = JO.get_jump_operator_static(X,H,J,gamma=gamma)
    LS = JO.get_lamb_shift_static(X,H,J,freqvec,gamma=gamma)*(-1)
    
    Self_energy= L.conj().T.dot(L)
    
    
    Dissipator = -1j*com(H+LS) + lm(L).dot(rm(L.conj().T))-0.5*(lm(Self_energy)+rm(Self_energy))
    
    [E,V]=eig(Dissipator)
    
    
    A0 = argmin(abs(E))
    
    V0 = V[:,A0]
    
    Rho_0  = vec_to_mat(V0)
    
    Rho_0=Rho_0/trace(Rho_0)
    [v0,vx,vy,vz]=[2*real(x) for x in get_pauli_components(Rho_0)]
    
    Gamma,tau = JO.get_lindblad_timescales(J,freqvec)

    print("")
    print("Parameters:")
    print(f"    Temp        =   {Temperature:.4} GHz")
    print(f"    Vz          =   {Vz*1.:.4} GHz")
    print(f"    gamma       =   {gamma:.4} GHz")
    print(f"    theta       =   {theta/pi} pi")
    print(f"    Lambda      =   {Lambda} GHz")
    
    print("")
    print("Time scales:")
    print(f"    Gamma_max   =   {Gamma*gamma:.4} GHz")
    print(f"    tau         =   {tau/ps:.4} picoseconds")
    print(f"    k_markov    =   {tau*Gamma*gamma:.3}")
    print("")
    print("Actual energy scales of the bath: ")
    print(f"    Gamma_act   =   {norm(L,ord=2)**2} GHz")
    print(f"    |LS|        =   {norm(LS-0.5*trace(LS)*eye(2),ord=2)} GHz")
    
#
    print("")
    print("Components of \\rho:")
    print(f"    rho_x       =   {vx}")
    print(f"    rho_y       =   {vy}")    
    print(f"    rho_z       =   {vz}")
    
    
    Lamb_vector =array([real(x) for x in get_pauli_components(LS)])
    HT = Lamb_vector/5
    figure(1)
    clf()
    plot([0,0],[0,Vz/5])
    plot([0,sin(theta)],[0,cos(theta)])
    plot([0,HT[1]],[0,HT[3]])
    plot([0,HT[1]],[0,Vz/5+HT[3]])
    plot([0,vx],[0,vz])
    plot([-1,1],[0,0],'--k')
    plot([0,0],[-1,1],'--k')
    
    thetavec=linspace(0,2*pi,500)
    plot(cos(thetavec),sin(thetavec))
#    xlim((-2,2))
#    ylim((-2,2))
    ax = gca()
    ax.set_aspect("equal")
#    legend(["H0","X","LS","H0+LS","Steady state"])

    title(f"T={Temperature:.4}GHz, Vz={Vz*1.:.4}GHz, gamma={gamma:.4} GHz, $\\theta$={theta:.4}")

    

    Filename = ID_gen()
    savefig(f"Figures/{Filename}.pdf",dpi=500)
    
    
    
    
    
    
    figure(2)
    g = JO.get_jump_correlator(J)
    
    G,tvec=JO.get_time_domain_function(g,freqvec)
    Jvec,tvec = JO.get_time_domain_function(J,freqvec)
    
    AS = argsort(tvec)
    tvec = tvec[AS]
    G=G[AS]
    Jvec=Jvec[AS]
    plot(tvec,real(G),'b')
    plot(tvec,imag(G),'r')
#    plot(tvec,log(abs(G)))
    xlim(-0.15,0.15)
#    xlim(-4,4)
#    ylim(-1e-2,1e-2)
    xlabel("Time, ns")
    ylabel("Value, GHz")
    title(f"Jump correlator: Temp={Temperature}GHz, $\Lambda$={Lambda}GHz, $\omega_0$=1GHz")
    FigName="Jump_Correlator_"+ID_gen()
    savefig(f"Figures/{FigName}.pdf",dpi=500)
    
    figure(3)
    
    plot(tvec,log(abs(G)),'darkviolet')
    plot(tvec,log(abs(Jvec)),'g')
    legend(["Jump correlator","Bath correlator"])
#    plot(tvec,log(abs(G)))
    xlim(-0.15,0.15)
    ylim(-9,8)
#    xlim(-4,4)
#    ylim(-1e-2,1e-2)
    xlabel("Time, ns")
    ylabel("Log(Value/GHz)")
    title(f"Logarithimic plot of Jump and bath correlator\nTemp={Temperature}GHz, $\Lambda$={Lambda}GHz, $\omega_0$=1GHz")
    FigName_2="Correlator_comparison"+ID_gen()
    savefig(f"Figures/{FigName_2}.pdf",dpi=500)