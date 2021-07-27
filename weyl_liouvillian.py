#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 09:10:23 2020

@author: frederik

Basis vectors are:
     
|0> = |00>
|1> = |u0>
|2> = |d0>
|3> = |du>

We work in units where hbar = e = 1.
Energies and times are in units of meV.
Length is in unit of Å.


Good parameters:
    
omega1 = 1*THz
omega2 = 0.618*omega1
tau    = 3000*picosecond
vF     = 1e5*meter/second

EF1 = 2e5*Volt/meter
EF2 = 1.2e5*Volt/meter

Mu =25*meV

NP = 25

V0 = array([0,0,0.9*vF])*1


v1: allowing for NP1 and NP2 to be different
"""
import os 

NUM_THREADS = 1

os.environ["OMP_NUM_THREADS"]        = str(NUM_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(NUM_THREADS)
os.environ["NUMEXPR_NUM_THREADS"]    = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"]        = str(NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"]   = str(NUM_THREADS)
#from scipy import *
from scipy.linalg import * 
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import * 
#from matplotlib.pyplot import *
from numpy.fft import *
import vectorization as ve
import numpy.random as npr
import scipy.optimize as optimize
import basic as B
from Units import *
import recursive_greens_function_v1 as rgf


# =============================================================================
# Conversion between vector spaces
# =============================================================================
def cvec_to_mat(cvec):
    """
    Compute Fock-space mat of 6-index vector (cvector)
    """
    D = len(shape(cvec))
    
    if D==2:
        
        NL = shape(cvec)[0]
        
        C = zeros((NL,16),dtype=complex)
        C[:,Ind]=cvec
        return C.reshape((NL,4,4))
    
    elif D==1:
        C = zeros((16),dtype=complex)
        C[Ind]=cvec
        return ve.vec_to_mat(C)
        
def get_1p(mat):
    S = len(shape(mat))
    
    if S==2:
        
        return mat[1:3,1:3]
    elif S==3:
        return mat[:,1:3,1:3]
    
def get_0p(mat):
    S = len(shape(mat))
    
    if S==2: 
        return mat[0,0]
    elif S==3:
        return mat[:,0,0]

def get_2p(mat):
    S = len(shape(mat))
    
    if S==2:
        
        return mat[3,3]
    elif S==3:
        return mat[:,3,3]
    
    

# =============================================================================
# Model-defining functions
# =============================================================================

def get_A(phi1,phi2):
    """
    Get vector potential as a function of the modes' phases
    """
    
    return array([A1  * sin(phi1),  A2  * sin(phi2),-A1 * cos(phi1)-A2*cos(phi2)])

def get_Electric_field(phi1,phi2):
    return array([omega1*A1*cos(phi1),omega2*A2*cos(phi2),omega1*A1*sin(phi1)+omega2*A2*cos(phi1)])

def get_h(k):
    """ 
    get array of Bltoch hamiltonians in array specified by k. kx[n] = k[n,0] and so on
    """ 

    if len(shape(k))==2:
        
        kx = k[:,0:1]
        ky = k[:,1:2]
        kz = k[:,2:3]
    
        NL = len(kx)
        H1 =  reshape(vF*(kx*sx+ky*sy+kz*sz),(NL,2,2))
        H0 =  outer(k@V0,I2).reshape(NL,2,2)
        return H0+H1
    else:
        [kx,ky,kz]=k
        H1 = vF*(kx*SX+ky*SY+kz*SZ)
        H0 = I2 * (k@V0)
        return H0+H1

def get_hc(k):
    """ 
    get commutator with h(k) in one-particle subspace
    """
    
    h = get_h(k)
    
    return ve.lm(h)-ve.rm(h)


def get_E1(k):
    """
    return energies of one-particle states
    """

    
    E0 = k@V0

    if len(shape(k))==2:
        E1 = vF*norm(k,axis=1)
    else:
        E1 = vF*norm(k)
        
    
    Em = E0-E1
    Ep = E0+E1
    
    return array([Em,Ep]).T
    

def get_h_components():
    """
    Get tunneling matrices in Fock space, corresponding to jumps in the postive n1 and n2 directions
    """
        
    h10 = vF*(-1/(2j)*A1*SX  -  (1/2)*A1*SZ)
    h01 = vF*(-1/(2j)*A2*SY  -  (1/2)*A2*SZ)
    
    H10 = bloch_to_fock(h10)
    H01 = bloch_to_fock(h01)
    
    return H10,H01


def get_E(k,mu=0):
    """
    return energies of many-particle states.
    [E0,E1,E2,E3]
    E0: zero-particle state
    E3: two-particle state
    E1: lower band of 1 particle state
    E2: upper band of 1 particle state
    """
    
    In = get_E1(k)-mu
    NL = shape(k)[0]
    E0 = zeros(NL)
    E1 = In[:,0]
    E2 = In[:,1]
    E3 = E1+E2
    
    return array([E0,E1,E2,E3]).T


# =============================================================================
# Functions relating to equilibrium distribution
# ============================================================================='
    
def get_rhoeq0(h,band=0):
    """
    get equilibrium position of single-particle density matrix. 
    """
    NL = shape(h)[0]
    
    E0 = trace(h,axis1=1,axis2=2)
    
    h = h-reshape(outer(0.5*E0,I2),(NL,2,2))
    
    I22  = reshape(outer(ones(NL),I2),(NL,2,2))
    
    Det = h[:,0,0]*h[:,1,1]-h[:,1,0]*h[:,0,1]
    
    E1 =sqrt(-1*Det).reshape((NL,1,1))
    
    if band==0:
        
        out = 0.5*(I22-h/E1)
    else:
        out = 0.5*(I22+h/E1)
    return out
    

def get_rhoeq(k,mu=0):
    """
    Get Fock-space equilibrium density matrix at momenta in klist, with chemical potential mu. 
    For now, at zero temperature.
    """
    NL = shape(k)[0]
    
    E = get_E(k,mu=mu)
    A0 = 1*argmin(E,axis=1)
    
    BoltzmanWeight = exp(-(E-amin(E,axis=1).reshape((NL,1)))/Temp)
    PartitionFunction = sum(BoltzmanWeight,axis=1)
    BoltzmanWeight = BoltzmanWeight/PartitionFunction.reshape((NL,1))
    
    A00 = argmax(BoltzmanWeight,axis=1)
    
    assert sum(abs(A0-A00))==0
    
    R1 = zeros((NL,4,4),dtype=complex)
    

    
    # one-particle density matrices
    r1 = get_rhoeq0(get_h(k))
    r2 = get_rhoeq0(get_h(k),band=1)
    

    R0 = BoltzmanWeight[:,0]
    R2 = BoltzmanWeight[:,3]
    R1 = r1*BoltzmanWeight[:,1].reshape((NL,1,1))
    R1 += r2*BoltzmanWeight[:,2].reshape((NL,1,1))
    
    ### Round to 13 digits to avoid overflow errors
    
    return R0,R1,R2




def bloch_to_fock(mat,fill=0):
    """ 
    Return Fock space matrix from Bloch space matrix
    """
    
    Out = 1*ZM
    Out[1:3,1:3]=mat
    Out[0,0]=fill
    Out[3,3]=fill
    
    return Out

def get_kgrid(Nphi,k0=array([0,0,0])):
    """ 
    Get grid of k-points reached by vector potential, with phase resolution Nphi
    """

    phirange = arange(0,Nphi)/Nphi * 2*pi
    
    phi1,phi2 = outer(phirange,ones(Nphi)),outer(ones(Nphi),phirange)
    
    Phi1 = phi1.flatten()
    Phi2 = phi2.flatten()
    
    NPhi = len(Phi1)
    A = get_A(Phi1,Phi2)
    K0 = outer(k0,ones(NPhi))
    
    return (K0 + A).T


def get_fourier_rhoeq(k0,mu=0,Nphi=30):
    """
    Get fourier transform of Rho_eq(k+A(\phi_1,\phi_2),mu), wrt. \phi_1,\phi_2
    """
    
    K = get_kgrid(Nphi,k0)
#    global R0,R1,R2,F0,F1,F2
    R0,R1,R2 = get_rhoeq(K,mu=mu)
    
    R0 = R0.reshape((Nphi,Nphi))
    F0 = ifftn(R0,axes=(0,1)) 
    
    R2 = R2.reshape((Nphi,Nphi))
    F2 = ifftn(R2,axes=(0,1)) 
    
    R1 = R1.reshape((Nphi,Nphi,2,2))
    F1 = ifftn(R1,axes=(0,1))
    return F0,F1,F2 


    
    
# =============================================================================
# Useful matrices on photon space
# =============================================================================
def get_nvec(NP):
    """
    Vector that counts number of photons
    """
    Nvec = arange(0,NP)-NP//2
    return Nvec
def get_IP(NP1,NP2):
    """
    Identity operator on photon space
    """    
    IP1  = sp.eye(NP1,dtype=complex,format="csc")
    IP2  = sp.eye(NP2,dtype=complex,format="csc")

    return IP1,IP2

def get_d(NP1,NP2):
    """ 
    Counting operators on photon space
    """
    Nvec_1 = get_nvec(NP1)
    Nvec_2 = get_nvec(NP2)
    IP1,IP2 = get_IP(NP1,NP2)
    D1 = sp.csc_matrix(diag(Nvec_1),dtype=complex)
    D2 = sp.csc_matrix(diag(Nvec_2),dtype=complex)
    
    D1 = sp.kron(D1,IP2)
    D2 = sp.kron(IP1,D2)
    
    return D1,D2 

def get_t(NP1,NP2):
    """
    Translation operator on photon space
    """
    IP1,IP2 = get_IP(NP1,NP2)
    T1  = sp.csc_matrix(roll(IP1.toarray(),1,axis=0),dtype=complex)
    T2  = sp.csc_matrix(roll(IP2.toarray(),1,axis=0),dtype=complex)
    
    T1[-1,0] = 0 
    T1[0,-1] = 0 
    
    T2[0,-1] = 0 
    T2[0,-1] = 0 
    
    T1 = sp.kron(T1,IP2)
    T2 = sp.kron(IP1,T2)

    return T1,T2 

def get_IF(NP1,NP2):
    """
    Full identity operator on entire space
    """
    return sp.eye((4*NP1*NP2),dtype=complex,format="csc")
    

# =============================================================================
# Liouvillian
# =============================================================================

def get_liouvillian(k):
    """ 
    Get Liouvillian at specific point
    """

def get_vn_liouvillian(k,NP1,NP2):
    """
    Get Liouvillian compoennt from -i[H,\rho] (von Neuman component)
    """
    
    
    T1,T2 = get_t(NP1,NP2)
    D1,D2 = get_d(NP1,NP2)
    IP1,IP2 = get_IP(NP1,NP2)
    
    IPP = sp.kron(IP1,IP2)

    H0 = bloch_to_fock(get_h(k))
    [H10,H01] = get_h_components()
    
    L_H00 = -1j*(ve.lm(H0)-ve.rm(H0))[Ind,:][:,Ind]
    L_H10 = -1j*(ve.lm(H10)-ve.rm(H10))[Ind,:][:,Ind]
    L_H01 = -1j*(ve.lm(H01)-ve.rm(H01))[Ind,:][:,Ind]
    
    [L_H00,L_H10,L_H01] = [sp.csc_matrix(x) for x in [L_H00,L_H10,L_H01]]
    ## Vectorized Fock space Liouvillian
    
    L_H =  sp.kron(IPP,L_H00) 
    L_H += sp.kron(T1,L_H10) - sp.kron(T1.T,L_H10.conj().T)
    L_H += sp.kron(T2,L_H01) - sp.kron(T2.T,L_H01.conj().T)

    return L_H

def get_floquet_liouvillian(NP1,NP2):
    """
    return Floqut component (-\omega_1 n - \omega_2 m )
    """
    D1,D2 = get_d(NP1,NP2)
    L_D = 1j* sp.kron(omega1*D1 + omega2*D2,eye(4))

    return L_D

def get_relax_liouvillian(NP1,NP2):
    """
    Get term from relaxation: -1/\tau * \rho
    """
    IF = get_IF(NP1,NP2)
    return -(1/tau)*IF

  
def get_liouvillian(k,NP1,NP2):
    """ 
    Get full Liouvillian in Floquet space, corresponding to -i[H(t),\rho] - 1/\tau \rho.
    """
    L_H = get_vn_liouvillian(k,NP1,NP2)    
    L_D = get_floquet_liouvillian(NP1,NP2)
    L_R = get_relax_liouvillian(NP1,NP2)
    
    L = L_H + L_D + L_R 
    
    return sp.csr_matrix(L)

def set_parameters(parameters):
    global omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp

    [omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp]=parameters
 
    global V0,P0,A1,A2
    V0 = array([V0x,V0y,V0z])
    P0 = omega1*omega2/(2*pi)
    
    
    ## vector field amplitude 
    A1 = EF1/omega1
    A2 = EF2/omega2

    
    global SX,SY,SZ,I2,sx,sy,sz,i2,ZM,Ind,I6,IP,IPP,IF,NPP,DH

    
    [SX,SY,SZ,I2] = [ve.SX,ve.SY,ve.SZ,ve.I2]
    [sx,sy,sz,i2] = [q.flatten() for q in [SX,SY,SZ,I2]]
    
    ZM = zeros((4,4),dtype=complex)
    
    # Indices in vectorized matrix space, where \rho may be nonzero. (we restrict ourselves to this subspace)
    Ind = array([5,6,9,10])


def get_n1n2(NP1,NP2):
    
    Nvec_1 = get_nvec(NP1)
    Nvec_2 = get_nvec(NP2)
    N1 = kron(Nvec_1,ones(NP2)).astype(int)#,ones(NP**2)
    N2 = kron(ones(NP1),Nvec_2).astype(int)#,ones(NP**2)
 
    return N1,N2 


if __name__=="__main__":
    
    omega1 = 20*THz
    omega2 = 0.61803398875*omega1
    tau    = 50*picosecond
    vF     = 7e5*meter/second
    
    EF1 = 0.6*1.5*2e6*Volt/meter
    EF2 = 0.6*1.25*1.2e6*Volt/meter
    
    Mu =15*meV*0
    Temp  = 25*Kelvin;
    
    V0 = array([0,0,0.8*vF])*1
    [V0x,V0y,V0z] = V0
    
    parameters = [omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp]
    
    set_parameters(parameters)
    
    NP1 = 3 
    NP2 = 5
    k = array([0.01,0,0])
    L = get_liouvillian(k,NP1,NP2)