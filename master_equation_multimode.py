#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 09:10:23 2020

@author: frederik

Main module for solving master equation of weyl semimetal

We work in units where hbar = e = 1.
Energies and times are in units of meV.
Length is in unit of Å.

History of module:

v2: clever cropping of photon space. Dynamical change of photon space, to speed up (i.e. only use large photon space when absolutely necessary. )
v3: Using recursive greens function to solve
v3.1: truncating to the one-particle sector.
v4: separated Weyl liouvillian out
v5: efficient hlist to save memory
v6: using time-domain solver
v7: fixed negative dissipation issue
v8: direct computation of <d\phi_2 H> and <d\phi_1 H>
v9: SO(3) implementation of time-domain solver. Using rotating frame interpolator


In case of commensurate frequencies, we average over phase.
"""

NP_MAX                 = 300    # Maximum number of photons before using time domain solver
INITIAL_NP             = 10
NPHI_RGF               = 200    # NPhi used to calculate rho_steady_state with rgf metho
NPHI_TDS               = 200    # Nphi used to calculate steadystate with tds method
CONVERGENCE_TRESHOLD   = 1e-6
TMAX_IN_MODE1_PERIODS  = 2000 # Number of periods of mode 1 to integrate over net (i.e. before division into parallel runs)
# TMAX_IN_MODE1_PERIODS  = 100 # override for testing
SAVE_STEADYSTATE       = False #True

# print("WARNING: saving evolution data ")
import os 
import sys 
from scipy.linalg import * 
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import * 
from numpy.fft import *
import numpy.random as npr
import scipy.optimize as optimize
import gc

import basic as B
from units import *
import weyl_liouvillian as wl
import recursive_greens_function as rgf
import time_domain_solver as tds



def get_rhoeq_vector(k,parameters,NP1,NP2,mu=0,Nphi=NPHI_RGF):
    """
    calculate freuqency-space vector corresponding to \rho_eq(k+A(phi1,phi2)), 
    for phi1,phi2 = 0,1,..Nphi * 2*pi/Nphi
    
    returns 3 flattned arrays of size Nphi**2, 4*Nphi**2, Nphi**2
    
    Here the nth array gives rhoeq in the k-particle sector 
    """
    wl.set_parameters(parameters)
    
    fourier_convergence = False
    while fourier_convergence == False:

        fourier_convergence = True
        R0,R1,R2 = wl.get_fourier_rhoeq(k,mu=mu,Nphi=Nphi)
        
        if NP1<=Nphi/2:
            N1,N2 = wl.get_n1n2(NP1,NP2)
            VR0 = R0[N1,N2].flatten()
            VR1 = R1[N1,N2,:,:].flatten()
            VR2 = R2[N1,N2].flatten()
            
            return VR0,VR1,VR2
        
        
        else:
            Outlist = []
            for R in R0,R1,R2:
                AX =tuple(x for x in range(2,ndim(R)))
                                
                A0 = amin(sum(abs(R),axis=AX))
                    
                if A0>1e-10:
                    Nphi = max(int(1.5*Nphi),Nphi+1)
                    
                    fourier_convergence = False
                    break
                
                
                else:
                        
                        
                    I1 = argmin(sum(abs(R),axis=AX).flatten())
                    I0 = unravel_index(I1,shape(R1)[:2])
                    
                    R[I0[0],I0[1]]=0
            
                    N1,N2 = wl.get_n1n2(NP1,NP2)
                
                    N1[abs(N1)>=Nphi/2]=I0[0]
                    N2[abs(N2)>=Nphi/2]=I0[1]
                    
                    VR = R[N1,N2].flatten()   
                    Outlist.append(1*VR)
                
    [VR0,VR1,VR2] = Outlist
        
    return VR0,VR1,VR2

def rgf_solve_steadystate(k,parameters,NP1,NP2,freqlist,mu,Nphi,evolution_file=None):
    """
    Solve driven weyl problem in one-particle sector using frequency domain 
    solver (with the recursive greens function approach)
    
    args:
        
        k           :   crystal momentum at which we work
        
        NP1,NP2     :   Photon space dimension used
        
        freqlist    :   frequency at which to compute the fourier transform 
                        elements of freqlist must be of the form (m,n) for 
                        integer m,n, and corresponds to physical frequency
                        m * omega1 + n * omega2 
                        
        mu          :   chemical potential to use
        
        Nphi        :   drive resolution used both for frequency space 
                        subroutine (to solve 0- and 2-particle sectors), and 
                        time-domain routine.
                        
    
    returns:
        
        r0,r1,r2    :   Fourier transform of steady state at frequencies given
                        in freqlist. Specifically rn[k] gives the fourier 
                        transform of the steadystate in the n-particle sector, 
                        at frequency m*omega1 + n*omega2, where 
                        (m,n) = freqlist[k]
                        

    """
    [omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,mu,Temp] = parameters
    
    global r1p
    global freq1_list,rho1
    assert (evolution_file is None) or type(evolution_file)==str,"save_evolution must be None or str"
    if type(evolution_file)==str:
        
        save_evolution = True
        evolution_file = evolution_file
    else:
        save_evolution = False
        evolution_file = ""
    
    if not save_evolution:    
        freq1_list = list(sort(list(set([f[0] for f in freqlist]))))
        
    else:
        freq1_list = list(wl.get_nvec(NP1))
       
        # find indices in freq1list which we want for output
        power_freqs = list(sort(list(set([f[0] for f in freqlist]))))
        
        ind = []
        for f in power_freqs:
            z = where(freq1_list==f)[0]
            ind.append(z)
            
        
            
    nfreqs = len(freqlist)
    nv1,nv2 = [list(x) for x in (wl.get_nvec(NP1),wl.get_nvec(NP2))]
    block_list = array([nv1.index(f) for f in freq1_list])
    ind = array(block_list)

    FR0,FR1,FR2 = get_rhoeq_vector(k,parameters,NP1,NP2,mu=mu,Nphi=Nphi)
    relaxation_vector  = (-1/tau * FR1)
    
    ### Construct Weyl Liouvillian
    L = wl.get_liouvillian(k,3,NP2)       

    hlist,J,Jp = rgf.get_matrix_lists(L,4*NP2)   
    
    h0 = hlist[1]
    rgf_eye = sp.eye(4*NP2,dtype=complex)
    
    def get_h(n):
        return h0 + omega1*1j*nv1[n]*rgf_eye
    
    S = rgf.rgf_solver([0]*NP1,J,Jp)
    S.get_h0 = get_h
    S_rgf = S 
    rho1_full = S(relaxation_vector,block_list,mode="l").reshape((len(freq1_list),NP2,2,2)) 
    
    if save_evolution:
        
        rho1 = rho1_full[ind,:]
    else:
        rho1 = rho1_full
        
    
    
    rho2= FR2.reshape((NP1,NP2))
    
    
    rho0 = FR0.reshape((NP1,NP2))
    
    
    nv = wl.get_nvec(NP2)
    nb0 = nv2.index(0)# argmax(nv==0)
    nb1 = nv2.index(1)#argmax(nv==1)
    
    r0p = zeros(nfreqs,dtype=complex)
    r1p = zeros((nfreqs,2,2),dtype=complex)
    r2p = zeros(nfreqs,dtype=complex)
    
    Out = ()
    nf=0
    for freq in freqlist:
        (m,n) = freq
        ind1 = freq1_list.index(m)
        ind2 = nv2.index(n)
        
        r0p[nf]        = rho0[nv1.index(m),nv2.index(n)]/(-1j*(m*omega1*tau+n*omega2*tau) +1)
        r1p[nf,:,:]    = rho1[ind1,ind2]        
        r2p[nf]        = rho2[nv1.index(m),nv2.index(n)]/(-1j*(m*omega1*tau+n*omega2*tau) +1)
    
        nf+=1

    if save_evolution:
        
        datadir = "../Frequency_domain_solutions/"
        filename = datadir + evolution_file
        
        savez(filename,k=k,parameters = parameters,freq_1 =nv1,freq_2=nv2,fourier_coefficients_1p = rho1_full,fourier_coefficients_0p=rho0,fourier_coefficients_2p=rho2)
    
    
    return r0p,r1p,r2p,rho1

def time_domain_solve_steadystate(k,parameters,freqlist,mu,Nphi,tmax,evolution_file=None):
    """
    Solve driven weyl problem in one-particle sector using time domain solver.
    
    args:
        
        k           :   crystal momentum at which we work
        
        freqlist    :   frequency at which to compute the fourier transform 
                        (in meV)
                        
        mu          :   chemical potential to use
        
        Nphi        :   drive resolution used both for frequency space 
                        subroutine (to solve 0- and 2-particle sectors), and 
                        time-domain routine.
                        
        tmax        :   maximum time over which the fourier transform is    
                        evaluated 
    
    returns:
        
        r0,r1,r2    :   Fourier transform of steady state at frequencies given
                        in freqlist. Specifically rn[k] gives the fourier 
                        transform of the steadystate in the n-particle sector, 
                        at frequency freqlist[k]     
        
    """
    omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp = parameters
    
    freqs = [omega1*m+omega2*n for (m,n) in freqlist]
    nfreqs = len(freqlist)
    global Solver,r2p,r1p,r0p
    global S_tds 
    Solver = tds.time_domain_solver(k,parameters,tmax,evolution_file=evolution_file)
    r1p_vec =  Solver.get_ft(freqlist)
    S_tds =Solver
    
    ### computing r_2p and r_0p
    FR0,FR1,FR2 = get_rhoeq_vector(k,parameters,Nphi,Nphi,mu=mu,Nphi=Nphi)
    rho2= FR2.reshape((Nphi,Nphi))
    rho0 = FR0.reshape((Nphi,Nphi))
    rho1 = FR1.reshape((Nphi,Nphi,2,2))
    tr_rho1 = rho1[:,:,0,0]+rho1[:,:,1,1]
    
    r0p = zeros(nfreqs,dtype=complex)
    r2p = zeros(nfreqs,dtype=complex)    
    R1p = zeros(nfreqs,dtype=complex)    
    r1p = zeros((nfreqs,2,2),dtype=complex)    

    nv1,nv2 = [list(x) for x in (wl.get_nvec(Nphi),wl.get_nvec(Nphi))]
    
    nf=0    
    
    for freq in freqlist:
        (m,n) = freq
        ind2 = nv2.index(n)
        
        r0p[nf]        = rho0[nv1.index(m),nv2.index(n)]/(-1j*(m*omega1*tau+n*omega2*tau) +1)
        r2p[nf]        = rho2[nv1.index(m),nv2.index(n)]/(-1j*(m*omega1*tau+n*omega2*tau) +1)
        R1p[nf]        = tr_rho1[nv1.index(m),nv2.index(n)]/(-1j*(m*omega1*tau+n*omega2*tau) +1)
        r1p[nf,:,:]    = 0.5*I2*R1p[nf] + SX * r1p_vec[nf,0] + SY * r1p_vec[nf,1] + SZ * r1p_vec[nf,2]
        nf+=1
        
    return r0p,r1p,r2p
        
def get_steady_state_components(k,parameters,freqlist,NP0=INITIAL_NP,convergence_treshold=1e-9,
                     return_convergence_parameters=False,tmax = None,evolution_file=None):
    """
    Get flattened frequency vectors corresponding to steady state at momentum k 
    for frequencies \{(f1,n)\},\{(f2,n)\}, \{fm,n\} where n\in Z, and 
    f1,f2,..fm = freq1list.
    
    returns rho0,rho1,rho2
    
    Here 
    
        rho0[m,n]   gives the (freq1_list[m],n)-frequency component of the  
                    steady state in the zero-particle sector (i.e. the 
                    (freq1_list,n)-component of the time-dependent probability 
                    of finding zero particles at momentum k)
    
        rho1[m,n,,:,:] gives the (freq1_list[m],n)-frequency component of the 
                    steady state in the one-particle sector,    
        
        rho2[m,n]   gives the (freq1_list[m],n)-frequency component of the 
                    steady state in the two-particle sector, (i.e. the 
                    (freq1_list,n)-component of the time-dependent probability 
                    of finding twp particles at momentum k)
        
        
    I.e. 
        
        rho{np}[m,n] = \rho^{(np)}[freqlist[m],n], 
        
    where 
    
        rho^{np}(t) = \sum_{n1,n2} \rho{np}[n1,n2]e^{-i(\omega_1 Nvec[n_1] 
                    +\omega_2*Nvec[n_2]) t}, with rho^np(t) 
    
    gives the steady state in the np-particle sector.
        
    The algoright increases NP until it sees convergence. Convergence is determined
    by comparing the difference of the output from the present and last iteration.
    When the 1-norm of the difference is less than convergence_treshold * norm(output), 
    convergence is reached
    """
    [omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,mu,Temp] = parameters
    B.tic(n=57)
    
    if tmax is None:
        tmax = 1000*2*pi/omega1
    NP1 = 1*NP0
    
    NP_converged = False
    nfreqs=len(freqlist)
    ### reference output from last iteration used to determine convergence (set to zero for the first iteration)
    ReferenceRho = zeros((nfreqs,2,2),dtype=complex)

    
    while True:

        NP2 = NP1 
        
        print(f"{80*'-'}")
        print(f"Solving for NP1,NP2 = {NP1},{NP2}")
        sys.stdout.flush()
        B.tic(n=166)
        

        ### Find frequency components of equilbrium state
                
        r0p,r1p,r2p,rho1= rgf_solve_steadystate(k,parameters,NP1,NP2,freqlist,mu,NPHI_RGF,evolution_file=evolution_file)       
            
        print(f"    done. Time spent: {B.toc(n=166,disp=0):.4}s")
        sys.stdout.flush()
        # Check for convergence
        nl2= list(wl.get_nvec(NP2))
        n0 = nl2.index(0)
        DRho = 1*r1p
        
        convergence_ratio  = sum(abs(DRho-ReferenceRho))/sum(abs(DRho))

        if convergence_ratio <convergence_treshold:
            print(f"Converged. Convergence ratio is {convergence_ratio:.4}")
            print("="*80)
            print(f"Steady state solver done. Total time spent to converge: {B.toc(n=57,disp=0):.4} s")
            print(f"{'='*80}")
            use_tds = False
            sys.stdout.flush()                                
            break
        
 
        else:
            
            ReferenceRho = 1*DRho
            
            NPnew=max(NP1+1,int(NP1*1.5))
            
            if NPnew%2 == 0:
                NPnew +=1 
                
            if NPnew>NP_MAX:
                print(f"{'-'*80}")
                print("Reached maximum bound for NP1. Using time domain solver")
                
                r0p,r1p,r2p = time_domain_solve_steadystate(k,parameters,freqlist,mu,NPHI_TDS,tmax,evolution_file=evolution_file)

                use_tds = True
                print("="*80)
                print(f"Steady state solver done. Total time spent to converge: {B.toc(n=57,disp=0):.4} s")
                print(f"{'='*80}")
                sys.stdout.flush()
                break
            

            else:
                NP1 = NPnew
                NP2 =NP1
            
                print(f"No convergence yet. Convergence ratio is {convergence_ratio}")
                
    return (r0p,r1p,r2p,use_tds)

def get_current_component(r0,r1,r2):
    """
    Get current and density from states where the density matrix is given by 
    r0,r1,r2 in the 0, 1, and 2-particle sectors, respectively.
    """
    r00=r0;r11=r1;r22=r2
    d1 = trace(r1)
    
    [sx,sy,sz,Dens_1] = [trace(q@r1) for q in [SX,SY,SZ,I2]]

    Density = (d1 + 2* r2)
  
    jx = vF*sx+V0[0]*Density
    jy = vF*sy+V0[1]*Density
    jz = vF*sz+V0[2]*Density
        
    return Density,array([jx,jy,jz])

def get_average_steadystate_energy(r0,r1,r2,freqlist,k):
        """ 
        Compute average energy in steady state
        
            \bar E_{ss,eq} \equiv 1/t0 \int_0^t0 <H(t) * \rho_{ss,eq}(t)>,
        
        for t0->inf
        """
        
        assert freqlist == [(0,0),(0,1),(1,0)],"wrong frequency components given"
        

        E0ss = 0
        E1ss = 0
        E2ss = 0
        
        [h00,h01,h10] = [wl.get_h_fourier_component(m,n,k) for (m,n) in [(0,0),(0,1),(1,0)]]
        
        
        
        E1ss += real(trace(h00@r1[0]))
        E1ss += 2*real(trace(h01.conj().T@r1[1]))
        E1ss += 2*real(trace(h10.conj().T@r1[2]))

        E2ss += real(trace(h00)*r2[0])
        E2ss += 2*real(trace(h01).conj().T*r2[1])
        E2ss += 2*real(trace(h10).conj().T*r2[2])

        Ess = E0ss+E1ss+E2ss 
        
        return Ess

def solve_weyl(k,parameters,NP0=INITIAL_NP,convergence_treshold=CONVERGENCE_TRESHOLD,
               tmax =None,evolution_file=None):
    """
    Get average power pumped into modes 1 and 2, from modes at crystal momentum k.


    Parameters
    ----------
    k : ndarray(3), float
        k-point to probe.
    parameters : ndarray(11), float
        Parameters of system to probe.
    NP0 : int, optional
        Number of photon states to include in frequency domain solver initially. 
        The default is INITIAL_NP.
    convergence_treshold : float, optional
        Treshold for determning that the frequency domain solver has converged. 
        If the relative change of the steady state rho after increasing photon 
        number by a factor 1.5 is smaller than convergence_treshold, the solver
        is determined to have converged. The default is CONVERGENCE_TRESHOLD.
    tmax : float, optional
        averaging time in case time-domain solver is used. 
        The default is TMAX_IN_MODE1_PERIODS.

    Returns
    -------
    density : float
        time-averaged density in steady state.
    P1 : float
        time-averaged energy transfer to (or from?) mode 1.
    P2 : float
        time-averaged energy transfer to (or from?) mode 2.
    Ess : float
        time-averaged energy in the steady state
    Eeq : float
        time_averaged energy in instantaneous equilibrium state.
    use_tds : bool
        Flags whether time-domain solver has been used. If True, time-domain
        solver has been used

    Details 
    -------
    The power is computed as

    P(k) = \lim_{t_0 \to \infty} \int_0^{t_0} <j_k (t)\cdot E_1(t)>.
    
    Where j_k(t) is the current from the modes at tme t. 
    
    It can be found as Re(<j_10(k)>\cdot(\omega_1 A_1,0,-i\omega_1 A_1 )),
    where j_mn(k) are the freuqency components of the current such that  
    
    j_k(t) = \sum_{mn} j_{mn}(k)e^{-i(\omega_1m + \omega_2 n )t}
    """    

    if tmax is None:
         tmax = TMAX_IN_MODE1_PERIODS*2*pi/omega1

    [omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,mu,Temp]  =  parameters
    global r1 
    r0,r1,r2,use_tds = get_steady_state_components(
            k,
            parameters,
            [(0,0),(0,1),(1,0)],
            NP0=NP0,
            convergence_treshold=convergence_treshold,
            tmax=tmax,
            evolution_file=evolution_file
            )

    derived_quantities = derive_quantities_from_steady_state_component(k,parameters,r0,r1,r2)
    
    density,P1,P2,Ess,Eeq,work,dissipation = derived_quantities
    
    if dissipation < 0:
        print("%"*80)
        print("WARNING: NEGATIVE DISSIPATION COMPUTED")
        print(f"   Work         :   {work:.4}")
        print(f"   Dissipation  :   {dissipation:.4}")
        print("%"*80)
    elif abs((work-dissipation)/ dissipation)>1e-2:
        print("%"*80)
        print("WARNING: WORK DONE BY DRIVING DOES NOT MATCH DISSIPATION")
        print(f"   Work         :   {work:.4}")
        print(f"   Dissipation  :   {dissipation:.4}")
        print("%"*80)
        sys.stdout.flush()
        
    return density,P1,P2,Ess,Eeq,work,dissipation,use_tds


def derive_quantities_from_steady_state_component(k,parameters,r0,r1,r2):
    """
    Derive quantities from the fourier components of the steady state at crystal
    momentum k, with parameters specified by parameters

    Parameters
    ----------
    k : ndarray(3),float
        k-point to be probed.
    parameters : ndarray(11), float
        parameters to be probed.
    r0 : ndarray(3), float
        Fourier coefficients (0,0),(0,1),(1,0) of steady-state in 0-particle sector
    r1 : ndarray(3,2,2), float
        Fourier coefficients (0,0),(0,1),(1,0) of steady-state in 1-particle sector.
    r2 : ndarray(3), float
        Fourier coefficients (0,0),(0,1),(1,0) of steady-state in 2-particle sector.

    Returns
    -------
    density : float
        Average density in steady state.
    mode1_power : float
        Average energy transfer rate to mode 1 (or the other way around?)
    mode2_power : float
        Average energy transfer rate to mode 2 (or the other way around?)
    Eav_steadystate : float
        Average energy in steady state
    Eav_eq : float
        Average energy in instantaneous equilibrium state
    work : float
        mode1_power+mode2_power. Average rate of energy transferred from the system to the modes (or the other way around?)
    dissipation : float
        Average rate of dissipation, 1/tau*(Eav_steadystate-Eav_eq)  .

    """
    [omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,mu,Temp]  =  parameters
      
    [dh00,dh01,dh10] = [wl.get_dhdt_fourier_component(m,n,k) for (m,n) in [(0,0),(0,1),(1,0)]]

    mode1_power_0    = 0
    mode1_power_1    = 2*real(trace(dh10.conj().T @ r1[2]))
    mode1_power_2    = 2*real(trace(dh10.conj())*r2[2])
    
    mode2_power_0    = 0
    mode2_power_1    = 2*real(trace(dh01.conj().T @ r1[1]))
    mode2_power_2    = 2*real(trace(dh01.conj()) * r2[1])
        
    mode1_power = mode1_power_0+mode1_power_1+mode1_power_2
    mode2_power = mode2_power_0+mode2_power_1+mode2_power_2
    
    Eav_steadystate = get_average_steadystate_energy(r0,r1,r2,[(0,0),(0,1),(1,0)],k)
    
    work = mode1_power+mode2_power    
    
    Eav_eq          = wl.get_average_equlibrium_energy(k,NPHI_TDS,mu)

    density = 2*r2[0]+trace(r1[0])   
    
    
    dissipation = 1/tau*(Eav_steadystate-Eav_eq)       
    
    
    return density,mode1_power,mode2_power,Eav_steadystate,Eav_eq,work,dissipation









    
    


    
def main_run(klist,parameterlist,Save=True,PreStr="",
             display_progress=10,
             savetime=10*3600,
             NP0=INITIAL_NP,
             convergence_treshold=CONVERGENCE_TRESHOLD,
             tmax=None,
             Nphi=NPHI_RGF,
             save_steadystate = SAVE_STEADYSTATE):
    """
    Main sweep.
    
    Compute pumping power into mode 1 and 2, along with average density, from modes at crystal momentum k. 
    
    P1_list: work done by system on mode 1 . (similar with P2_list)
    
    Output is just used for testing purposes -- saves autmoatically
    """
    global use_tds 
    Nk = shape(klist)[0]
    assert len(parameterlist)==Nk,"Parameterlist must be of same length as klist"
            
    FileName        = PreStr+f"_{Nk}_"+B.ID_gen()
    
    DataPath        = "../Data/"+FileName
    
    P1_list         = zeros(Nk)
    P2_list         = zeros(Nk)
    Eeq_list        = zeros(Nk)
    Ess_list        = zeros(Nk)
    density_list    = zeros(Nk)
    dissipation_list= zeros(Nk)
    work_list       = zeros(Nk)
    convlist        = zeros((Nk,2),dtype=int)
    crlist          = zeros((Nk))
    use_tds_list    = zeros((Nk),dtype=bool)
    B.tic(n=1)
    B.tic(n=2)
    B.tic(n=3)
    
    for n in range(0,Nk):
        if save_steadystate:
            evolution_file = FileName + f"_{n}"
        else:
            evolution_file = None 
            
        if n % display_progress == 0 and n>0:
            print("="*80)
            print(f"At step {n}/{Nk}. Time spent: {B.toc(n=2):>6.3} s")
            print("="*80)
            sys.stdout.flush()
            
        k = klist[n]
        parameters=parameterlist[n]
        
        omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp = parameters
        
        if tmax==None:
            tmax_input = TMAX_IN_MODE1_PERIODS*2*pi/omega1 
        else:
            tmax_input = tmax
            
        density,P1,P2,Eav_ss,Eav_eq,work,dissipation,use_tds = solve_weyl(k,
                                                                          parameters,
                                                                          NP0=NP0,
                                                                          convergence_treshold=convergence_treshold,
                                                                          tmax = tmax_input,
                                                                          evolution_file=evolution_file)
        gc.collect()

        
        
        P1_list[n]=-P1
        P2_list[n]=-P2
        Eeq_list[n] = Eav_eq
        Ess_list[n] = Eav_ss
        use_tds_list[n] = use_tds
        density_list[n]=real(density)
        dissipation_list[n] = dissipation
        work_list[n] = work
   

        if B.toc(n=1,disp=False)>savetime and Save:

            savez(DataPath,
                  parameterlist=parameterlist[:n+1],
                  klist=klist[:n+1],
                  P1list=P1list[:n+1],
                  P2list=P2list[:n+1],
                  Dissipation=Dissipation[:n+1],
                  density_list=density_list[:n+1],
                  use_tds_list=use_tds_list[:n+1],
                  t_res = tds.T_RES)
            B.tic(n=1)
    
    dissipation_list = 1/tau * (Ess_list-Eeq_list)
    
    if Save:
        savez(DataPath,
              parameterlist=parameterlist,
              klist=klist,
              P1_list=P1_list,
              P2_list=P2_list,
              dissipation_list=dissipation_list,
              density_list=density_list,
              Eeq_list=Eeq_list,
              Ess_list=Ess_list,
              use_tds_list=use_tds_list,
              t_res = tds.T_RES)
        
    
        
    print("="*80)
    print(f"Done with whole sweep. Total time spent on all {Nk} runs: {B.toc(n=3,disp=0):.5} seconds")
    print("="*80)
    sys.stdout.flush()
    
    return P1_list,P2_list,density_list,Eeq_list,Ess_list,work_list,dissipation_list




    


    

    
if __name__=="__main__":
    try:
        from matplotlib.pyplot import *
    except ModuleNotFoundError:
        pass
    
    # =========================================================================
    # Simulation parameters
    # =========================================================================
    
    
    fds_dir = "../Frequency_domain_solutions/"
    
    # filename_fds = "_1_210729_1114-06.037_0.npz"
    # fds_data = load(fds_dir+filename_fds)


    
    omega2 = 20*THz
    # omega1 = 0.61803398875*omega2
    omega1 = 1.500015* omega2 
    tau    = 10*picosecond
    vF     = 1e6*meter/second
    
    EF1 = 0.6*1.25*1.2e6*Volt/meter
    EF2 = 0.6*1.5*2e6*Volt/meter
    
    T1 = 2*pi/omega1
    
    Mu =115
    mu = Mu
    Temp  = 20*Kelvin;
    V0 = array([0,0,0.8*vF])
    [V0x,V0y,V0z] = V0
    parameters = 1*array([omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp])
    # set_parameters(parameters[0])
    # k= array([[0.0024023 , 0.02387947, 0.068    ]])
    
    # integration_time = 20
    # TMAX_IN_MODE1_PERIODS  = 200
    
    omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp = parameters
    
    # tau = 1* picosecond
    klist= array([[  0.02631654,  0.03833653, -0.004  ]])
    parameterlist = 1*array([[omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp]])
    parameters = parameterlist[0]
    k = klist[0]
    parameterlist = 1*array([parameters])
    klist= array([k])


    P1,P2,density,Eeq,Ess,work,dissipation=main_run(
            klist,parameterlist,
            Save=False)
    
    

        #%%
        
    if use_tds:
            
        # t0 = array([0])
        # a = [(0,0),(1,2),(3,2),(4,5)]
        # A=S.get_phase_mat()
        B = S_tds.get_fourier_component(0,1)
        # phi1= mod(S_tds.sampling_phases[:,:,0].flatten(),2*pi)
        # phi2= mod(S_tds.sampling_phases[:,:,1].flatten(),2*pi)
        
        from matplotlib.pyplot import *
        
        X,Y = meshgrid(S_tds.bin_edges,S_tds.bin_edges)
        figure(1)
        pcolormesh(X,Y,S_tds.n_array.T)
        ylim((0,amax(S_tds.n_array)))
        # plot(phi1,phi2,'.w',markersize = 0.2)
        title("Number of data points")
        xlabel("$\phi_1$")
        ylabel("$\phi_2$")
        xlim(0,2*pi)
        ylim(0,2*pi)
        colorbar()
        figure(2)
        pcolormesh(X,Y,S_tds.phase_mat[:,:,2])
        # plot(phi1,phi2,'.w',markersize = 0.1)    
        title("Accumulated data")
        xlabel("$\phi_1$")
        ylabel("$\phi_2$")
        colorbar()   
        figure(3)
        pcolormesh(X,Y,S_tds.phase_mat[:,:,2]/S_tds.n_array)
        # plot(phi1,phi2,'.w',markersize = 0.4)
        title("Acc. Data/Number of data points")
        xlabel("$\phi_1$")
        ylabel("$\phi_2$")
        colorbar()
        
        figure(4)
        pcolormesh(X,Y,S_tds.rho_mat[:,:,0]);colorbar()
        # plot(phi1,phi2,'.w',markersize = 0.4)
        title("Interpolated")
        xlabel("$\phi_1$")
        ylabel("$\phi_2$")
    
        
    # # =========================================================================
    # # Simulation parameters
    # # =========================================================================
    
#     parameters = array([2.98881237e+00, 4.83600000e+00, 3.30851944e+03, 2.41800000e+03,
#         0.00000000e+00, 0.00000000e+00, 1.93440000e+03, 9.00000000e-02,
#         1.80000000e-01, 1.15000000e+02, 1.72400000e+00])

# # array([2.98881237e+00, 4.83600000e+00, 4.13564930e-02, 2.41800000e+02,
# #         0.00000000e+00, 0.00000000e+00, 1.93440000e+02, 9.00000000e-02,
# #         1.80000000e-01, 1.15000000e+01, 1.72400000e-01])
#     [omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,mu,Temp]  =  parameters
    
#     klist= array([[ 0.1,  0.        , 0       ]])
#     parameterlist = 1*array([parameters])



#     P1,P2,density,Eeq,Ess,work,dissipation=main_run(
#             klist,parameterlist,
#             display_progress=1000,
#             Save=False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    