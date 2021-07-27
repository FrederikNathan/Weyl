#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 09:10:23 2020

@author: frederik


We work in units where hbar = e = 1.
Energies and times are in units of meV.
Length is in unit of Å.


v2: clever cropping of photon space. Dynamical change of photon space, to speed up (i.e. only use large photon space when absolutely necessary. )
v3: Using recursive greens function to solve
v3.1: truncating to the one-particle sector.
v4: separated Weyl liouvillian out
v5: efficient hlist to save memory
v6: using time-domain solver
v7: fixed negative dissipation issue
v8: direct computation of <d\phi_2 H> and <d\phi_1 H>
"""

NP_MAX = 3 # Maximum number of photons before using time domain solver
INITIAL_NP = 30
NPHI_RGF=200    # NPhi used to calculate rho_steady_state with rgf metho
NPHI_TDS=200   # Nphi used to calculate steadystate with tds method
CONVERGENCE_TRESHOLD = 1e-6
TMAX_IN_MODE1_PERIODS = 2000



import os 



import sys 
from scipy.linalg import * 
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import * 
from numpy.fft import *
import vectorization as ve
import numpy.random as npr
import scipy.optimize as optimize
import basic as B
from Units import *
import weyl_liouvillian_v1 as wl
import recursive_greens_function_v1 as rgf
import gc

import time_domain_solver_v4 as tds



def get_rhoeq_vector(k,NP1,NP2,mu=0,Nphi=NPHI_RGF):
    """
    calculate freuqency-space vector corresponding to \rho_eq(k+A(phi1,phi2)), 
    for phi1,phi2 = 0,1,..Nphi * 2*pi/Nphi
    
    returns 3 flattned arrays of size Nphi**2, 4*Nphi**2, Nphi**2
    
    Here the nth array gives rhoeq in the k-particle sector 
    """
#    global VR0,VR1,VR2,R0,R1,R2
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

def rgf_solve_steadystate(k,NP1,NP2,freqlist,mu,Nphi):
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
    freq1_list = list(sort(list(set([f[0] for f in freqlist]))))
    
    nfreqs = len(freqlist)
    nv1,nv2 = [list(x) for x in (wl.get_nvec(NP1),wl.get_nvec(NP2))]
    block_list = array([nv1.index(f) for f in freq1_list])
    ind = array(block_list)

    FR0,FR1,FR2 = get_rhoeq_vector(k,NP1,NP2,mu=mu,Nphi=Nphi)
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

    rho1 = S(relaxation_vector,block_list,mode="l").reshape((len(freq1_list),NP2,2,2)) 
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

    
    
    return r0p,r1p,r2p,rho1

def time_domain_solve_steadystate(k,freqlist,mu,Nphi,tmax):
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
    
    freqs = [omega1*m+omega2*n for (m,n) in freqlist]
    nfreqs = len(freqlist)
    global Solver,r0p,r1p,r2p
    Solver = tds.time_domain_solver(k,Nphi=Nphi)

    r1p,t_final =  Solver.get_ft(freqs,tmax)
    
    ### computing r_2p and r_0p

    FR0,FR1,FR2 = get_rhoeq_vector(k,Nphi,Nphi,mu=mu,Nphi=Nphi)
    rho2= FR2.reshape((Nphi,Nphi))
    rho0 = FR0.reshape((Nphi,Nphi))
    r0p = zeros(nfreqs,dtype=complex)
    r2p = zeros(nfreqs,dtype=complex)    
    R1p = zeros(nfreqs,dtype=complex)    

    nv1,nv2 = [list(x) for x in (wl.get_nvec(Nphi),wl.get_nvec(Nphi))]
    
    nf=0    
    
    for freq in freqlist:
        (m,n) = freq
        ind2 = nv2.index(n)
        
        r0p[nf]        = rho0[nv1.index(m),nv2.index(n)]/(-1j*(m*omega1*tau+n*omega2*tau) +1)
        r2p[nf]        = rho2[nv1.index(m),nv2.index(n)]/(-1j*(m*omega1*tau+n*omega2*tau) +1)
        nf+=1
        
    
    return r0p,r1p,r2p
        
def get_steady_state_components(k,freqlist,mu,NP0=INITIAL_NP,convergence_treshold=1e-9,
                     return_convergence_parameters=False,tmax = None):
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

    B.tic(n=57)
    
    if tmax is None:
        tmax = 1000*2*pi/omega1
    NP1 = 1*NP0
    
    NP_converged = False
    nfreqs=len(freqlist)
    ### reference output from last iteration used to determine convergence (set to zero for the first iteration)
    ReferenceRho = zeros((nfreqs,2,2),dtype=complex)
#    global r0p,r1p,r2p,rho1

    while True:

        NP2 = NP1 
        
        print(f"{80*'-'}")
        print(f"Solving for NP1,NP2 = {NP1},{NP2}")
        sys.stdout.flush()
        B.tic(n=166)
        
        ### Find indices of blocks that we are interested in 

        ### Find frequency components of equilbrium state
                
        r0p,r1p,r2p,rho1= rgf_solve_steadystate(k,NP1,NP2,freqlist,mu,NPHI_RGF)       
            
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
                
                r0p,r1p,r2p = time_domain_solve_steadystate(k,freqlist,mu,NPHI_TDS,tmax)

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
#    global d1,Density,sx,sy,sz,jx,jy,jz
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
#        global E1ss,E2ss,h00,h01,h10
        

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
def get_density(r0,r1,r2):
    return 2*r2+trace(r1)    

def solve_weyl(k,mu,NP0=INITIAL_NP,convergence_treshold=CONVERGENCE_TRESHOLD,tmax =None):
    """
    Get average power pumped into modes 1 and 2, from modes at crystal momentum k.

    P(k) = \lim_{t_0 \to \infty} \int_0^{t_0} <j_k (t)\cdot E_1(t)>.
    
    Where j_k(t) is the current from the modes at tme t. 
    
    It can be found as Re(<j_10(k)>\cdot(\omega_1 A_1,0,-i\omega_1 A_1 )),
    where j_mn(k) are the freuqency components of the current such that  
    
    j_k(t) = \sum_{mn} j_{mn}(k)e^{-i(\omega_1m + \omega_2 n )t}
    """    
    global r0,r1,r2
    if tmax is None:
         tmax = 1000*2*pi/omega1

#    global Eav_eq,mode1_power_0,mode1_power_1,mode1_power_2,r0,r1,r2,Eav_steadystate,dh00,dh10,dh01
    r0,r1,r2,use_tds = get_steady_state_components(
            k,
            [(0,0),(0,1),(1,0)],
            mu,
            NP0=NP0,
            convergence_treshold=convergence_treshold,
            tmax=tmax
            )

       
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

    density = get_density(r0[0],r1[0],r2[0])
    
    
    dissipation = 1/tau*(Eav_steadystate-Eav_eq)
    
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
        
    
    
    return density,mode1_power,mode2_power,Eav_steadystate,Eav_eq,use_tds









    
def set_parameters(parameters):
    """
    Set parameters in modules used to solve weyl"""
    
    wl.set_parameters(parameters)
    tds.wl.set_parameters(parameters)
    tds.set_parameters(parameters)

    global omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp

    [omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp]=parameters
    assert omega2>=omega1,"omega2 must be larger than omega1 (due to efficiency considerations)"
    
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
    


    
def main_run(klist,parameterlist,Save=True,PreStr="",
             display_progress=10,
             savetime=10*3600,
             NP0=INITIAL_NP,
             convergence_treshold=CONVERGENCE_TRESHOLD,
             tmax=None,
             Nphi=NPHI_RGF):
    """
    Main sweep.
    
    Compute pumping power into mode 1 and 2, along with average density, from modes at crystal momentum k. 
    """
    
    Nk = shape(klist)[0]
    assert len(parameterlist)==Nk,"Parameterlist must be of same length as klist"
            
    FileName        = PreStr+f"_{Nk}_"+B.ID_gen()
    
    DataPath        = "../Data/"+FileName
    
    P1_list         = zeros(Nk)
    P2_list         = zeros(Nk)
    Eeq_list        = zeros(Nk)
    Ess_list        = zeros(Nk)
    density_list    = zeros(Nk)
    convlist        = zeros((Nk,2),dtype=int)
    crlist          = zeros((Nk))
    use_tds_list    = zeros((Nk),dtype=bool)
    B.tic(n=1)
    B.tic(n=2)
    B.tic(n=3)
    
    for n in range(0,Nk):
        if n % display_progress == 0 and n>0:
            print("="*80)
            print(f"At step {n}/{Nk}. Time spent: {B.toc(n=2):>6.3} s")
            print("="*80)
            sys.stdout.flush()
            
        k = klist[n]
        parameters=parameterlist[n]
        
        set_parameters(parameters)
        
        if tmax==None:
            tmax_input = TMAX_IN_MODE1_PERIODS*2*pi/omega1 
        else:
            tmax_input = tmax
            
        density,P1,P2,Eav_ss,Eav_eq,use_tds = solve_weyl(k,mu=Mu,NP0=NP0,convergence_treshold=convergence_treshold,tmax = tmax_input)
        gc.collect()

        # P1_list: work done by system on mode 1 . (similar with P2_list)
        
        P1_list[n]=-P1
        P2_list[n]=-P2
        Eeq_list[n] = Eav_eq
        Ess_list[n] = Eav_ss
        use_tds_list[n] = use_tds
        density_list[n]=real(density)
        
   

        if B.toc(n=1,disp=False)>savetime and Save:
            dissipation_list = -(P1list+P2list)

            savez(DataPath,
                  parameterlist=parameterlist[:n+1],
                  klist=klist[:n+1],
                  P1list=P1list[:n+1],
                  P2list=P2list[:n+1],
                  Dissipation=Dissipation[:n+1],
                  density_list=density_list[:n+1],
                  use_tds_list=use_tds_list[:n+1])
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
              use_tds_list=use_tds_list)
        
    
        
    print("="*80)
    print(f"Done with whole sweep. Total time spent on all {Nk} runs: {B.toc(n=3,disp=0):.5} seconds")
    print("="*80)
    sys.stdout.flush()
    
    return P1_list,P2_list,density_list,dissipation_list,Eeq_list,Ess_list




    


    

    
if __name__=="__main__":
    try:
        from matplotlib.pyplot import *
    except ModuleNotFoundError:
        pass
    
    # =========================================================================
    # Simulation parameters
    # =========================================================================
    

    omega2 = 20*THz
    omega1 = 0.61803398875*omega2
    tau    = 10*picosecond
    vF     = 1e5*meter/second
    
    EF2 = 0.6*1.5*2e6*Volt/meter
    EF1 = 0.6*1.25*1.2e6*Volt/meter*1e-10
    
    T1 = 2*pi/omega1
    
    Mu =115*0.1
    Temp  = 20*Kelvin*0.1;
    
    V0 = array([0,0,0.8*vF])
    [V0x,V0y,V0z] = V0
    
    ky=0;kx=0
    kzlist = linspace(-00,0,20)
#    kzlist = [-0.4427672955974843]

    klist = array([[kx,ky,kz] for kz in kzlist])

    klist= array([[ 0,  0.        , -0.2      ]])
    parameters = 1*array([[omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp]]*len(klist))
    set_parameters(parameters[0])

    # =========================================================================
    # Solve
    # =========================================================================



#    tds.DT = 0.005
    P1_list,P2_list,dissipation_list,density_list,Eeq_list,Ess_list =main_run(
            klist,parameters,
            Save=False)
    
    
    print(f"\n Dissipation: {dissipation_list[0]:.4}\n")
    

    
        













    