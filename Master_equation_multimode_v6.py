#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 09:10:23 2020

@author: frederik


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


v2: clever cropping of photon space. Dynamical change of photon space, to speed up (i.e. only use large photon space when absolutely necessary. )
v3: Using recursive greens function to solve
v3.1: truncating to the one-particle sector.
v4: separated Weyl liouvillian out
v5: efficient hlist to save memory
"""

NP_MAX = 300 # Maximum number of photons before using time domain solver
INITIAL_NP = 50
NUM_THREADS = 1
CONVERGENCE_TRESHOLD = 1e-10
import os 


os.environ["OMP_NUM_THREADS"]        = str(NUM_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(NUM_THREADS)
os.environ["NUMEXPR_NUM_THREADS"]    = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"]        = str(NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"]   = str(NUM_THREADS)


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
#import time_domain_solver_v1 as tds
#import c_recursive_greens_function_v2 as c_rgf
import time_domain_solver_v1 as tds
def get_rhoeq_vector(k,NP1,NP2,mu=0,Nphi=200):
    """
    calculate freuqency-space vector corresponding to \rho_eq(k+A(t))
    """

    R0,R1,R2 = wl.get_fourier_rhoeq(k,mu=mu,Nphi=Nphi)
    
    if NP1<=Nphi/2:
        N1,N2 = wl.get_n1n2(NP1,NP2)
        VR0 = R0[N1,N2].flatten()
        VR1 = R1[N1,N2,:,:].flatten()
        VR2 = R2[N1,N2].flatten()
    
    else:
                
        Outlist = []
        for R in R0,R1,R2:
            AX =tuple(x for x in range(2,ndim(R)))
                            
            A0 = amin(sum(abs(R),axis=AX))
            
            assert A0<1e-10,"Fourier transform not converged"
                
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

def get_steady_state(k,freq1_list,mode ="full",mu=0,Nphi=200,NP0=INITIAL_NP,convergence_treshold=1e-9,
                     return_convergence_parameters=False,plot_progress=False,tmax = None):
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
    
    B.tic()    
    B.tic(n=57)
    
    if tmax is None:
        tmax = 1000*2*pi/omega1
    NP1 = 1*NP0
    
    
    
    NP_converged = False

    ### reference output from last iteration used to determine convergence (set to zero for the first iteration)
    ReferenceRho = zeros((2,12),dtype=complex)
    global L,NP2,rho1

    global S,steady_state,block_list,rho1,r00,r01,r10,FR0,FR1,FR2,ss2
    while True:

        NP2 = NP1 
        
            
        print("")
        print(f"{80*'-'}")
        print(f"Solving for NP1,NP2 = {NP1},{NP2}")
        
        ### Find indices of blocks that we are interested in 
#        global nv1
        nv1= list(wl.get_nvec(NP1))
        nv2= list(wl.get_nvec(NP2))
        block_list = [nv1.index(f) for f in freq1_list]  
        
        
        ### Find frequency components of equilbrium state
        FourierConverged=False
        Nphi_new = 1*  Nphi
        
        ### Iterate until convergence is seen
        B.tic(n=6)
        while not FourierConverged: 
            
            try:
                FR0,FR1,FR2 = get_rhoeq_vector(k,NP1,NP2,mu=mu,Nphi=Nphi_new)

                relaxation_vector  = (-1/tau * FR1)
                
                FourierConverged = True
            
            except AssertionError:
                
                Nphi_new = int(1.5*Nphi_new)
                
                print(f"Increased Nphi to {Nphi_new}")
                
        ### Solve
        

        ### Construct Weyl Liouvillian
        L = wl.get_liouvillian(k,3,NP2)
 
#        L0 = wl.get_liouvillian(k,NP1,NP2)
        
#        ss2 = spla.spsolve(L0,relaxation_vector)
        B.tic(n=7);
        hlist,J,Jp = rgf.get_matrix_lists(L,4*NP2)   
        
        h0 = hlist[1]
        rgf_eye = sp.eye(4*NP2,dtype=complex)
        
        def get_h(n):
            return h0 + omega1*1j*nv1[n]*rgf_eye
        
        
        S = rgf.rgf_solver([0]*NP1,J,Jp)
        S.get_h0 = get_h
        
        steady_state = S(relaxation_vector,block_list,mode="l") 
        rho1= steady_state.reshape((len(freq1_list),NP2,2,2))

        B.toc(n=7)


        n0 = nv2.index(0)
        DRho = 1*steady_state[:,4*(n0-1):4*(n0+2)]
        
        convergence_ratio  = norm(DRho-ReferenceRho,ord=1)/norm(DRho,ord=1)
        if plot_progress:
            
            if not "plt" in dir():
                import matplotlib.pyplot as plt
            plt.figure(1)
            plt.plot(real(DRho.flatten()));plt.plot(real(ReferenceRho.flatten()))
            plt.title("Real")
            plt.legend(("new","old"))
            plt.show()
            plt.figure(2)
            plt.plot(imag(DRho.flatten()));plt.plot(imag(ReferenceRho.flatten()))
            plt.title("Imag");plt.legend(("new","old"))
            plt.show()


        if convergence_ratio <convergence_treshold:
            print("")
            print(f"{'-'*80}")
            print(f"Converged with NP1,NP2 = {NP1},{NP2}. Convergence ratio is {convergence_ratio}")
            print(f"    total time spent to converge: {B.toc(n=57):.4} s")
            
            Convergence = True
            
            ind = array(block_list)
            rho2= FR2.reshape((NP1,NP2))[ind,:]
            rho0 = FR0.reshape((NP1,NP2))[ind,:]

            rho1= steady_state.reshape((len(freq1_list),NP2,2,2))
            nv = wl.get_nvec(NP2)
            nb0 = argmax(nv==0)
            nb1 = argmax(nv==1)
        
            r0_01,r1_01,r2_01 = rho0[0,nb1],rho1[0,nb1,:,:],rho2[0,nb1]
            r0_10,r1_10,r2_10 = rho0[1,nb0],rho1[1,nb0,:,:],rho2[1,nb0]
            r0_00,r1_00,r2_00 = rho0[0,nb0],rho1[0,nb0,:,:],rho2[0,nb0]   
            
            r01 = r0_01,r1_01,r2_01
            r10 = r0_10,r1_10,r2_10
            r00 = r0_00,r1_00,r2_00
            
            break
            
 
        else:
            
            ReferenceRho = 1*DRho
            
            NPnew=max(NP1+1,int(NP1*1.5))
            
            if NPnew%2 == 0:
                NPnew +=1 
                
            if NPnew>NP_MAX:
                print(f"{'-'*80}")
                print("Reached maximum bound for NP1. Using time domain solver")
#                print(f"    total time spent to reach maximum: {B.toc(n=6,disp=False):.4} s")
                global Solver
                
                Solver = tds.time_domain_solver(k,dtmax=0.01*pi/omega2,Nphi=200)
                freqs = [omega1,omega2,0]
                global Out,Outlist
                Out,Outlist= Solver.get_ft(freqs,tmax)
                ind = array(block_list)
#                assert 0
                (r1_10,r1_01,r1_00)=Out
                rho2= FR2.reshape((NP1,NP2))[ind,:]
                rho0 = FR0.reshape((NP1,NP2))[ind,:]
    
#                rho1= steady_state.reshape((len(freq1_list),NP2,2,2))
                nv = wl.get_nvec(NP2)
                nb0 = argmax(nv==0)
                nb1 = argmax(nv==1)
            
                r0_01,r2_01 = rho0[0,nb1],rho2[0,nb1]
                r0_10,r2_10 = rho0[1,nb0],rho2[1,nb0]
                r0_00,r2_00 = rho0[0,nb0],rho2[0,nb0]   
                r01 = r0_01,r1_01,r2_01
                
                r10 = r0_10,r1_10,r2_10
                r00 = r0_00,r1_00,r2_00
                
                global alpha,beta,gamma
                alpha=1*r00;
                beta=1*r01,
                gamma=1*r10
                break 
                
            else:
                NP1 = NPnew
                NP2 =NP1# 1+int(NP1*omega1/omega2)
            
                print(f"No convergence yet. Convergence ratio is {convergence_ratio}")
                   
    return (r00,r01,r10)

    print("="*80)
    print("")
    

    ### Compute rho0 and rho2 (this is trivial since the Liouvillian does not act on these sectors)


    
    


def get_current_component(r0,r1,r2):
    """
    Get current and density from states where the density matrix is given by 
    r0,r1,r2 in the 0, 1, and 2-particle sectors, respectively.
    """
    global d1,Density,sx,sy,sz,jx,jy,jz
    r00=r0;r11=r1;r22=r2
    d1 = trace(r1)
    
    [sx,sy,sz,Dens_1] = [trace(q@r1) for q in [SX,SY,SZ,I2]]

    Density = (d1 + 2* r2)
  
    jx = vF*sx+V0[0]*Density
    jy = vF*sy+V0[1]*Density
    jz = vF*sz+V0[2]*Density
        
    return Density,array([jx,jy,jz])

def solve_weyl(k,mode="full",mu=0,Nphi=200,NP0=INITIAL_NP,convergence_treshold=5e-5,tmax =None):
    """
    Get average power pumped into modes 1 and 2, from modes at crystal momentum k.

    P(k) = \lim_{t_0 \to \infty} \int_0^{t_0} <j_k (t)\cdot E_1(t)>.
    
    Where j_k(t) is the current from the modes at tme t. 
    
    It can be found as Re(<j_10(k)>\cdot(\omega_1 A_1,0,-i\omega_1 A_1 )),
    where j_mn(k) are the freuqency components of the current such that  
    
    j_k(t) = \sum_{mn} j_{mn}(k)e^{-i(\omega_1m + \omega_2 n )t}
    """    
    if tmax is None:
         tmax = 1000*2*pi/omega1
    
    global jz_1,jy_1,jx_1,jx_2,jy_2,jz_2,J,power_1,power_2
    if mode=="full":
    
        (r_00,r_01,r_10) = get_steady_state(
                k,[0,1],mu=mu,Nphi=Nphi,
                NP0=NP0,
                convergence_treshold=convergence_treshold,
                plot_progress=0,
                tmax=tmax
                )
    
        (r0_10,r1_10,r2_10)=r_10
        (r0_00,r1_00,r2_00)=r_00
        (r0_01,r1_01,r2_01)=r_01

        density_1,[jx_1,jy_1,jz_1] = get_current_component(r0_10,r1_10,r2_10)    
        density_2,[jx_2,jy_2,jz_2] = get_current_component(r0_01,r1_01,r2_01)
        density,J =get_current_component(r0_00,r1_00,r2_00)
        
    elif mode =="approx":
        raise NotImplementedError
        
    power_1 = A1*omega1*real(jx_1-1j*jz_1)
    power_2 = A2*omega2*real(jy_2-1j*jz_2)
    
    
    
    return density,power_1,power_2,J


    




    
def set_parameters(parameters):
    wl.set_parameters(parameters)
    tds.wl.set_parameters(parameters)
    tds.set_parameters(parameters)

    global omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp

    [omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp]=parameters
    assert omega2>=omega1,"omega2 must be larger than omega1 (due to efficiency)"
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
    


    
def main_run(klist,parameterlist,Plot=False,Save=True,SaveFig=True,PreStr="",
             display_progress=10,
             savetime=10*3600,
             NP0=INITIAL_NP,
             convergence_treshold=CONVERGENCE_TRESHOLD,
             tmax=None):
    """
    Main sweep.
    
    Compute pumping power into mode 1 and 2, along with average density, from modes at crystal momentum k. 
    """


        

    Nk = shape(klist)[0]
    if not len(parameterlist)==Nk:
        raise ValueError("Parameterlist must be of same length as klist")
            
    FileName = PreStr+f"_{Nk}_"+B.ID_gen()
    
    DataPath = "../Data/"+FileName
    
    P1_list = zeros(Nk)
    P2_list = zeros(Nk)
    E1_list = zeros(Nk)
    E2_list = zeros(Nk)
    density_list = zeros(Nk)
    convlist = zeros((Nk,2),dtype=int)
    crlist = zeros((Nk))
    B.tic(n=1)
    B.tic(n=2)
    B.tic(n=3)
    for n in range(0,Nk):
        if n % display_progress == 0 and n>0:
            print("="*80)
            print(f"At step {n}/{Nk}. Time spent: {B.toc(n=2):>6.3} s")
            print("="*80)
        k = klist[n]
        parameters=parameterlist[n]
        
        set_parameters(parameters)
        
        if tmax==None:
            tmax_input = 1000*2*pi/omega1 
        else:
            tmax_input = tmax
            
        density,P1,P2,J = solve_weyl(k,mu=Mu,NP0=NP0,convergence_treshold=convergence_treshold,tmax = tmax_input)
        gc.collect()
        
        P1_list[n]=-P1
        P2_list[n]=-P2

        density_list[n]=real(density)
        
   

        if B.toc(n=1,disp=False)>savetime and Save:
            dissipation_list = -(P1list+P2list)

            savez(DataPath,parameterlist=parameterlist[:n+1],klist=klist[:n+1],P1list=P1list[:n+1],P2list=P2list[:n+1],Dissipation=Dissipation[:n+1],density_list=density_list[:n+1])
            B.tic(n=1)
    
    dissipation_list = -(P1_list+P2_list)
    
    if Save:
        savez(DataPath,parameterlist=parameterlist,klist=klist,P1_list=P1_list,P2_list=P2_list,dissipation_list=dissipation_list,density_list=density_list)
        
        
    print("="*80)
    print(f"Done with whole sweep. Total time spent on all {Nk} runs: {B.toc(n=3,disp=0):.5} seconds")
    print("="*80)
    return P1_list,P2_list,dissipation_list,density_list,convlist,crlist

if __name__=="__main__":
    try:
        
        from matplotlib.pyplot import *
    except ModuleNotFoundError:
        pass
    
    # =============================================================================
    # Simulation parameters
    # =============================================================================
    

    omega2 = 20*THz
    omega1 = 0.61803398875*omega2
    tau    = 0.01*picosecond
    vF     = 1e6*meter/second
    
    EF2 = 0.6*1.5*2e6*Volt/meter
    EF1 = 0.6*1.25*1.2e6*Volt/meter
    
    T1 = 2*pi/omega1
    
    Mu =76.7*meV
#    Mu = 300*meV
    Temp  = 0.000001*Mu;
    
    V0 = array([0,0,0.8*vF])*1
    [V0x,V0y,V0z] = V0
    
#    NPmax = 511
    # =============================================================================
    # K grid
    # =============================================================================
    kzlist = linspace(-0.2,0.2,16)#,0.2,20)*(Å**-1)
    kx=0.0
    ky=0
    dk = kzlist[1]-kzlist[0]
#    kzlist  = array([0.1,0.2])
#    kzlist=kzlist[36:]$
    kzlist = [-0.02]
#    kzlist = array([0.001])
    klist = array([[kx,ky,kz] for kz in kzlist])
#    klist = klist[:2]
#    S = shape()
    parameters = 1*array([[omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp]]*len(klist))
#,V0list,ROlist
    V0list =[]
    ROlist = []   
    Rlist = []
    RFlist = []     
    P1_list,P2_list,dissipation_list,density_list,convlist,crlist=main_run(
            klist,parameters,
            Plot=1,
            Save=False,
            display_progress=1,
            NP0=INITIAL_NP,
            convergence_treshold=CONVERGENCE_TRESHOLD,
            tmax=1000*T1)
    
    [E1list,E2list] = wl.get_E1(klist).T
    
    figure(1)
    title("Energy transfer")
    plot(kzlist*Å,P1_list/P0,'b')
    plot(kzlist*Å,P2_list/P0,'r')
    plot(kzlist*Å,-(P1_list+P2_list)/P0,'k')
    ylabel("Pumping power [$\omega_1\omega_2/2\pi$]")
    xlabel("$k_z$ [${\\rm Å}^{-1}$]")
#    title(")
    figure(2)
    title("Average density vs  z-crystal momentum (kx,ky=0)")
    ylabel("Probability of finding one particle")
    xlabel("$k_z$ [${\\rm Å}^{-1}$]")
    #ylim((0,1.1))
    plot(kzlist*Å,density_list,'g')
    
    #
    #figure(3)
    Emax = max(amax(abs(E1list)),amax(abs(E2list)))
    plot(kzlist*Å,E1list/Emax,'k')#/max(abs(E1list)))
    plot(kzlist*Å,E2list/Emax,'k')#/max(abs(E1list)))
    plot(kzlist*Å,Mu/Emax+0*kzlist,'r')
    
    
    print(f"Total power: {sum(P1_list)*dk/P0,sum(P2_list)*dk/P0}")
    #plot(kzlist,EFlist/Emax,'b')
    #
    
    #
    figure(3)
    title("Average pumping per particle vs  z-crystal momentum (kx,ky=0)")
    ylabel("Probability of finding one particle")
    xlabel("$k_z$")
    plot(kzlist,P1_list/(P0*density_list),"g")
    #ylabel("Pumping power [$\omega_1\omega_2/2\pi$]")
    
    legend(["Mode 1","Mode 2","Dissipation"])
