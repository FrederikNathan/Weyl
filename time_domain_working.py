#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 08:51:26 2020

@author: frederik

Solution of master equation in time domain
"""

from scipy import *
from matplotlib.pyplot import *
from Units import *
import weyl_liouvillian as wl

omega1 = 20*THz
omega1 = 2*pi
omega2 = 0.61803398875*omega1
#tau    = 50*picosecond
tau = 10
vF     = 1e3*meter/second
T1 = 2*pi/omega1
T2 = 2*pi/omega2
EF1 = 0.6*1.5*2e6*Volt/meter
EF2 = 0.6*1.25*1.2e6*Volt/meter*0

Mu =15*meV*0
Temp  = 25*Kelvin;

V0 = array([0,0,0.8*vF])*1
[V0x,V0y,V0z] = V0


k = array([[0.1,0.2,0.3]])
parameters = [omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp]

wl.set_parameters(parameters)

h = wl.get_h(k)
hc = wl.get_hc(k)
r0 = wl.get_rhoeq(k)[1]
r0 = ve.mat_to_vec(r0)



def get_unitary_array(kgrid,dt=None):
    """
    get array of e^{-ih(\phi_1,\phi_2)dt} for \phi_1,\phi_2 = (0,1,...N)*2pi/N
    """
    
    global generator,Nmax
    
    S = shape(kgrid)[0]
    Nphi = int(sqrt(S)+0.1)
    assert abs(Nphi**2-S) < 1e-15
    
    
    Out = array([eye(2,dtype=complex) for n in range(0,S)])# zeros((S,4,4),dtype=complex)
    global X
    h  =wl.get_h(kgrid)
    Amax = amax(norm(h,axis=(1,2)))
    
    if dt==None:
        
        dt =0.1/Amax

    generator = -1j*h*dt

    X = 1*generator 
    
    n=1
    while True:
        Out += X
        
        XN = norm(X)
        if XN<1e-30:
            print(f"Unitaries converged with n={n}")
            break


        X = (generator @ X)
        X= X/(n+1)
        n=n+1
        

    return Out.reshape((Nphi,Nphi,2,2)),dt
    

def get_steadystate(t0,dt,U_array,Rgrid):

    global svec 


    U0 = eye(2,dtype=complex)
    

    R0  =zeros((2,2),dtype=complex)
    
    ns = 0
    R0list = []
    global T0,svec,ind1,ind2
    svec =arange(t0,t0-32*tau,-dt)

    B.tic()
    T0 =0
    ind1list=[]
    for s in svec:

        ind1,ind2 = get_ind(s-dt,Nphi)

   
        R0  += U0@Rgrid[ind1,ind2]@(U0.conj().T)*dt*exp(-(t0-s)/tau)/tau
        T0 += dt*exp(-(t0-s)/tau)/tau
        if isnan(R0[0,0]):
            raise ValueError
         
        dU = 1*U_array[ind1,ind2]
        U0 = U0@dU
        
        ind1list.append(ind1)


        
        ns +=1
        
    return R0,R0list,T0,ind1list,U0
   
    
def get_ind(t,Nphi):
    ind1 = int(t/T1+0.1*pi)%Nphi
    ind2 = int(t/T2+0.1*pi)%Nphi
    return (ind1,ind2)



Nphi=200
kgrid = wl.get_kgrid(Nphi,k0=k[0])

dt =0.5*T1*0.01

U_array,dt=get_unitary_array(kgrid,dt);
#U_array = ones(shape(U_array)[:2])
Rgrid = wl.get_rhoeq(kgrid)[1].reshape((Nphi,Nphi,2,2))
t= T1/20
R0,R0list,A0,ind0list,U00 =get_steadystate(t,dt,U_array,Rgrid)
R = 1*R0
Rlist = []

nt = 0
Out = T0

U= eye(2)

for t in arange(t,T1,dt):
    nt+=1
    ind1,ind2 = get_ind(t,Nphi)
    ind1_m = int(((t-dt)/T1)%1 *Nphi)#        print(t0)
    ind2_m = int(((t-dt)/T2)%1 *Nphi)#  
    
    dU = 1*U_array[ind1,ind2]
    
    R = exp(-dt/tau)*R 
    R = dU@R@(dU.conj().T)
    R = R+ dt/tau*Rgrid[ind1,ind2]
    Out =exp(-dt/tau)*Out + dt/tau
    
    U = dU@U
    if nt%1000==0:
        Rlist.append(1*R)
        
    
#
R1,R1list,A1,ind1list,U01 = get_steadystate(t+dt,dt,U_array,Rgrid)
print("Answer")
print(R)
print("")
print("Actual answer")
print(R1)
print("Norm of difference:")
print(norm(R1-R))


NR0 = len(R0list)
NR = len(Rlist)
Rlist = array(Rlist).reshape((NR,4))
plot(abs(Rlist))
figure(2)
R0list = array(R0list).reshape(NR0,4)
plot(abs(R0list))
#    
#    
#
#
#







