#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:12:17 2020

@author: frederik
"""

from numpy import * 
from numpy.random import *
from matplotlib.pyplot import *

#Nout = []
def compute_cube_grid(k,Data,width,order=5,center=(0,0,0)):
    """
    Compute cube grid. 
    Grid consists of cubes of width <width>/2**n
    """
    global CS,ND,Z,Nlist,B1,B2,Countlist
    N = shape(Data)[0]
    
    if len(shape(Data))>1:
        
        ND = shape(Data)[1]
    else:
        ND = 1 
        Data= Data.reshape((N,1))
        
    k = k-center
    
    if N==0:
        raise ValueError("Empty set passed to interpolator")
    
    if amax(abs(k))>width/2:
        raise ValueError("Some k points are located outside of grid. Crop input and try again")
        
    global k1,nx,ny,nz,M,dx,n,xvec,B1,CS,nm,A
        
    M = 1 
    
    for n in range(0,order+1):
#        print(f"  at order {n}")
        xvec = width*(arange(0,M)/M-0.5)

        if n>0:
            
            dx = xvec[1]-xvec[0]
        else:
            dx = width 
        
        k1 = ((k+width/2)/dx).astype(int)
        
        [nx,ny,nz]=list(k1.T)
        
        
        nm = ravel_multi_index((nx,ny,nz),(M,M,M))
        
        Y = zeros((M,M,M),dtype=float)
        
        
        AS = argsort(nm)
        k1=k1[AS]
        nm=nm[AS]
        Y0=1*Data[AS,:]
        
        CS = cumsum(Y0,axis=0)
        CS = concatenate((array([[0]]*ND).T,CS))
        
        A = searchsorted(nm,arange(0,M**3))
        B1 = concatenate((A[1:],array([N])))
        B2 = A
        
        Nlist = B1-B2
        Countlist=1*Nlist.reshape((M,M,M))
        Z = (CS[B1,:]-CS[B2,:])
        
        Z = Z.reshape((M,M,M,ND))
        Nlist = Nlist.reshape((M,M,M,1))
        
        M=M*2
    
    
    
        if n>0:
            N1,N2,N3 = where(Nlist==0)[:3]
            
            (M1,M2,M3) = (x//2 for x in (N1,N2,N3))
            
            Nlist[N1,N2,N3] = Nlist_old[M1,M2,M3]
            Z[N1,N2,N3] = Z_old[M1,M2,M3]
            
        if n<10:
                
            Nlist_old = 1*Nlist
            Z_old = 1*Z
            
            

        
        Y = Z/Nlist

    return [xvec+q for q in center],Y,Nlist,Countlist #Nlist

def compute_square_grid(k,Data,width,order=5,center=(0,0)):
    """
    Compute square grid. 
    Grid consists of cubes of width <width>/2**n
    """
    global CS,ND,Z,Nlist,B1,B2,Countlist
    N = shape(Data)[0]
    
    if len(shape(Data))>1:
        
        ND = shape(Data)[1]
    else:
        ND = 1 
        Data= Data.reshape((N,1))
        
    k = k-center
    
    if N==0:
        raise ValueError("Empty set passed to interpolator")
    
    if amax(abs(k))>width/2:
        print(amax(abs(k)))
        print(width/2)
        raise ValueError("Some k points are located outside of grid. Crop input and try again")
        
    global k1,nx,ny,nz,M,dx,n,xvec,B1,CS,nm,A
        
    M = 1 
    
    for n in range(0,order+1):
#        print(f"  at order {n}")
        xvec = width*(arange(0,M)/M-0.5)

        if n>0:
            
            dx = xvec[1]-xvec[0]
        else:
            dx = width 
        
        k1 = ((k+width/2)/dx).astype(int)
        
        [ny,nz]=list(k1.T)
        
        
        nm = ravel_multi_index((ny,nz),(M,M))
        
        Y = zeros((M,M),dtype=float)
        
        
        AS = argsort(nm)
        k1=k1[AS]
        nm=nm[AS]
        Y0=1*Data[AS,:]
        
        CS = cumsum(Y0,axis=0)
        CS = concatenate((array([[0]]*ND).T,CS))
        
        A = searchsorted(nm,arange(0,M**2))
        B1 = concatenate((A[1:],array([N])))
        B2 = A
        
        Nlist = B1-B2
        Countlist=1*Nlist.reshape((M,M))
        Z = (CS[B1,:]-CS[B2,:])
        
        Z = Z.reshape((M,M,ND))
        Nlist = Nlist.reshape((M,M,1))
        
        M=M*2
    
#        raise OSError
    
        if n>0:
            N2,N3 = where(Nlist==0)[:2]
            
            (M2,M3) = (x//2 for x in (N2,N3))
            
            Nlist[N2,N3] = Nlist_old[M2,M3]
            Z[N2,N3] = Z_old[M2,M3]
            
        if n<10:
                
            Nlist_old = 1*Nlist
            Z_old = 1*Z
            
            

        
        Y = Z/Nlist

    return [xvec+q for q in center],Y,Nlist,Countlist #Nlist





 
def compute_grid_square(k,Data,center,cubewidth,ncubes,order=3):
    """
    Compute data to grid. 
    Grid is centered at <center> and has dimension (nx,ny,nz)*cubewidth, where (nx,ny,nz)=ncubes.
    Here nx,ny,nz must be integers. 
    The grid has spacing cubewidth/2**order.
    
    Returns k_out,Y,Nlist
    
    k_out : 3 vectors spanning the grid
    Y     : Data interpolated to grid
    Nlist : Number of data point for each cube on the grid. 
    
    """
    
    N = shape(Data)[0]
    
    if len(shape(Data))>1:
        
        ND = shape(Data)[1]
    else:
        ND = 1 
        Data= Data.reshape((N,1))
        
    MY,MZ = ncubes
    
#    SX = center[0]+cubewidth*(arange(MX+1)-(MX)/2)
    SY = center[0]+cubewidth*(arange(MY+1)-(MY)/2)
    SZ = center[1]+cubewidth*(arange(MZ+1)-(MZ)/2)
    
    [ky,kz]=k.T
    K=1*k
    
    
    NC = 2**order
    
    dk = cubewidth/NC
    
    Y_out  =  zeros((MY*2**order,MZ*2**order,ND))
    N_out = zeros((MY*2**order,MZ*2**order),dtype=int)
    
#    kx_out = (arange(NC*MX)-NC*MX/2)*dk+center[0]
    ky_out = (arange(NC*MY)-NC*MY/2)*dk+center[0]
    kz_out = (arange(NC*MZ)-NC*MZ/2)*dk+center[1]
    
    n_count = 0

#        
#        BX1  =SX[mx]
#        BX2 = SX[mx+1]
#        IX = (K[:,0]<BX2)*(K[:,0]>=BX1)
#        
#        Kx = K[IX,:]
#        Dx = Data[IX]
#        

    for my in range(0,MY):
        
        
            
        BY1  =SY[my]
        BY2 = SY[my+1]
        IY = (K[:,0]<BY2)*(K[:,0]>=BY1)
        
        Dy = Data[IY]
        Ky = K[IY,:]
        
        for mz in range(0,MZ):

            BZ1  =SZ[mz]
            BZ2 = SZ[mz+1]
            IZ = (Ky[:,1]<BZ2)*(Ky[:,1]>=BZ1)

            if sum(IZ)==0:
                continue 
            Dz = Dy[IZ]
            Kz = Ky[IZ,:]
        
            center = tuple(x+cubewidth*0.5 for x in (BY1,BZ1))
            width = cubewidth
            
            Xspan,Y,Nlist,Countlist = compute_square_grid(Kz,Dz,width,center=center,order=order)
            
            
            Y_out[my*NC:(my+1)*NC,mz*NC:(mz+1)*NC,:]=Y
            N_out[my*NC:(my+1)*NC,mz*NC:(mz+1)*NC]=Countlist#[:,:,:,0]
            n_count +=1 
            
            
#                print(f"   At step {n_count}")
                
    return (ky_out,kz_out),N_out,Y_out       
        
            
    

def compute_grid(k,Data,center,cubewidth,ncubes,order=3):
    """
    Compute data to grid. 
    Grid is centered at <center> and has dimension (nx,ny,nz)*cubewidth, where (nx,ny,nz)=ncubes.
    Here nx,ny,nz must be integers. 
    The grid has spacing cubewidth/2**order.
    
    Returns k_out,Y,Nlist
    
    k_out : 3 vectors spanning the grid
    Y     : Data interpolated to grid
    Nlist : Number of data point for each cube on the grid. 
    
    """
    
    N = shape(Data)[0]
    
    if len(shape(Data))>1:
        
        ND = shape(Data)[1]
    else:
        ND = 1 
        Data= Data.reshape((N,1))
        
    MX,MY,MZ = ncubes
    
    SX = center[0]+cubewidth*(arange(MX+1)-(MX)/2)
    SY = center[1]+cubewidth*(arange(MY+1)-(MY)/2)
    SZ = center[2]+cubewidth*(arange(MZ+1)-(MZ)/2)
    
    [kx,ky,kz]=k.T
    K=1*k
    
    
    NC = 2**order
    
    dk = cubewidth/NC
    
    Y_out  =  zeros((MX*2**order,MY*2**order,MZ*2**order,ND))
    N_out = zeros((MX*2**order,MY*2**order,MZ*2**order),dtype=int)
    
    kx_out = (arange(NC*MX)-NC*MX/2)*dk+center[0]
    ky_out = (arange(NC*MY)-NC*MY/2)*dk+center[1]
    kz_out = (arange(NC*MZ)-NC*MZ/2)*dk+center[2]
    
    n_count = 0
    for mx in range(0,MX):
        
        BX1  =SX[mx]
        BX2 = SX[mx+1]
        IX = (K[:,0]<BX2)*(K[:,0]>=BX1)
        
        Kx = K[IX,:]
        Dx = Data[IX]
        

        for my in range(0,MY):
            
            
                
            BY1  =SY[my]
            BY2 = SY[my+1]
            IY = (Kx[:,1]<BY2)*(Kx[:,1]>=BY1)
            
            Dy = Dx[IY]
            Ky = Kx[IY,:]
            
            for mz in range(0,MZ):
    
                BZ1  =SZ[mz]
                BZ2 = SZ[mz+1]
                IZ = (Ky[:,2]<BZ2)*(Ky[:,2]>=BZ1)
    
                if sum(IZ)==0:
                    continue 
                Dz = Dy[IZ]
                Kz = Ky[IZ,:]
            
                center = tuple(x+cubewidth*0.5 for x in (BX1,BY1,BZ1))
                width = cubewidth
                Xspan,Y,Nlist,Countlist = compute_cube_grid(Kz,Dz,width,center=center,order=order)
                
                
                Y_out[mx*NC:(mx+1)*NC,my*NC:(my+1)*NC,mz*NC:(mz+1)*NC,:]=Y
                N_out[mx*NC:(mx+1)*NC,my*NC:(my+1)*NC,mz*NC:(mz+1)*NC]=Countlist#[:,:,:,0]
                n_count +=1 
                
                
#                print(f"   At step {n_count}")
                
    return (kx_out,ky_out,kz_out),N_out,Y_out
