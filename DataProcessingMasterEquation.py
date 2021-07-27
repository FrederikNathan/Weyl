#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TDS = 0     : using RGF solver, 
TDS = 1     : using time domain solver; 
TDS = 2     : solution method was not recorded 

Script for updating files (transforming raw data to processed), and to load it as a python object.
Also contains some parameter printing functions.

Explanation

FileList ndarray(NF), str: List of filenames 
ParameterList: ndarray(NP,X) List of parameters. System parameterized by X parameters
PPointList: Parameter pointers ndarray(NK), int. data point z (z=1...NK) has parameters ParameterList[PPointList[z]])
klist: ndarray(NK,3), float. k points
P1list: ndarrray(NK), float. conversion power to mode 1
P2list: ndarray(NK), float. conversion power to mode 2
Nlist: ndarray(NK), float. particle density
NDP: ndarray(NP), int. Number of data points for each parameter set
Ess: ndrarray(NK): steady state energy
Eeq: ndarray(NK), flaot: equiliibrium energy


Parameters:
    omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp  = ParameterList[z]
    
    

    
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
import Master_equation_multimode_v5 as MA
import multiprocessing as mp
import time as time 
from matplotlib.pyplot import *

import scipy.interpolate as scint
#import pandas as pan
from Units import *


ArchivePath = "../Processed/Archive.npz"
ArchivePath_sc = "../Processed/Archive_sc.npz"
DataDir = "../Data/"
Files = os.listdir(DataDir)

### TDS = 0: using RGF solver, TDS=1: using time domain solver; TDS = 2: not known
def load_archive(update=True):
    
    if not update:
            
        try:
            Archive=load(ArchivePath)
            FileList = list(Archive["FileList"])
            ParameterList = Archive["ParameterList"]
            PPointList = Archive["PPointList"]
            klist = Archive["klist"]
            P1list = Archive["P1list"]
            P2list = Archive["P2list"]    
            Nlist = Archive["Nlist"]
            NDP = Archive["NDP"]
            Esslist = Archive["Esslist"]
            Eeqlist = Archive["Eeqlist"]
            TDSlist = Archive["TDSlist"]
#            Convlist = Archive["Convlist"]
#            Crlist=  Archive["Crlist"]
        except FileNotFoundError:
            FileList = []
            ParameterList = zeros((0,11))
            PPointList = zeros(0,dtype=int)
            klist  = zeros((0,3))
            P1list = zeros(0)
            P2list = zeros(0)
            Nlist = zeros(0)
            NDP   = zeros(0,dtype=int)
            Esslist = zeros(0)
            Eeqlist = zeros(0)
            TDSlist = zeros(0,dtype=int)
            #            Convlist=zeros((0,2),dtype=int)
#            Crlist = zeros(0)

        return FileList,ParameterList,PPointList,klist,P1list,P2list,Nlist,NDP,Esslist,Eeqlist,TDSlist

    else:
        return update_archive()
        



def update_archive():
    
    FileList,ParameterList,PPointList,klist,P1list,P2list,Nlist,NDP,Esslist,Eeqlist,TDSlist = load_archive(update=False)        
    nf = 0
    global D
    for file in Files:
        nf+=1 
        
        if nf%5000==0:
            print(f"At file {nf}/{len(Files)}")
        try:
                
            if file[-4:] == ".npz":
                if not file in FileList:
                    FileList.append(file)
                    
                    D = load(DataDir+file)
                    
                    K = D["klist"]
                    NK = shape(K)[0]
                    Parameters=D["parameterlist"]
                    P1 = D["P1_list"]
                    P2 = D["P2_list"]
                    N = D["density_list"]
                    Ess = D["Ess_list"]
                    Eeq = D["Eeq_list"]
                    
                    try:
                        TDS = D["use_tds_list"]*1
                    except KeyError:
                        TDS = 2*ones((NK),dtype=int)
                        
                    
                        
    #                Conv = D["convlist"]
    #                Cr = D["crlist"]
                    
                    NP1 = shape(Parameters)[0]
                    if shape(ParameterList)[0]==0:
                        ParameterList=Parameters[0,:].reshape((1,11))           
    
                    nit=0
                    while True:
                        NP0 = shape(ParameterList)[0]
                        Mat = zeros((NP1,NP0))
                        
                        for n in range(0,NP0):
                            Mat[:,n]=sum(abs(Parameters-ParameterList[n,:]),axis=1)
                     
                        
                        
                        if amax(amin(Mat,axis=1))<1e-9:
                            ParameterPointer = argmin(Mat,axis=1)                    
                            break
                        else:
                            N1 = argmax(amin(Mat,axis=1))
                            
                            ParameterList = concatenate((ParameterList,1*Parameters[N1].reshape((1,11))))
                        nit+=1
                          
                    
                    PPointList = concatenate((PPointList,ParameterPointer))
                    klist = concatenate((klist,K))
                    P1list = concatenate((P1list,P1))
                    P2list = concatenate((P2list,P2))
                    Nlist = concatenate((Nlist,N))
                    Eeqlist =  concatenate((Eeqlist,Eeq))
                    Esslist =  concatenate((Esslist,Ess))
                    
                    TDSlist = concatenate((TDSlist,TDS))
    #                Convlist = concatenate((Convlist,Conv))
    #                Crlist=concatenate((Crlist,Cr))
                
        except KeyError:
            print(f"File {file} outdated. Remove? (y/n)")
            
            while True:
                Input = input()
                if Input == "y":
                    os.system(f"rm {DataDir+file}")
                    print("removed file")
                    break
                elif Input=="n":
                    print("skipped file")
                    break
                else:
                    print("Invalid input. Try again")
    Nparms = shape(ParameterList)[0]
    NDP = zeros(Nparms,dtype=int)
    for n in range(0,Nparms):
        NDP[n] = sum(PPointList==n)
        
    
        

    savez(ArchivePath,FileList=FileList,ParameterList=ParameterList,PPointList=PPointList,klist=klist,P1list=P1list,P2list=P2list,Nlist=Nlist,NDP=NDP,Eeqlist=Eeqlist,Esslist=Esslist,
          TDSlist=TDSlist)
    
    savez(ArchivePath_sc,FileList=FileList,ParameterList=ParameterList,PPointList=PPointList,klist=klist,P1list=P1list,P2list=P2list,Nlist=Nlist,NDP=NDP,Eeqlist=Eeqlist,Esslist=Esslist,
          TDSlist=TDSlist)
    
                
    return FileList,ParameterList,PPointList,klist,P1list,P2list,Nlist,NDP,Eeqlist,Esslist,TDSlist
            
#class archive():
#    def __init___(self):            
#        FileList,ParameterList,PPointList,klist,P1list,Nlist     =load_archive()
#        self.filelist = FileList
#        self.parameterpointer = PPointList
#        self.klist=klist
#        self.p1list = p1list
#        self.p2list = p2list
#        self.

def parameterprint(n):
    omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp = pl[n,:]
    

    print("-"*80)
    print(f"    omega_1  :  {omega1/THz:<8.4} THz")
    print(f"    omega_2  :  {omega2/THz:<8.4} THz\n")
    print(f"    tau      :  {tau/picosecond:<8.4} ps\n")
    print(f"    vF       :  {vF/(meter/second):<8.4} m/s")
    print(f"    v0       :  {V0z/(meter/second):<8.4} m/s\n")
    print(f"    E1       :  {EF1/(Volt/meter):<8.4} V/m")
    print(f"    E2       :  {EF2/(Volt/meter):<8.4} V/m\n")
    print(f"    mu       :  {Mu/meV:<8.4} meV")
    print(f"    Temp     :  {Temp/Kelvin:<8.4} K")
#    print(f"    Npmax    :  {NPmax}")




    
print("Loading files")
FileList,pl,pp,kl,P1list,P2list,Nlist,NDP,Eeqlist,Esslist,TDSlist = load_archive()    
NPARMS = len(NDP)
#pl : parameterlist
#pp : parameter pointer

print("   done.\n")
    
     
def print_parameters():
    N = shape(pl)[0]
    for n in range(0,N):
        
        print("="*80)
        print(f"   Parameter set {n}       {NDP[n]}   data points")
        parameterprint(n)
        print("")
        
   
    
def pprint(n=None):
    if n==None:
        
        print_parameters()
    else:
        print("="*80)
        print(f"   Parameter set {n}       {NDP[n]}   data points")
        parameterprint(n)
        print("\n"+"-"*80)
        
def get_data(n,disp=True):
    parameters = pl[n,:]
    global Ind
    Ind = where(pp==n)[0]
    Npoints = len(Ind)
    
    k_out = kl[Ind,:]
    P1_out = P1list[Ind]
    P2_out = P2list[Ind]
    N_out = Nlist[Ind]
    Eeq = Eeqlist[Ind]
    Ess = Esslist[Ind]
    TDS = TDSlist[Ind]

    if disp:
            
        pprint(n=n)
#        print(f"Number of data points: {Npoints}")
#        print("-"*80)
    return parameters,k_out,P1_out,P2_out,N_out,Eeq,Ess,TDS
    
    
    
    
    
    
    
    