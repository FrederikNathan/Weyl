#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for saving raw data files to a combined npz archive.
Automatically loads data.

Also contains some parameter printing functions.

TDS = 0     : using RGF solver, 
TDS = 1     : using time domain solver; 
TDS = 2     : solution method was not recorded 

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
import numpy.random as npr
import scipy.optimize as optimize
import multiprocessing as mp
import time as time 
from matplotlib.pyplot import *
import scipy.interpolate as scint

import basic as B
from units import *


ArchivePath = "../Processed/Archive.npz"
ArchivePath_sc = "../Processed/Archive_sc.npz"
DataDir = "../Data/"
Files = os.listdir(DataDir)

Times  = array([int(X[-22:-16]+X[-15:-11]) for X in Files])
AS = argsort(Times)
Files = list(array(Files)[AS])
Times = Times[AS]

### TDS = 0: using RGF solver, TDS=1: using time domain solver; TDS = 2: not known
def load_archive(update=True):
    if update:
        update_archive()

        
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
        DateList = Archive["DateList"]
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
        DateList = zeros(0,dtype=int)
        #            Convlist=zeros((0,2),dtype=int)
#            Crlist = zeros(0)

    return FileList,ParameterList,PPointList,klist,P1list,P2list,Nlist,NDP,Eeqlist,Esslist,TDSlist,DateList

    # else:
    #     return update_archive()
        



def update_archive():
        
    FileList,ParameterList,PPointList,klist,P1list,P2list,Nlist,NDP,Eeqlist,Esslist,TDSlist,DateList = load_archive(update=False)        
    nf = 0

    NP0 = 0
    
    B.tic()
    for file in Files:
        
        
        nf+=1 
        if nf%5000==0:
            print(f"At file {nf}/{len(Files)}.")#"\n    Parameter sets recorded: {NP0} \n    Time spent: {B.toc(disp=False):.4} s")
            # print(f"    ")
        try:
                
            if file[-4:] == ".npz":
                
                Date = int(file[-22:-16]+file[-15:-11])
                
                if (not file in FileList) and Date>2108040900:
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
                    
                    # Date = int(file[-22:-16]+file[-15:-11])
                    
                    NP1 = shape(Parameters)[0]

                    Date_array = ones((NP1),dtype=int)*Date
                    
                    try:
                        TDS = D["use_tds_list"]*1
                    except KeyError:
                        TDS = 2*ones((NK),dtype=int)
                        
                    
                        
    #                Conv = D["convlist"]
    #                Cr = D["crlist"]
                    
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
                    DateList = concatenate((DateList,Date_array))
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
          TDSlist=TDSlist,DateList=DateList)
    
    savez(ArchivePath_sc,FileList=FileList,ParameterList=ParameterList,PPointList=PPointList,klist=klist,P1list=P1list,P2list=P2list,Nlist=Nlist,NDP=NDP,Eeqlist=Eeqlist,Esslist=Esslist,
          TDSlist=TDSlist,DateList=DateList)
    
                
    # return FileList,ParameterList,PPointList,klist,P1list,P2list,Nlist,NDP,Eeqlist,Esslist,TDSlist
            
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
    # first_date,last_date = get_dates(n)

    print("-"*80)
    print(f"    omega_1            :  {omega1/THz:<8.4} THz")
    print(f"    omega_2            :  {omega2/THz:<8.4} THz")
    print(f"    ratio              :  {omega2/omega1:<8.9}\n")
    print(f"    tau                :  {tau/picosecond:<8.4} ps\n")
    print(f"    vF                 :  {vF/(meter/second):<8.4} m/s")
    print(f"    v0                 :  {V0z/(meter/second):<8.4} m/s\n")
    print(f"    E1                 :  {EF1/(Volt/meter):<8.4} V/m")
    print(f"    E2                 :  {EF2/(Volt/meter):<8.4} V/m\n")
    print(f"    mu                 :  {Mu/meV:<8.4} meV")
    print(f"    Temp               :  {Temp/Kelvin:<8.4} K\n")
    # print("-"*80)

#    print(f"    Npmax    :  {NPmax}")




    
print("Loading files")
FileList,pl,pp,kl,P1list,P2list,Nlist,NDP,Eeqlist,Esslist,TDSlist,DateList = load_archive(update=False)    
frequency_ratio = pl[:,1]/pl[:,0]
omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp = [pl[:,n] for n in range(0,shape(pl)[1])]
NPARMS = len(NDP)
print("   done.\n")
    
     
def print_parameters():
    N = shape(pl)[0]
    for n in range(0,N):
        first_dstr,last_dstr = [format_datestr(x) for x in get_dates(n)]

        print("="*80)
        print(f"   Parameter set {n}       {NDP[n]}   data points")
        print("-"*80)
        print(f"    First run          :  {first_dstr}  ")
        print(f"    Last  run          :  {last_dstr} ")
        print("")
        parameterprint(n)
        print("")
        

def format_datestr(datestr):
    year = datestr[:2]
    month = datestr[2:4]
    day   = datestr[4:6]
    hour  = datestr[6:8]
    minute = datestr[8:10]
   
    outstr = f"{day}/{month} 20{year} {hour}:{minute}"
    return outstr

def get_dates(n):
    """
    Get first and last date of data in set n

    Parameters
    ----------
    n : int
        Index of parameter set.

    Returns
    -------
    first_date : int
        Date and time for first run in parameter set.
    last_date : int
        Date and time for last run in parameter set.

    """
    Ind = where(pp==n)[0]
    
    Dates = DateList[Ind]
    first_date = str(amin(Dates))
    last_date  = str(amax(Dates))
    
    # date_first = format_datestr(first_date)
    # date_last  = format_datestr(last_date)    

    return first_date,last_date
    
def pprint(n=None):
    if n==None:
        
        print_parameters()
    else:
        first_dstr,last_dstr = [format_datestr(x) for x in get_dates(n)]
        print("="*80)
        print(f"   Parameter set {n}        {NDP[n]}   data points")
        # print("")

        # print("")

        parameterprint(n)
        print("-"*80)
        print(f"    First run          :  {first_dstr}  ")
        print(f"    Last run           :  {last_dstr} ")
        print("-"*80)
        
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
    

# pprint(n=7)
    
    
    
    
    
    
    