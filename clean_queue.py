#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:31:50 2020

@author: frederik
"""
import os 
from scipy import *
from scipy.linalg import * 
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy import * 

import numpy.random as npr
import scipy.optimize as optimize
import basic as B
from Units import * 
import sys 
QueueDir = "../Queues/"

QueuePath = sys.argv[1]
try:
    Num = int(sys.argv[2])
    Partial = 1
    
except:
    Partial = 0
    


D=load(QueuePath)
PL = D["Parameterlist"]
KL = D["Klist"]
SV = D["Status"]

if Partial:
    Ind = where(SV[:Num]==1)[0]
else:
    Ind = where(SV==1)[0]
    
N0 = sum(SV==0)
N1 = sum(SV==1)
N2 = sum(SV==2)
print(f"\nStatus of queue {QueuePath[10:]} : \n\n     {N0+N1+N2} runs ({N0} waiting, {N1} running, {N2} finished)")
print("")


if Partial == False:
        
    print(f"Change status of all {N1} 'running' jobs back to 'waiting'? (y/n). Press 'f' if status should be changed to done")

else:
    print(f"There are {len(Ind)} running jobs among the first {Num} jobs. Change status of these back to 'waiting'? (y/n). Press 'f' if status should be changed to done")



while True:
    
    I = input()
    
    if I=="y":
        break
    elif I=="n":
        break
    elif I=="f":
        print("Change status to done. Sure? (y/n)")
        while True:
            
            I2 = input()
            if I2=="y":
                break
            elif I2=="n":
                I="n"
                break
            else:
                print("Invalid input. Try again.")

        break
    else:
        print("Invalid input. Try again.")
        
        

        
if I=="y" or I=="f":
    
    D=load(QueuePath)
    PL = D["Parameterlist"]
    KL = D["Klist"]
    SV = D["Status"]
    
    
    if I=="y":
        SV[Ind]=0        
        savez(QueuePath,Parameterlist=PL,Klist=KL,Status=SV)
        print("")
        print(f"Changed status of the first {len(Ind)} 'running' jobs to 'waiting' in Queue {QueuePath[10:]}\n")
    if I=="f":
        SV[Ind]=2        
        savez(QueuePath,Parameterlist=PL,Klist=KL,Status=SV)

        print("")
        print(f"Changed status of the first {len(Ind)} 'running' jobs to 'done' in Queue {QueuePath[10:]}\n")

    
else:
    print("\nDid not change anything\n")
    