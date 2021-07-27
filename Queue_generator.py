#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:31:50 2020

@author: frederik
"""

import socket

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

QueueDir = "../Queues/"

HN = socket.gethostname()
print(HN)
if HN=="Frederiks-MacBook-Pro-2.local":
    HN = "mac"
elif HN=="wakanda" or HN== "yukawa":
    HN = "yu"
elif HN=="cmtrack2.caltech.edu":
    HN="wa"
else:
    raise SystemError(f"Host {HN} not recognized")
def add_to_queue(parameterlist,klist):
    print("Enter name of queue:")
    QueueName = input()    
    
    QueuePath=QueueDir+QueueName+"_"+HN+".npz"

    NK= shape(klist)[0]
    
    if os.path.exists(QueuePath):
        D=load(QueuePath)
        PL = D["Parameterlist"]
        KL = D["Klist"]
        SV = D["Status"]
        
        N0 = sum(SV==0)
        N1 = sum(SV==1)
        N2 = sum(SV==2)
        print(f"\nQueue {QueueName} already exists, with {N0+N1+N2} runs. ({N0} waiting, {N1} running, {N2} finished). \n\nAdd {NK} runs to queue? (y/n)\n")
        while True:
            
            I = input()
            
            if I=="y":
                break
            elif I=="n":
                break
            else:
                print("Invalid input. Try again.")
    
        if I=="y":
            
            D=load(QueuePath)
            PL = D["Parameterlist"]
            KL = D["Klist"]
            SV = D["Status"]
            
            PL=concatenate((PL,parameterlist))
            KL=concatenate((KL,klist))
            SV=concatenate((SV,zeros(NK,dtype=int)))
            
            
            savez(QueuePath,Parameterlist=PL,Klist=KL,Status=SV)
            
            print(f"\nAdded {NK} runs to queue {QueueName}\n")
    else:
        print(f"\nQueue {QueueName}.npz will be generated, with {NK} runs. Proceed? (y/n)")
        while True:
            I = input()
            
            if I=="y":
                break
            elif I=="n":
                break
            else:
                print("Invalid input. Try again.")
                
                
        if I=="y":
            savez(QueuePath,Parameterlist=parameterlist,Klist=klist,Status=zeros(NK,dtype=int))
            print(f"\nAdded {NK} runs to queue {QueueName}")
