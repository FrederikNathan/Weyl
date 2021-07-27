#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import socket
import sys
import os
import weyl_queue as Q
import basic as B

if len(sys.argv)<2:
    Queue="../Queues/newtest_mac.npz"
    N_runs = 4
    Serieslength = 5
else:
        
    Queue = sys.argv[1]
    N_runs = int(sys.argv[2])
    Serieslength = int(sys.argv[3])
    
Q.slurm_launch(Queue,N_runs,Serieslength)
#print("\n"*2)
#print(A[0])