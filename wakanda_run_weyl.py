#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import weyl_queue as Q
import socket
import sys

#try:
        
Queue = sys.argv[1]
N_parallel = int(sys.argv[2])
Serieslength = int(sys.argv[3])

#except:
#    pass

HN = socket.gethostname()
    
HN_mac = "Frederiks-MacBook-Pro-2.local";
HN_wa = 'wakanda'
HN_yu = "yukawa"
HM_ca = "cmtrack2.caltech.edu"


#if HN == HN_mac or HN==HN_wa or HN ==HN_yu:
hour = 3600
day = 24*hour
week = 7*day

    
live_time = 1*week

Q.wakanda_launch(Queue,N_parallel,Serieslength,TimeOut = 1*week)


#else:
#    raise SystemError("Host not recognized")    
#
#    











