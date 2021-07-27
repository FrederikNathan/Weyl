#!/usr/bin/env python3
# -*- coding: utf-8 -*-sshpass -p "Lervangen6419" ssh qxn582@yukawa.int.nbi.dk "rm -r EnergyPump/Code/*"  
"""
Created on Thu Sep 24 12:21:24 2020

@author: frederik

Python script to execute when launching a single simulation run
"""
import sys
from weyl_queue import *
logfile = str(sys.argv[1])
prestr = str(sys.argv[2])
queue = str(sys.argv[3])
serieslength=int(sys.argv[4])


print(f"sys.argv: {sys.argv}")
print(f"prestr : {prestr}")
print(f"logfile: {logfile}")
print(f"queue  : {queue}")
print(f"sl     : {serieslength}")
B.redirect_output(logfile,prestr=prestr,timestamp=True)
readfromqueue(queue,serieslength)
