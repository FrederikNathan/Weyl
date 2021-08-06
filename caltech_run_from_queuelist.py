#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to launch a weyl sweep. 
"""

import sys
import os
import weyl_queue as Q
#import basic as B
from socket import gethostname

# Test if just launching here. 
if len(sys.argv)<2:
    # Queue="../Queues/test_ca.npz"
    N_runs = 4
    Serieslength = 5
else:
        
    queue_list = sys.argv[1]
    N_runs = int(sys.argv[2])
    Serieslength = int(sys.argv[3])
    

Q.launch_from_queue_list("caltech",queue_list,N_runs,Serieslength)


#script  = "pbs_script.sh"
#argstring = f"LF={logfile},PRESTR='{prestr}',Q={Queue},SL={Serieslength}"
#qsub_str=f'qsub -N {job_name} -l walltime={Tmax_h}:{Tmax_m}:{Tmax_s} -v {argstring} {script} &'
#os.system(qsub_str)
#time.sleep(0.2*(1+npr.rand()))
        