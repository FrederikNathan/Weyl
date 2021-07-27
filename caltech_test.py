#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import socket
import sys
import os
import weyl_queue as Q
import basic as B


#try:
sys.argv[1:] = ["../Queues/newtest.npz","30","20"]
sys
Queue = sys.argv[1]
N_runs = int(sys.argv[2])
Serieslength = int(sys.argv[3])
Ndp = int(N_runs*Serieslength+1)


QueueData=load(Queue)["Parameterlist"]
NPmax = amax(QueueData[:Ndp,-1])
Mem_max = (NPmax/100)**2 * 50 # max memory requirement in MB
Mem_max = int(Mem_max+1)
Tmax = 1.3*(NPmax/100)**4     # Max time requirement in cpu-seconds (update when I know better the speed of Caltech clusters)

Tmax_m = int(Tmax//60 + 5)
Tmax_h = Tmax_m//60 
Tmax_s = int(Tmax % 60+1)
#Tmax_m = Tmax_0 + 30


def fz(x):
    st = str(x)
    if len(st)==1:
        return "0"+st
    else:
        return st
    
mem_str= f"mem={Mem_max}mb"
nodestr = "nodes=1:ppn=1"
timestr = f"cput={fz(Tmax_h)}:{fz(Tmax_m)}:{fz(Tmax_s)}"

resource_str=f"{mem_str},{nodestr},{timestr}"

outdir = "/home/frederik_nathan/Weyl/Out/"
    
#r




#lgscript = "python3 -c;

ID0 = Queue[10:-4]+"_"+B.ID_gen()[:11]
for n in range(0,N_runs):
##    cmd = f"./weyl_pbs_script.sh {Queue} {Serieslength}"
##    print(cmd)
##    os.system(cmd)    
#    
#    resource_str="mem=2600mb,nodes=1:ppn=1,cput=20:00:00"  # Request 2.6 GB, 4 cores per process. 20 hours of cputime. Nic
#
#
    prestr = Q.get_prestr(n)
    logfile = Q.get_logfile(Queue)
    
    job_name = ID0+"_"+str(n)
    output_path = outdir+job_name+".out"
    err_path    = outdir+job_name+".err"
    script = Q.get_launch_command(Queue,Serieslength,logfile,prestr)
    script = "python3 -c 'print(\"hej\")'"
    
    qsub_str=f'cd /home/frederik_nathan/Test;qsub -l {resource_str} -N {job_name} -o {output_path} -e {err_path} {script}'
#        
#        
#        # l: resuorce requirements 
#         N: Job Name
#         q: Name of Que to be submitted to 
#         o: Path where stdout goes
#         e: Path where stderr goes
#         F: Argument passed to script
#    

#
#
#
##ResourceStr="mem=2600mb,nodes=1:ppn=4,cput=20:00:00,nice=15"  # Request 2.6 GB, 4 cores per process. 20 hours of cputime. Nic
#
#OutPath="/storage/ph_lindner/frederik/AFAI/Output/Out/"
#ErrPath="/storage/ph_lindner/frederik/AFAI/Output/Err/"
#QueName="N"
#JobBaseName="DiagSmall_%dx%d_%d_"%(Lx,Ly,Np)
#Script="LaunchDiagonalizationSmall_v1.sh"
#def argstr(List):
#    string=""
#    for l in List:
#        string+=str(l)+" "
#    
#    return string
#
#
#
#Command  = f"python3 -c \"from Run_master_equation_v5 import *;{LogStr}readfromqueue(\'{Queue}\',{Serieslength})\">>/dev/null &"
#
#ParameterList=[]
#for Int in IntList:
#    ParameterList.append([W,Int,DeltaT,Ntrotter,Mult])
#
#for parameters in ParameterList:
#        W=parameters[0]
#        V=parameters[1]
#        JobName=JobBaseName+"W=%d_V=%d" %(int(1000*W),int(1000*V))+"_"+timestring
#        RunParameters=Dims+parameters
#        ArgString=argstr(RunParameters)
##        
##        
#        print("    Running with parameters "+str(RunParameters))
#        print("    Job ID: "+str(JobName))
#        print("")
#        time.sleep(0.01)



