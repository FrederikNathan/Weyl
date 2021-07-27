#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:31:50 2020

@author: frederik
"""
import os 

NUM_THREADS = 2

os.environ["OMP_NUM_THREADS"]        = str(NUM_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(NUM_THREADS)
os.environ["NUMEXPR_NUM_THREADS"]    = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"]        = str(NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"]   = str(NUM_THREADS)

import socket

from scipy import *
from scipy.linalg import * 
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy import * 

import numpy.random as npr
import scipy.optimize as optimize
import basic as B
from Units import * 
import master_equation_multimode_v9 as MA
import time 

QueueDir = "../Queues/"

TIME_PER_RUN_MINUTES = 45


HN = socket.gethostname()
if HN=="Frederiks-MacBook-Pro-2.local":
    HN = "mac"
elif HN=="wakanda" or HN== "yukawa":
    HN = "yu"
else: #HN=="cmtrack2.caltech.edu":
    HN="ca"
#else:
#    raise SystemError(f"Host {HN} not recognized")
    
    
def add_to_queue(parameterlist,klist,Name=None):
    
    if Name==None:
        
        print("Enter name of queue:")
        QueueName = input()+"_"+HN
    
    else:
        QueueName = Name+"_"+HN
        
    QueuePath=QueueDir+QueueName+".npz"

    NK= shape(klist)[0]
    
    if os.path.exists(QueuePath):
        D=load(QueuePath)
        PL = D["Parameterlist"]
        KL = D["Klist"]
        SV = D["Status"]
        
        N0 = sum(SV==0)
        N1 = sum(SV==1)
        N2 = sum(SV==2)
        print(f"\nQueue {QueueName} already exists, with {N0+N1+N2} runs. ({N0} waiting, {N1} active, {N2} finished). \n\nAdd {NK} runs to queue? (y/n)\n")
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



def is_queue_done(Queue):
    QueueData = load(Queue)
    Status = QueueData["Status"]
    
    I = where(Status==0)[0]

    if len(I)==0:
        return 1
    else:
        return 0
    
def readfromqueue(Queue,Serieslength):
    time.sleep(npr.rand()*0.2)
    QueueData = load(Queue)
    Status = QueueData["Status"]
    Klist  = QueueData["Klist"]
    Parameterlist = QueueData["Parameterlist"]
    try:
        
        I=where(Status==0)[0][:Serieslength]
    except IndexError:
        assert sum(Status==0)<=Serieslength
        
        I = where(Status==0)[0]


    if len(I)==0:
        print("Queue empty. finishing")
        Main.EXIT = True 
        
    
        
    Status[I]=1
    savez(Queue,Status=Status,Klist=Klist,Parameterlist=Parameterlist)
            
            
    klist = Klist[I,:]
    parameterlist = Parameterlist[I,:]
    

    QueueName = Queue[10:-4]
    MA.main_run(klist,parameterlist,savetime=inf,display_progress=1,PreStr=QueueName)
    
    QueueData = load(Queue)
    Status = QueueData["Status"]
    Klist  = QueueData["Klist"]
    Parameterlist = QueueData["Parameterlist"]
    
    Status[I]=2
    savez(Queue,Status=Status,Klist=Klist,Parameterlist=Parameterlist)
    
    return 1

def make_safety_copy(Queue):
    QueueData = load(Queue)
    Status = QueueData["Status"]
    Klist  = QueueData["Klist"]
    Parameterlist = QueueData["Parameterlist"]
    Queue_sc= Queue[:-4]+"_sc.npz"
#    print(Queue_sc)
    savez(Queue_sc,Status=Status,Klist=Klist,Parameterlist=Parameterlist)
    

        
def print_queue_status(Queue):
    QueueData = load(Queue)
    SV = QueueData["Status"]
    
    N0 = sum(SV==0)
    N1 = sum(SV==1)
    N2 = sum(SV==2)
    try:
        
        n1 = min(amin(where(SV==1)[0]),amin(where(SV==0)[0]))
        n2 = min(n1+1000,len(SV))
    except ValueError:
        n1=0
        n2 = 1000
        
    print(f"\nStatus of queue {Queue[10:-4]}: {N0+N1+N2} runs ({N0} waiting, {N1} active, {N2} finished).")
    print("")
    print(f"Status of jobs {n1}-{n2}:")
    print("")
    print(SV[n1:n2])
    print("")
    
    
    Parameterlist = QueueData["Parameterlist"]
    omega1,omega2,tau,vF,V0x,V0y,V0z,EF1,EF2,Mu,Temp = Parameterlist[n1,:]
    
    print("="*80)
    print(f"Parameters of run {n1}:")
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
    print("-"*80)
    
#    print(f"    Npmax    :  {NPmax}")


    
    
def get_launch_command(Queue,Serieslength,logfile,prestr,no_output=True):
    
    LogStr =f"B.redirect_output(\'{logfile}\',prestr=\'{prestr}\',timestamp=True); "

    Command  = f"python3 -c \"from weyl_queue import *;{LogStr}readfromqueue(\'{Queue}\',{Serieslength})\""
    if no_output:
        Command = Command + ">>/dev/null &"
    else:
        Command = Command + " &"
    
    return Command
  
def get_prestr(NP):
    return f"run {NP:>3} : "

def get_logfile(Queue):
    return "../Logs/"+Queue[10:]+".log"
    
def wakanda_launch(Queue,N_parallel,Serieslength,TimeOut=3600):
    B.tic(n=37)
    SC_time = 7200*(0.5+0.5*rand())
    
    logfile = get_logfile(Queue)
    ID0 = B.ID_gen()[:11]

    prestr = ID0+" "+f"main    : "
    B.tic(n=42)       
    B.tic(n=43)
    B.redirect_output(logfile,prestr=prestr,timestamp=True)    
    NP= 0 
    print("")
    print("="*80)
    print("   Launching new series of runs")
    print("="*80)
    print("")
    

    while True:
    

        try:
      
            
            if B.toc(n=42)>TimeOut:
                print("="*80)
                print("Timeout reached. Terminating")
                print("="*80)
                break
                  
            if B.GetNumberOfMyPythonProcesses()<N_parallel:
                
                
                prestr = ID0+" "+get_prestr(NP)
                Command = get_launch_command(Queue,Serieslength,logfile,prestr)
                os.system(Command)
                
                print("")
                print("Launching new series")
                print("")
                time.sleep(10*(0.5+npr.rand()))
                NP+=1 
            else:
                time.sleep(60*(0.5+npr.rand()))
    
                        
            print(f"Current number of processes: {B.GetNumberOfMyPythonProcesses()}")
        
            
        except:
            time.sleep(60)
        
            
        if B.toc(n=43,disp=0)>1800:
            B.tic(n=43)
            try:
                
                QD= is_queue_done(Queue)
                if QD:
                    print("="*80)
                    print("Queue done. Terminating")
                    print("="*80)
                    break
            except:
                time.sleep(60*(0.5+npr.rand()))
        
        if B.toc(n=37,disp=0)>SC_time:
            print("="*80)
            print("Making safety copy of queue")
            print("="*80)
        
            make_safety_copy(Queue)
            B.tic(n=37)
            SC_time = 7200*(0.5+0.5*rand())


            
            
    print("="*80)
    print("="*80)
    print("Launcher done")    
    print("="*80)
    print("="*80)
    
def caltech_launch(Queue,N_runs,Serieslength):
    Ndp = int(N_runs*Serieslength+1)
    ID_series = B.ID_gen()[:11]    
    
    QueueData=load(Queue)["Parameterlist"]
    Mem_max = 800
    Tmax    = Serieslength * 60*TIME_PER_RUN_MINUTES  # Assign 20 minutes to each run
    
        
    Tmax_s = int(Tmax % 60+1)
    Tmax_m = int(Tmax/60 + 5)%60
    Tmax_h = int(Tmax/3600)
    
    #Tmax_m = Tmax_0 + 30
    
    
    def fz(x):
        st = str(x)
        if len(st)==1:
            return "0"+st
        else:
            return st
        
    Tmax_h,Tmax_m,Tmax_s = [fz(x) for x in [Tmax_h,Tmax_m,Tmax_s]]
        
    mem_str= f"mem={Mem_max}mb"
    nodestr = "nodes=1:ppn=1"
    walltime_str = f"{fz(Tmax_h)}:{fz(Tmax_m)}:{fz(Tmax_s)}"
    
#    resource_str=f"{mem_str},{nodestr},{timestr}"
    
    outdir = "/home/frederik_nathan/out/"
        
    strlist=[]
    
    ID0 = Queue[10:-4]+"_"+B.ID_gen()[:11]
    for n in range(0,N_runs):
    
        prestr = ID_series+" "+get_prestr(n)
        logfile = get_logfile(Queue)
        
        job_name = ID0+"_"+str(n)
        output_path = outdir+job_name+".out"
        err_path    = outdir+job_name+".err"
#        command = get_launch_command(Queue,Serieslength,logfile,prestr)
#        command = "python3 -c \"print(\'hej\')\""
        
        
#        JOBNAME=$1
#        WALLTIME=$2
#        LOGFILE=$3
#        PRESTR=$4
#        QUEUE=$5
#        $SERIESLENGTH=$6

        script  = "pbs_script.sh"
        argstring = f"LF={logfile},PRESTR='{prestr}',Q={Queue},SL={Serieslength}"
        qsub_str=f'qsub -N {job_name} -l walltime={Tmax_h}:{Tmax_m}:{Tmax_s} -v {argstring} {script} &'
        os.system(qsub_str)
        time.sleep(0.2*(1+npr.rand()))
#        e=03:33:33
#        os.system(f"./{script} {argstring}")
#        print(qsub_str)
#        strlist.append(qsub_str)
        
#    return strlist

  
def slurm_launch(Queue,N_runs,Serieslength):
    Ndp = int(N_runs*Serieslength+1)
    ID_series = B.ID_gen()[:11]    
    
    QueueData=load(Queue)["Parameterlist"]
    Mem_max = 800
    Tmax    = Serieslength * 60*TIME_PER_RUN_MINUTES  # Assign 20 minutes to each run
    
        
    Tmax_s = int(Tmax % 60+1)
    Tmax_m = int(Tmax/60 + 5)%60
    Tmax_h = int(Tmax/3600)
    
    #Tmax_m = Tmax_0 + 30
    
    
    def fz(x):
        st = str(x)
        if len(st)==1:
            return "0"+st
        else:
            return st
        
    Tmax_h,Tmax_m,Tmax_s = [fz(x) for x in [Tmax_h,Tmax_m,Tmax_s]]
        
    mem_str= f"mem={Mem_max}mb"
    nodestr = "nodes=1:ppn=1"
    walltime_str = f"{fz(Tmax_h)}:{fz(Tmax_m)}:{fz(Tmax_s)}"
    
#    resource_str=f"{mem_str},{nodestr},{timestr}"
    
    outdir = "/home/frederik_nathan/out/"
        
    strlist=[]
    
    ID0 = Queue[10:-4]+"_"+B.ID_gen()[:11]
    for n in range(0,N_runs):
    
        prestr = ID_series+" "+get_prestr(n)
        logfile = get_logfile(Queue)
        
        job_name = ID0+"_"+str(n)
        output_path = outdir+job_name+".out"
        err_path    = outdir+job_name+".err"

        script  = "pbs_script.sh"
        argstring = f"LF={logfile},PRESTR='{prestr}',Q={Queue},SL={Serieslength}"
        qsub_str=f'qsub -N {job_name} -l walltime={Tmax_h}:{Tmax_m}:{Tmax_s} -v {argstring} {script} &'
        os.system(qsub_str)
        time.sleep(0.2*(1+npr.rand()))
#        e=03:33:33
#        os.system(f"./{script} {argstring}")
#        print(qsub_str)
#        strlist.append(qsub_str)
        
#    return strlist

