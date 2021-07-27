#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:36:07 2020

@author: frederik
"""

from numpy import *
from scipy import *
from scipy.integrate import *
from scipy.linalg import *
import time
from numpy.fft import * 
import os 
import sys
import datetime as datetime 
from scipy.special import factorial as factorial
#builtins.print as pp

main_module=  sys.modules["__main__"]
from builtins import print as old_print

SX = array(([[0,1],[1,0]]),dtype=complex)
SY = array(([[0,-1j],[1j,0]]),dtype=complex)
SZ = array(([[1,0],[0,-1]]),dtype=complex)
I2 = array(([[1,0],[0,1]]),dtype=complex)

def print(x):
    old_print(x)
    

    

def tprint(n=None):
    print("    Done. Time spent: "+str(round(toc(n),4)))
    return

def get_ct():
    return time.time()

global T_tic 
T_tic= get_ct()
tic_list = ones(500)*get_ct()

def tic(n=None):
    global T_tic
    global tic_list
    
    if n==None:
        
        T_tic = get_ct()
    else:
        tic_list[n]=get_ct()
    
def toc(n=None,disp=True):
    
    if n == None:
        out = get_ct()-T_tic
    else:
        out = get_ct()-tic_list[n]
    
    if disp:
        print(f"    time spent: {out:.4} s")
    
    return out
    
    
def ID_gen():
    timestring=datetime.datetime.now().strftime("%y%m%d_%H%M-%S.%f")[:-3]
    
    return timestring 


def binom(n,k):
    """
    returns n choose k
    """
    
    return int(prod([x for x in range(n-k+1,n+1)])/factorial(k,exact=1)+0.1)

def time_stamp():
    return datetime.datetime.now().strftime("%d/%m %H:%M")

def redirect_output(File,prestr="",timestamp=False):  
    
    global old_stdout
    global old_stderr
        
    old_stdout = sys.stdout
    old_stderr = sys.stderr
 
#    def new_writer(text):
        

        
#    class 
            
    sys.stdout = open(File, 'a')
    sys.stderr = open(File, 'a')
    if timestamp==True:
           
        def new_write(text):
            prestring = time_stamp()+"  "+prestr 
            if text!="\n":
            
                open(File,'a').write(prestring+text)
            else:
                open(File,'a').write(text)
    else:    
        def new_write(text):
            prestring = prestr
            if text!="\n":
            
                open(File,'a').write(prestr+text)
            else:
                open(File,'a').write(text)
    sys.stdout.write = new_write
    sys.stderr.write = new_write
#    def sys.stderr.write(text):
#        
#        open(File,'a').write(prestr+text)
            
    
    

if sys.platform=="linux":
    """ 
    Define commands for getting CPU load and number of processes
    """
    def GetCPU():
        STR=os.popen('''sar 1 1|awk '{usage = $3} END {print usage}' ''').readline()
    
        """
        sar %a %b   :  monitor system for a seconds b times. Output $1=time, %2=Cpus ('all' in our case) %3=Usage, percent, ....
        
        
        | awk 'Command'   :  process output with following commands
        """
        
        
        try:
            
            D1 = float(STR[0:2])
            D2 = float(STR[3:5])*0.01

        except ValueError:
            D1 = float(STR[0:1])

            D2 = float(STR[2:4])*0.01
        
        return D1+D2
    
    
    def GetNumberOfMyPythonProcesses():

        STR = os.popen("ps auxwww|grep python|grep qxn|wc -l").readline()
        
        if STR[-1]=="\n":
            Out =int(STR[:-1])
            
        else:
            Out =int(STR)
        Out = Out -2
        
       
        return Out
                
    
else:
    import psutil
    
    def GetCPU():
        return psutil.cpu_percent()
    
    def GetNumberOfMyPythonProcesses():

        STR = os.popen("ps auxwww|grep python|grep Queue|wc -l").readline()
        
        
        if STR[-1]=="\n":
            Out =int(STR[:-1])
            
        else:
            Out =int(STR)
        
        Out=Out- 2 

        
        
       
        return Out
            
if __name__ == "__main__":
    redirect_output("test.log",prestr="dfsd")
    print("Hej")