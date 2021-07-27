#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:11:53 2020

@author: frederik
"""
import os 

NUM_THREADS = 1

os.environ["OMP_NUM_THREADS"]        = str(NUM_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(NUM_THREADS)
os.environ["NUMEXPR_NUM_THREADS"]    = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"]        = str(NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"]   = str(NUM_THREADS)

import scipy.sparse as sp
from scipy import * 
from scipy.linalg import *
import scipy.sparse.linalg as spla
import time as time 
import numpy.random as npr 
import basic as B
import test_objects as to
import sparse_dot_mkl as sdmkl



def get_matrix_lists(H,D,block=None):
    """
    Extract hlist, J, Jp from matrix H, assuming D block structure. 
    """
    
    if type(block)==int:
        block = (block)
     
    DH = shape(H)[0]
    
    NB = int(DH/D+0.1)
    
    assert abs(D*NB - DH)<1e-7, "D is not a divisor of the matrix dimension"
    
    
    
    block_indices = arange(DH)//D

    if block==None:
        
        hlist = array([H[block_indices==n,:][:,block_indices==n] for n in range(0,NB)])
        
        
    Jp = H[block_indices==(1),:][:,block_indices==0]
    J  = H[block_indices==(0),:][:,block_indices==(1)]
    
    return hlist,J,Jp





class rgf_solver():
    """ 
    Recursive Green's functionsolver of the problem 
    
    u = H^{-1}v, 
    
    where H is of the block form 
                              
            |H0  J   0   0   0  ..  |
            |Jp  H1  J   0   0  ..  |
    H   =   |0   Jp  H2  J   0  ..  |
            |0   0   Jp  H3  0  ..  |
            |:   :   :   :   :      |       
      
    Here Hn = Hlist[n].
    
    Hlist       List of NB DxD matrices
    
    J, Jp:      DxD matrices (to be solved wth)


    Main method: 
        
    ulist, nlist = self(v,nb): 
        
    input
    
    v:          D*NB vector (i.e. array of dimension (D*NB))
    nb:         int, between 0 and D*NB
    
    returns:
    
    ulist = (u[n_1],..u[n_2])
    
    nlist = array([n_1,..,n_2])
    
    Here u[n] gives u in the nth block of u, where  u = H^-1 v. 
    
    Moreover, n_1 = nb - nqr andn_2 = nb + nqr, where nqr is the dynamical qr 
    precision parameter (always 1 or greater), i.e. the number of steps the 
    iteration can be trusted without doing a qr decomposition.
    
    
    """
    
    def __init__(self,hlist,J,Jp):
        
        self.D      = shape(J)[0]
        self.NB     = len(hlist)  
        self.DH     = self.D*self.NB
               
        self.hlist = hlist
        self.J     = sp.csc_matrix(J)
        self.Jp    = sp.csc_matrix(Jp)
  
        self.alpha = zeros((self.D,self.D),dtype=complex)#*self.glist[0]
        self.beta  = ascontiguousarray(zeros((self.D,self.D),dtype=complex))
        
        self.I     = eye(self.D,dtype=complex)
        
        self.psi_l = zeros((self.D),dtype=complex)
        self.psi_r = zeros((self.D),dtype=complex)
        
        self.Gamma = zeros((self.D,self.D),dtype=complex)
        
        
    def __rightcall__(self,vector,k):
        """
        main method (see above).
        Compute <vector | G | k+1>
        
        I.e. compute 
        """
        if k<0:
            raise ValueError("k must be zero or greater")
        
        if k>= self.NB:
            raise ValueError(f"k = {k} is too large")
        def get_vector_segment(n):
            ### Returns \langle \psi_{n+1}|
            return vector[n*self.D:(n+1)*self.D]
        
        

        ### Left sweep, to obtain \alpha _{k},\langle \psi_{k}^L
        for n in range(0,k):
            ### update alpha, such that self.alpha = \alpha_{n+1} (c.f. pdf notes)
            self.A1 = self.Jp@self.alpha
            self.A2 = self.A1@self.J
            self.A3 = self.hlist[n]-self.A2
            self.alpha = inv(self.A3,check_finite=False,overwrite_a=True)

                
            ### update psi_l, such that self.psi_l = \langle\psi^L_{n+1}| (c.f. pdf notes)
            self.psi_l = -(self.psi_l + get_vector_segment(n))@self.alpha@self.J

                
       
        ### Right sweep, to obtain \beta _{k+2},\langle \psi_{k+2}^R
        for n in range(self.NB-1,k,-1):
                
            ### update beta, such that self.beta = \beta{n+1} (c.f. pdf notes)
            self.A1 = self.J@self.beta
            self.A2 = self.A1 @ self.Jp
            self.A3 = self.hlist[n] - self.A2
            
            self.beta = (inv(self.A3,overwrite_a=True,check_finite=False))#,overwrite_a=True,overwrite_b=True,check_finite=False))

            ### update psi_l, such that self.psi_l = \langle\psi^L_{n+1}| (c.f. pdf notes)
            self.psi_r = -(self.psi_r + get_vector_segment(n))@self.beta@self.Jp

            
        ### Compute \Gamma_{k+1}.
        self.Gamma = inv(self.hlist[k]-(self.Jp@self.alpha@self.J + self.J@self.beta @ self.Jp))
        
        
        # glist[k] = \gamma _{k+1}
        return (self.psi_l + self.psi_r + get_vector_segment(k))@self.Gamma
    
    def __leftcall__(self,vector,k):
        
        vector = vector.conj()
        Jp_old = 1*self.Jp
        
        self.Jp = self.J.conj().T
        self.J = Jp_old.conj().T
        
        self.hlist = [x.conj().T for x in self.hlist]
        
        
        out = self.__rightcall__(vector,k)
        
        self.hlist = [x.conj().T for x in self.hlist]
        Jp_old = 1*self.Jp
        
        self.Jp = self.J.conj().T
        self.J = Jp_old.conj().T
        
        return out.conj()
    
    def __call__(self,vector,k,mode="l"):
        if mode =="l":
            return self.__leftcall__(vector,k)
        elif mode=="r":
            return self.__rightcall__(vector,k)
        else:
            raise ValueError("'mode' must be either 'l' or 'r'")


if __name__=="__main__":
        
    
    from matplotlib.pyplot import *
    print("Testing module")
    for q in range(0,10):
        N= 2+npr.randint(30);D=1+npr.randint(30)
        k=npr.randint(N)
    #    N=5;D=5
    #    k= 3
        H0 = to.test_hamiltonian(N,D,singular=True)
        #H0 = real(H0)
        hlist,J,Jp = get_matrix_lists(H0,D)
        
        V = npr.rand(D*N)+ 1j*npr.rand(D*N)
        #V[2]=1
        
        t0 = time.time()
        S = rgf_solver(hlist,J,Jp)
        t1=time.time()
        Q  = S(V,k,mode="l")
        t2 = time.time()
        
        H0 = H0.toarray()
        
        
        
        G = inv(H0)
        t3 = time.time()
        Ans = (G@V)[k*D:(k+1)*D]
        
        Dif = norm(Q-Ans)
        
        if Dif > 1e-10:
        
#            print("-"*80)
#            print(f"Difference of computed vs. exact result :    {norm(Q-Ans)}")
#            print("="*80)
#            print("")
#            print(f"    Time spent constructing rgf solver  : {t1-t0:.4} s")
#            print(f"    Time spent evaluating rgf solver    : {t2-t1:.4} s")
#            print(f"    Time spent using inv                : {t3-t2:.4} s")
#        
        
            raise AssertionError(f"Test failed. Difference between computed and exact result : {Dif}")


    
    print("Test proceeded ok")
    
    













