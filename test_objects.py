from numpy import *
import numpy.random as npr
from time import *
#from numpy.linalg import *
import scipy.sparse as sp
import scipy.sparse.linalg as spla

### Make Hamiltonian
def test_hamiltonian(N,D,rand=True,singular= True):
    global J,Tmat,H0,T
    
    H0 = tuple(npr.rand(N,D,D))
    H0 = sp.block_diag(H0,format="csc")
    if rand==False:
        H0 =  tuple(2*ones((N,D,D)))
        H0 = 2*eye(N*D)
    T = sp.diags(ones(D*N),offsets =D,format="csc")
    T = T[:D*N,:D*N]
    
    J = T + T.T
    
    T = sp.diags(ones(N),offsets =1,format="csc")
    T = T[:N,:N]
    
    J = sp.eye(D,format="csc")
    J = J[::-1,:]
    
    if singular:
        
        J[D//2,:] = 0 
    
    Tmat = sp.kron(T,J)
    Tmat = Tmat + Tmat.conj().T
    H = 3*(H0+H0.T) + Tmat# + Tmat.T

    return H




#test_hamiltonian(5,1,rand=False)
    

    
def test(n):
    assert n<10 or n>20, "n is too large"
    print(n)
    
    assert n<20,"n is still too large"
    
def newtest(n):
    try:
        test(n)
    except AssertionError as err:# ,"n is too large":
        if err.args[0]=="n is too large":        
            print(n)    
        else: 
            raise AssertionError(err.args[0])
            
            
X = test_hamiltonian(14,3)