# SOURCE: http://arxiv.org/pdf/1304.2070v1.pdf

import numpy as np
import scipy as sp

from VyPy.tools import atleast_2d


def linear(X,Y):
    
    X = atleast_2d(X)
    Y = atleast_2d(Y,oned_as='col')
    
    #X,Y = scale_data(X,Y)
    
    m,n = X.shape
    
    A = np.hstack([ np.ones([m,1]) , X ])
    
    a = sp.linalg.lstsq(A,Y)[0]
    
    v = a[1:,:]
    v = v / sp.linalg.norm(v)
        
    v = v.T
    
    return v


def gradient(DY,k=None):
    
    DY = atleast_2d(DY)
    
    #X,Y,DY = scale_data(X,Y,DY)
    
    # m-samples, n-dimension
    m,n = DY.shape
    k = k or n
    
    C = np.zeros([n,n])
    
    # emperical covariance matrix
    for i in range(m):
        dy = DY[[i],:]
        dy = dy / np.linalg.norm(dy)  # this is needed for some reason
        C += np.dot(dy.T,dy)
    C = C/m
    
    # eigen value problem, find principle directions
    [d,V] = np.linalg.eig(C)
    
    # sort directions
    i = np.argsort(d)[::-1]
    d = d[i]
    V = V[:,i]
    
    V = np.real(V)
    
    if not k is None:
        V = V[:,0:k].T
    
    return V, d
        
        
def scale_data(X, Y, DY=None, n_std=2.0):
    
    mean_x = np.mean(X,axis=0)
    std_x  = np.std(X,axis=0)
    
    X = (X - mean_x[None,:]) / ( n_std * std_x[None,:] )
        
    mean_y = np.mean(Y,axis=0)
    std_y  = np.std(Y,axis=0)
    
    Y = (Y - mean_y[:,None]) / ( n_std * std_y[:,None] )
    
    if DY is None:
        return X,Y
    else:
        DY = DY / (std_y[:,None]/std_x[None,:])
        return X,Y,DY
        
    
    
    