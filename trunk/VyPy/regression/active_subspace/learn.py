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
        
    return v


def gradient(DY,Y=None):
    
    DY = atleast_2d(DY)
    #Y = atleast_2d(Y)
    #M = 1. - ((Y-np.min(Y)) / (np.max(Y)-np.min(Y)))
    #M[M<0.5] = 0
    
    #X,Y,DY = scale_data(X,Y,DY)
    
    # m-samples, n-dimension
    m,n = DY.shape
    
    C = np.zeros([n,n])
    
    # emperical covariance matrix
    #for i in range(m):
        #dy = DY[None,i,:]
        ##dy = dy / np.linalg.norm(dy)  # this is needed for some reason
        ##dy = dy * M[i,0]
        #C += np.dot(dy.T,dy)
        #wait = 0
    #C = C/m
    
    ## eigenvalue problem, find principle directions
    #[d,W] = np.linalg.eig(C)
    #W = np.real(W)
    
    U, sig, W = np.linalg.svd(DY, full_matrices=False)
    d = (sig**2) / DY.shape[0]
    W = W.T
    W = W*np.sign(W[0,:])    
    
    ## normalize d
    #d = d / np.linalg.norm(d)
    #d = np.abs(d)
    
    # sort directions
    i = np.argsort(d)[::-1]
    d = d[i]
    W = W[:,i]
    
    return W,d
        

def bootstrap_ranges(DF, d, W, n_boot=1000):

    # set integers
    M, m = DF.shape
    k = d.shape[0]

    # bootstrap
    e_boot = np.zeros((k, n_boot))
    sub_dist = np.zeros((k-1, n_boot))
    ind = np.random.randint(M, size=(M, n_boot))

    # can i parallelize this?
    for i in range(n_boot):
        W0,e0 = gradient(DF[ind[:,i],:])
        e_boot[:,i] = e0[:k]
        for j in range(k-1):
            sub_dist[j,i] = np.linalg.norm(np.dot(W[:,:j+1].T, W0[:,j+1:]), ord=2)

    e_br = np.zeros((k, 2))
    sub_br = np.zeros((k, 3))
    for i in range(k):
        e_br[i,0] = np.amin(e_boot[i,:])
        e_br[i,1] = np.amax(e_boot[i,:])
    for i in range(k-1):
        sub_br[i,0] = np.amin(sub_dist[i,:])
        sub_br[i,1] = np.mean(sub_dist[i,:])
        sub_br[i,2] = np.amax(sub_dist[i,:])
        
    err_d = np.mean(np.abs(e_br-d[:,None]),axis=1)
    err_W = sub_br[:,1]# * d
    
    metric = np.zeros_like(d)
    for i in range(len(d)):
        metric[i] = err_W[i] * np.sqrt( np.sum(d[:i+1]) ) + np.sqrt( np.sum(d[i+1:]) )
    

    return err_d, err_W, metric        
        
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
        
    
    
    