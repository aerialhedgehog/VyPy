
import sys, copy, weakref,gc

import numpy as np
import scipy as sp

#import pyOpt

from IndexableBunch import IndexableBunch

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt


def LHC_uniform(XB,NI,XI=None,maxits=100):
    ''' Latin Hypercube Sampling with uniform density
        iterates to maximize minimum L2 distance
    '''
    
    print "Latin Hypercube Sampling ... "
    
    # dimension
    ND = XB.shape[0]
    
    # initial points to respect
    if XI is None:
        XI = np.empty([0,ND])
       
    # output points
    XO = []
    
    # initialize
    mindiff = 0;
    
    # maximize minimum distance
    for it in range(maxits):
        
        # samples
        S = np.zeros([NI,ND])
        
        # populate samples
        for i_d in range(ND):
            
            # uniform distribution [0,1], latin hypercube binning
            S[:,i_d] = ( np.random.random([1,NI]) + np.random.permutation(NI) ) / NI
            
        # scale to hypercube bounds
        XS = S*(XB[:,1]-XB[:,0]) + XB[:,0]        
        
        # add initial points
        XX = np.vstack([ XI , XS ])
        
        # calc distances
        vecdiff = vector_distance(XX)[0]
        
        # update
        if vecdiff > mindiff:
            mindiff = vecdiff
            XO = XX
        
    #: for iterate
    
    print 'Minimum Distance = %.4g' % mindiff
    
    return XO
    
def vector_distance(X,P=None):
    ''' calculates distance between points in matrix X 
        with each other, or optionally to given point P
        returns min, max and matrix/vector of distances
    '''
    
    # distance matrix among X
    if P is None:
        
        nK,nD = X.shape
        
        d = np.zeros([nK,nK,nD])
        for iD in range(nD):
            d[:,:,iD] = np.array([X[:,iD]])-np.array([X[:,iD]]).T
        D = np.sqrt( np.sum( d**2 , 2 ) )
        
        diag_inf = np.diag( np.ones([nK])*np.inf )
        dmin = np.min(np.min( D + diag_inf ))
        dmax = np.max(np.max( D ))
        
    # distance vector to P
    else:
        P = check_array(P,'row')
        assert P.shape[0] == 1 , 'P must be a horizontal vector'
        D = np.array([ np.sqrt( np.sum( (X-P)**2 , 1 ) ) ]).T
        dmin = D.min()
        dmax = D.max()
        
    return (dmin,dmax,D)


def check_list(val):
    if not isinstance(val,list): val = [val]
    return val

def check_array(A,oned_as='row'):
    ''' ensures A is an array and at least of rank 2
    '''
    if not isinstance(A,np.ndarray):
        A = np.array(A)
    if np.rank(A) < 2:
        A = np.array(np.matrix(A))
        if oned_as == 'row':
            pass
        elif oned_as == 'col':
            A = A.T
        else:
            raise Exception , "oned_as must be 'row' or 'col' "
            
    return A


def plot_spider_axis(FIG,X0,XB):

    X0 = check_array(X0,'row')
    XB = check_array(XB,'row')
    
    n_d = X0.shape[1]
    
    # thetas
    TP = np.linspace(0,np.pi,n_d+1)
    
    # axis
    AX = FIG.gca(projection='3d')
    
    # each dimension
    for i_d in range(n_d):
        
        # this axis coordinates
        Xi = [XB[i_d,0],XB[i_d,1]]
        
        # rotate into plot axis
        XP = ( Xi-X0[0,i_d] )* ( np.cos(TP[i_d]) )
        YP = ( Xi-X0[0,i_d] )* ( np.sin(TP[i_d]) )
        
        # plot axis line
        AX.plot( [ XP[0] , XP[1] ] ,
                 [ YP[0] , YP[1] ] ,
                 [ 0.    , 0.     ] , 'k-' )
        # plot axis ends
        AX.plot([XP[0]],[YP[0]],[0],'ko',mfc='none',mew=1.5)
        AX.plot([XP[-1]],[YP[-1]],[0],'ko')
        # plot axis labels
        AX.text(XP[0] ,YP[0] ,0.0,'X%i_L'%(i_d+1),fontsize=10)
        AX.text(XP[-1],YP[-1],0.0,'X%i_U'%(i_d+1),fontsize=10)
                
    #: for each dimension
    
    # disable x and y axis tick labels
    AX.xaxis.set_ticklabels([])
    AX.yaxis.set_ticklabels([])
    
    return AX

def plot_spider_trace(AX,FF,X0,XB,NP,*plt_args,**plt_kwarg):
    ''' plots a multidimensional spider with function 
        FF, centered at X0, in bounds XB, on plot axis AX
    '''
    
    X0 = check_array(X0,'row')
    XB = check_array(XB,'row')
    
    n_d = X0.shape[1]
    n_x = NP
    
    # thetas
    TP = np.linspace(0,np.pi,n_d+1)
        
    # each dimension
    for i_d in range(n_d):
        
        # this axis coordinates
        Xi = np.linspace(XB[i_d,0],XB[i_d,1],n_x)
        XX = np.ones([n_x,n_d]) * X0
        XX[:,i_d] = Xi
        
        # rotate into plot axis
        XP = ( Xi-X0[0,i_d] )* ( np.cos(TP[i_d]) )
        YP = ( Xi-X0[0,i_d] )* ( np.sin(TP[i_d]) )

        # evaluate function
        ZP = FF(XX)[:,0]
        
        # plot function
        AX.plot(XP,YP,ZP,*plt_args,**plt_kwarg)
        
        if i_d == 0 and plt_kwarg.has_key('label'):
            del plt_kwarg['label']
        
    #: for each dimension
    
    # disable x and y axis tick labels
    AX.xaxis.set_ticklabels([])
    AX.yaxis.set_ticklabels([])
    
    # for legend
    P = AX.plot([0.],[0.],[np.nan],*plt_args,**plt_kwarg)
    
    return P

class functional_bunch(object):
    def __init__(self,func,key):
        self.func = func
        self.key  = key
    def __call__(self,*args,**kwarg):
        return self.func(*args,**kwarg)[self.key]
        
class pyopt_objective(object):
    def __init__(self,func,cons):
        self.func = func
        self.cons = cons
    def __call__(self,*args,**kwarg):
        try:
            F = self.func(*args,**kwarg)
            C = self.cons(*args,**kwarg)
            fail = 0
        except Evaluation_Failure:
            F = 1.e20
            C = []
            fail = 1
        return F,C,fail
    
class Evaluation_Failure(Exception):
    pass