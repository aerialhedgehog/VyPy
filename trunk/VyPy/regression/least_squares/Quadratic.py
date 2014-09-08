
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import numpy as np
import scipy as sp

from VyPy.exceptions import EvaluationFailure
from VyPy.tools import vector_distance, atleast_2d_row

from Modeling import Modeling


# ----------------------------------------------------------------------
#   Quadratic Regression Model
# ----------------------------------------------------------------------

class Quadratic(Modeling):
    
    def __init__(self,Train):
        
        self.Train = Train
        
        # coefficients
        self.A = None
        self.b = None
        self.c = None
        
        
    def learn(self):
        """
            regresses data with functions and gradients (if available)
            to this form:
                f = c + b_i*x_i + 0.5*A_i,j*x_i*x_j
            
        """
        
        # unpack
        X = self.Train.X
        Y = self.Train.Y
        DY = self.Train.DY
        
        NX ,ND = X.shape
        NDY,_  = DY.shape
        
        print 'Build Information Matricies ...'
        
        # functions
        ay0 = np.array([[1.]]*NX)
        ay1 = X
        ay2 = np.reshape( np.einsum('ij,ik->ijk',X,X) , [-1,ND*ND] )

        # reduce redundant basis variables
        i_doub = np.tri(ND,k=-1).T == 1
        ay2[:,i_doub.ravel()] = ay2[:,i_doub.ravel()] * 2.        
        i_keep = np.tri(ND,k=0).T == 1
        ay2 = ay2[:,i_keep.ravel()]

        # basis matrix, functions
        Ay  = np.hstack([ay0,ay1,ay2])
        
        # arrays for the least squares regression
        At = Ay
        Yt = Y
        
        # gradients
        if NDY:
            ad0 = np.array([[0.]]*NX*ND)
            
            ad1 = np.tile( np.eye(ND) , [NX,1]  )
            
            ad2a = np.repeat( np.eye(ND)[:,None,:] , ND , 1 )
            ad2a = np.reshape( ad2a , [-1,ND*ND] )    
            ad2a = np.repeat( ad2a, NX, axis=0 ) * np.repeat( np.tile( X, [ND,1] ) , ND, axis=1 )
            
            ad2b = np.repeat( np.eye(ND)[:,:,None] , ND , 2 )
            ad2b = np.reshape( ad2b , [-1,ND*ND] ) 
            ad2b = np.repeat( ad2b, NX, axis=0 ) * np.tile( np.tile( X, [ND,1] ) , [1,ND] )
            
            ad2 = ad2a + ad2b
            
            # reduce redundant bases
            ad2[:,i_doub.ravel()] = ad2[:,i_doub.ravel()] * 2.
            ad2 = ad2[:,i_keep.ravel()]            
            
            Ad = np.hstack([ad0,ad1,ad2])
            
            # add to arrays for least squares regression
            At = np.vstack([At,Ad])
            Yt = np.vstack([Yt, np.ravel(DY.T)[:,None]])
        
        print 'Least Squares Solve ...'
        B = sp.linalg.lstsq(At,Yt)[0]        
        
        # unpack data
        c = B[0,0]
        b = B[1:ND+1]
        
        A = np.zeros([ND,ND])
        A[i_keep]   = B[ND+1:,0]
        A[i_keep.T] = A.T[i_keep.T]
        
        # problem forumulation
        A = A*2.
        
        # store results
        self.c = c
        self.b = b
        self.A = A
        
        print ''
        
    
    def predict(self,XI):
        
        NX ,ND = XI.shape
        
        a0 = np.array([[1.]]*NX)
        a1 = XI
        a2 = np.reshape( np.einsum('ij,ik->ijk',XI,XI) , [-1,ND*ND] ) / 2.
        A  = np.hstack([a0,a1,a2])
        
        b0 = self.c
        b1 = self.b
        b2 = np.ravel(self.A)[:,None]
        B  = np.vstack([b0,b1,b2])
        
        YI = np.dot(A,B)      
        
        # TODO: DYI
        
        return YI
        
# ----------------------------------------------------------------------
#   Alternate way to build quadratic model, without gradients
# ----------------------------------------------------------------------

def quadratic_model(X,F):
    """ written by Paul Constantine in matlab
        translated by Trent Lukaczyk to python
    """
    
    from numpy import flipud, zeros, ones, prod, sum, arange
    from numpy.linalg import lstsq
    from VyPy.tools import index_set
    
    M,m = X.shape
    
    # coefficients
    I = flipud( index_set('full',2,m) )
    A = zeros([M,I.shape[1]])
    for i in range(I.shape[1]):
        ind = I[:,i,None]
        A[:,i] = prod( X ** ind.T , axis=1 )
    
    # solve    
    t = lstsq(A,F)[0]
    
    # unwrap
    be = t[1:m+1,:]
    Al = zeros([m,m])
    for i in range(m+1,I.shape[1]):
        ind = I[:,i]
        loc = arange(m)[ind != 0]
        if len(loc) == 1:
            Al[loc,loc] = 2*t[i]
        else:
            Al[loc[0],loc[1]] = t[i]
            Al[loc[1],loc[0]] = t[i]
    
    return be,Al


# ----------------------------------------------------------------------
#   Unit Test
# ----------------------------------------------------------------------
# See tests/regression/quadratic.py