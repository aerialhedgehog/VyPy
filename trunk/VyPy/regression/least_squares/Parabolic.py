
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import sys, copy, weakref,gc

import numpy as np
import scipy as sp
import scipy.linalg

from VyPy.exceptions import EvaluationFailure
from VyPy.data import IndexableDict
from VyPy.tools import vector_distance, atleast_2d_row

from Modeling import Modeling


# ----------------------------------------------------------------------
#   Quadratic Regression Model
# ----------------------------------------------------------------------

class Parabolic(Modeling):
    
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
                f = c + b_i*x_i + a_j*x_j^2
            
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
        ay2 = X**2.
        Ay  = np.hstack([ay0,ay1,ay2])
        
        At = Ay
        Yt = Y
        
        # gradients
        if NDY:
            ad0 = np.array([[0.]]*NX*ND)
            ad1 = np.tile( np.eye(ND) , [NX,1] )
            ad2 = 2. * ad1 * np.repeat( X , ND, axis=0 )
        
            Ad = np.hstack([ad0,ad1,ad2])
            
            At = np.vstack([At,Ad])
            Yt = np.vstack([Yt, np.reshape(DY,[-1,1])])
        
        print 'Least Squares Solve ...'
        B = sp.linalg.lstsq(At,Yt,1e-12)[0]        

        c = B[0,0]
        b = B[1:ND+1]
        A = np.eye(ND) * B[ND+1:]
        
        self.c = c
        self.b = b
        self.A = A
        
        print ''
        
    
    def predict(self,XI):
        
        NX ,ND = X.shape
        NDY,_  = DY.shape        
        
        a0 = np.array([[1.]]*NX)
        a1 = XI
        a2 = XI**2.
        A  = np.hstack([a0,a1,a2])
        
        b0 = self.c
        b1 = self.b
        b2 = np.diag(self.A)[:,None]
        B  = np.vstack([b0,b1,b2])
        
        YI = np.dot(A,B)      
        
        # TODO: DYI
        
        return YI
        

# ----------------------------------------------------------------------
#   Unit Test
# ----------------------------------------------------------------------
        
if __name__ == '__main__':
    
    from VyPy.regression.gpr.training import Training
    from VyPy.sampling import lhc_uniform

    # test function
    def parabolic_function(X):
        X = X - 2.
        D = X.shape[1]
        C = np.ones([1,D]) * 10
        #C = np.array([ np.arange(D)+1. ])
        Y  = np.dot( X**2. , C.T  ) - 10.
        DY = 2.*X*C
        return Y,DY    
    
    # Design of Experiments
    ND = 18
    NX = 100
    XB = np.array([[-1.,1.]]*ND)
    X = lhc_uniform(XB,NX,maxits=5)
                      
    # Sample the function
    Y,DY = parabolic_function(X)
    
    # Build the Model
    train = Training(XB,X,Y,DY)
    model = Parabolic(train)
    model.learn()
    
    # Training Error
    YI = model.predict(X)
    print 'Training Error:' , np.sqrt( np.sum( (Y-YI)**2 ) )
    print ''
    
    # Testing Error
    NX = 100
    XI = lhc_uniform(XB,NX)  
    YT,DYT = parabolic_function(XI)
    YI = model.predict(XI)
    print 'Testing Error:' , np.sqrt( np.sum( (YT-YI)**2 ) )
    print ''
    
    # Optimize?
    try:    import cvxopt
    except: pass
    else:
        # Optimize
        q = cvxopt.matrix(model.b)
        P = cvxopt.matrix(model.A)*2.    
        r = cvxopt.solvers.qp(P,q)
        x = np.array(r['x']).T    
        print x
        print parabolic_function(x)[0]    