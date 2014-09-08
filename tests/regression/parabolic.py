
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------


import sys, copy, weakref,gc
import numpy as np

from VyPy.regression.least_squares import Parabolic
from VyPy.regression.gpr.training import Training

from VyPy.tools import vector_distance, atleast_2d_row
from VyPy.sampling import lhc_uniform


# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():
    
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
    
    #NX = 4    
    #X = np.array([[1.,2,1],
                  #[3,4,3],
                  #[5,6,5],
                  #[7,8,7]])    
                      
    # Sample the function
    Y,DY = parabolic_function(X)
    
    # Build the Model
    train = Training(XB,X,Y,DY)
    model = Parabolic(train)
    model.learn()
    
    # Training Error
    YI = model.predict(X)
    EI = np.sqrt( np.sum( (Y-YI)**2 ) )
    print 'Training Error:' , EI
    print ''
    
    # Testing Error
    NX = 100
    XI = lhc_uniform(XB,NX)  
    YT,DYT = parabolic_function(XI)
    YI = model.predict(XI)
    ET = np.sqrt( np.sum( (YT-YI)**2 ) )
    print 'Testing Error:' , ET
    print ''
    
    assert EI < 1e-6
    assert ET < 1e-6
    
    # Optimize?
    try:    import cvxopt
    except: pass
    else:
        # Optimize
        P = cvxopt.matrix(model.A)
        q = cvxopt.matrix(model.b)
        r = cvxopt.solvers.qp(P,q)
        x = np.array(r['x']).T    
        
        print 'Optimum:'
        print x
        print parabolic_function(x)[0]    
        
        EO = np.sqrt( np.sum( (x-2.)**2 ) )
        assert EO < 1e-6
        
        
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

if __name__ == '__main__':
    main()
