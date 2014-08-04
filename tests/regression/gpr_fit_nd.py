

# ------------------------------------------------------------
#   Imports
# ------------------------------------------------------------

import time, os, sys, copy

import numpy as np
from numpy import pi

import VyPy
from VyPy.regression import gpr
from VyPy.tools import atleast_2d

from warnings import warn, simplefilter
simplefilter('error')


# ------------------------------------------------------------
#   Main 
# ------------------------------------------------------------

def main():
    
    # ---------------------------------------------------------
    #  Setup
    # ---------------------------------------------------------
    # Choose the function and bounds
    
    # the target function, defined at the end of this file
    The_Func = rosenbrock_function
    
    # hypercube bounds
    ND = 6 # dimensions
    XB = np.array( [[-2.,2.]]*ND )
    
    # ---------------------------------------------------------
    #  Training Data
    # ---------------------------------------------------------
    # Select training data randomly with Latin Hypercube Sampling
    
    # number of samples
    ns = 80 
    
    # perform sampling with latin hypercube
    XS = VyPy.sampling.lhc_uniform(XB,ns)
    
    # evaluate function and gradients
    FS,DFS = The_Func(XS)
    
    
    # ---------------------------------------------------------
    #  Machine Learning
    # ---------------------------------------------------------
    # Build a gpr modeling object and train it with data
    
    # this class factory sets up up all the data structures
    Model = gpr.library.Gaussian(XB,XS,FS,DFS)
    
    # pull function handles for plotting and evaluating
    g_x = Model.predict_YI          # the surrogate
    f_x = lambda(Z): The_Func(Z)[0] # the truth function
    
    
    # ---------------------------------------------------------
    #  Evaluate a Testing Set
    # ---------------------------------------------------------      
    # Run a test sample on the functions
    
    nt = 200 # number of test samples
    XT = VyPy.sampling.lhc_uniform(XB,nt)
    
    # functions at training data locations
    FSI = g_x(XS)
    FST = f_x(XS)
    
    # functions at grid testing locations
    FTI = g_x(XT)
    FTT = f_x(XT)
    
    
    # ---------------------------------------------------------
    #  Model Errors
    # ---------------------------------------------------------
    # estimate the rms training and testing errors
    
    print 'Estimate Modeling Errors ...'
    
    # the scaling object
    Scaling = Model.Scaling
    
    # scale data - training samples
    FSI_scl = Scaling.Y.set_scaling(FSI)
    FST_scl = Scaling.Y.set_scaling(FST)
    
    # scale data - grid testing samples
    FTI_scl = FTI / Scaling.Y # alternate syntax
    FTT_scl = FTT / Scaling.Y
    
    # rms errors
    ES_rms = np.sqrt( np.mean( (FSI_scl-FST_scl)**2 ) )
    EI_rms = np.sqrt( np.mean( (FTI_scl-FTT_scl)**2 ) )
    
    print '  Training Error = %.3f%%' % (ES_rms*100.)
    print '  Testing Error  = %.3f%%' % (EI_rms*100.)
    
    assert ES_rms < 1e-4
    assert EI_rms < 1e-2
    
    # Done!
    return
    
#: def main()


# -------------------------------------------------------------
#  Test Function - Rosenbrock
# -------------------------------------------------------------

def rosenbrock_function(X):
    X = atleast_2d(X)
    D = X.shape[1]
    Y = 0.
    DY = X*0.
    for I in range(D):
        if I < D-1:
            Y = Y + 100.*( X[:,I+1]-X[:,I]**2. )**2. + ( 1-X[:,I] )**2.
            DY[:,I] = DY[:,I] - 400.*( X[:,I+1]-X[:,I]**2. ) * X[:,I] - 2.*( 1.-X[:,I] )
        if I>0:
            DY[:,I] = DY[:,I] + 200.*( X[:,I]-X[:,I-1]**2. )
    Y = atleast_2d(Y,'col')
    return Y,DY


# -------------------------------------------------------------
#  Start Main
# -------------------------------------------------------------

if __name__=='__main__':
    main()


