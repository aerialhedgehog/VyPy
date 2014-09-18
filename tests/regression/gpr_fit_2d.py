
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
    XB = np.array([ [-2. , 2. ] ,   # [ lb_0 , ub_0 ]
                    [-2. , 2. ] ])  # [ lb_1 , ub_1 ]
    
    # seed the random number generator
    np.random.seed(128)    
    
    # ---------------------------------------------------------
    #  Training Data
    # ---------------------------------------------------------
    # Select training data randomly with Latin Hypercube Sampling
    
    # number of samples
    ns = 15 
    
    # perform sampling with latin hypercube
    XS = VyPy.sampling.lhc_uniform(XB,ns)
    
    # evaluate function and gradients
    FS,DFS = The_Func(XS)
    
    
    # ---------------------------------------------------------
    #  Machine Learning
    # ---------------------------------------------------------
    # Build a gpr modeling object and train it with data
    
    # start a training data object
    Train = gpr.training.Training(XB,XS,FS,DFS)
    
    # find scaling factors for normalizing data
    Scaling = gpr.scaling.Linear(Train)
    
    # scale the training data
    Train_Scl = Scaling.set_scaling(Train)
    
    # build Model
    Kernel = gpr.kernel.Gaussian(Train_Scl)
    Infer  = gpr.inference.Gaussian(Kernel)
    Learn  = gpr.learning.Likelihood(Infer)
    Model  = gpr.modeling.Regression(Learn)  
    Model.learn()
    
    
    # ---------------------------------------------------------
    #  Evaluate the Model
    # ---------------------------------------------------------    
    # Run a grid sample of the model for plotting
    
    # grid number of points per dimension
    nx = 20 
    
    # generate a grid, in the original design space bounds
    xi = np.linspace(XB[0,0], XB[0,1], nx) 
    yi = np.linspace(XB[1,0], XB[1,1], nx)
    xi,yi = np.meshgrid(xi,yi)
    
    # zip data into a design matrix
    XI = np.zeros([nx*nx,2])
    it = 0
    for ix in range(nx):
        for iy in range(nx):
            XI[it,:] = [xi[ix,iy],yi[ix,iy]]
            it = it+1
    
    # scale feature locations to match the model
    XI_scl = XI / Scaling.XI
    
    # ---------------------------------------------------------    
    # evaluate the model with the scaled features
    The_Data_Scl = Model.predict(XI_scl)
    # ---------------------------------------------------------    

    # un-scale the estimated targets
    The_Data = Scaling.unset_scaling(The_Data_Scl)
    
    # pull the estimated target function values
    FI = The_Data.YI
    
    
    # ---------------------------------------------------------
    #  Evaluate the Truth Function
    # ---------------------------------------------------------      
    # Run the same grid on the truth function
    
    # truth function at training data locations
    FST,_ = The_Func(XS)
    
    # truth funcion at grid testing locations
    FT,_ = The_Func(XI)    


    # ---------------------------------------------------------
    #  Model Errors
    # ---------------------------------------------------------
    # estimate the rms training and testing errors
    
    print 'Estimate Modeling Errors ...'
    
    # scale data - training samples
    FS_scl  = Scaling.Y.set_scaling(FS)
    FST_scl = FST / Scaling.Y # alternate syntax
    
    # scale data - grid testing samples
    FI_scl = FI / Scaling.Y
    FT_scl = FT / Scaling.Y
    
    # rms errors
    ES_rms = np.sqrt( np.mean( (FS_scl-FST_scl)**2 ) )
    EI_rms = np.sqrt( np.mean( (FI_scl-FT_scl )**2 ) )
    
    print '  Training Error = %.3f%%' % (ES_rms*100.)
    print '  Testing Error  = %.3f%%' % (EI_rms*100.)
    
    assert ES_rms < 1e-8
    assert EI_rms < 1e-3
    
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


