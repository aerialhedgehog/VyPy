""" gpr_fit_2d.py
    An example that uses functionality from the GPR module
    to regress a 2-Dimensional Function
"""

# ------------------------------------------------------------
#   Imports
# ------------------------------------------------------------

import time, os, sys, copy

import numpy as np
from numpy import pi

import pylab as plt
from matplotlib import cm

import VyPy
from VyPy.regression import gpr

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
    ## ns = 10  # try for lower model accuracy
    
    # perform sampling with latin hypercube
    XS = VyPy.sampling.lhc_uniform(XB,ns)
    
    # evaluate function and gradients
    FS,DFS = The_Func(XS)
    
    
    # ---------------------------------------------------------
    #  Machine Learning
    # ---------------------------------------------------------
    # Build a gpr modeling object and train it with data
    
    # this class factory setups up all the data structures
    # and will learn the hyperparemeters for the surrogate model
    Model = gpr.library.Gaussian(XB,XS,FS,DFS)
    ## Model = gpr.library.Gaussian(XB,XS,FS) # no gradients, try for lower model accuracy
    
    # this involves the following, see gpr_fit_2d.py for more details
    #   Train = gpr.training.Training(XB,X,Y,DY)
    #   Scaling = gpr.scaling.Linear(Train)
    #   Train = Scaling.set_scaling(Train)
    #   Kernel = gpr.kernel.Gaussian(Train,**hypers)
    #   Infer  = gpr.inference.Gaussian(Kernel)
    #   Learn  = gpr.learning.Likelihood(Infer)
    #   Model  = gpr.modeling.Regression(Learn,Scaling)
    #   Model.learn()
    
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
    
    
    # ---------------------------------------------------------
    #  Plotting
    # ---------------------------------------------------------
    # plot the estimated and truth surface, evaluate rms error
    
    print 'Plot Response Surface ...'
    
    # center point - global minimum
    X0 = [1.0] * ND
    
    ## local minimum for 4 <= dim <= 7
    #X0[0] = -1.
    
    # plot spider legs
    fig = plt.figure(1)
    ax = VyPy.plotting.spider_axis(fig,X0,XB)
    VyPy.plotting.spider_trace(ax,g_x,X0,XB,20,'b-',lw=2,label='Fit')
    VyPy.plotting.spider_trace(ax,f_x,X0,XB,20,'r-',lw=2,label='Truth')
    ax.legend()
    ax.set_zlabel('F')
    
    plt.draw()
    
    # show the plot
    ## plt.show(block=True) 
    # moved to the end of the file to allow regression testing
    
    
    # Done!
    return
    
#: def main()


# -------------------------------------------------------------
#  Test Functions
# -------------------------------------------------------------

from VyPy.tools import atleast_2d

# --- Rosenbrock Function ---
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

# --- Rastrigin Function ---
def rastrigin_function(X):
    X = atleast_2d(X)
    scl = 1./2.5
    sgn = 1.0
    X = X * scl
    D = X.shape[1]
    Y  = sgn*( 10.*D + np.sum( X**2. - 10.*np.cos(2.*pi*X) , 1 ) );
    DY = sgn*( 2.*X + 20.*pi*np.sin(2.*pi*X) ) * scl;
    Y = atleast_2d(Y,'col')
    return Y,DY

# --- Parabolic Function ---
def parabolic_function(X):
    X = atleast_2d(X)
    D = X.shape[1]
    C = np.ones([1,D])
    #C = np.array([ np.arange(D)+1. ])
    Y  = np.dot( X**2. , C.T  ) - 10.
    DY = 2.*X*C
    Y = atleast_2d(Y,'col')
    return Y,DY
  
# --- Hyperplane Function ---
def hyperplane_function(X):
    X = atleast_2d(X) + 0.5
    N,D = X.shape
    C = np.array([ np.arange(D)+1. ])
    I = np.ones([N,D])
    Y  = np.dot( X , C.T  )
    DY = C * I
    Y = atleast_2d(Y,'col')
    return Y,DY


# -------------------------------------------------------------
#  Start Main
# -------------------------------------------------------------

if __name__=='__main__':
    main()
    plt.show(block=True)        

