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
    XB = np.array([ [-2. , 2. ] ,   # [ lb_0 , ub_0 ]
                    [-2. , 2. ] ])  # [ lb_1 , ub_1 ]
    
    # ---------------------------------------------------------
    #  Training Data
    # ---------------------------------------------------------
    # Select training data randomly with Latin Hypercube Sampling
    
    # number of samples
    ns = 15 
    ## ns = 7  # try for lower model accuracy
    
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
    ## Train = gpr.Training(XB,XS,FS) # no gradients, try for lower model accuracy
    
    # find scaling factors for normalizing data
    Scaling = gpr.scaling.Linear(Train)
    
    # scale the training data
    Train_Scl = Scaling.set_scaling(Train)
    # In this case the scaling is performed by normalizing on 
    # the experimental range ( max()-min() ) for input feature
    # samples XS (by dimension) and output target samples FS.
    
    # choose a kernel 
    Kernel = gpr.kernel.Gaussian(Train_Scl)
    
    # choose an inference model (only one for now)
    Infer  = gpr.inference.Gaussian(Kernel)
    
    # choose a learning model (only one for now)
    Learn  = gpr.learning.Likelihood(Infer)
    
    # start a gpr modling object
    Model  = gpr.modeling.Regression(Learn)  
    # This object holds all the model's assumptions and 
    # data, in order to expose a simple interface.
    # Optionally, the Scaling data can be provided to
    # allow this object to accept and return dimensional data:
    #   Model = gpr.modeling.Regression(Learn,Scaling) 
    
    # learn on the training data
    Model.learn()
    # In this case we use Marinal Likelihood Minimization, 
    # first with a global optimizer (CMA_ES), followed up by
    # a local optimizer (SLSQP)
    
    # The model is now available for estimation, and can be used
    # for prediction with Model.predict(XI).
    # The remainer of this script is mostly post processing.
    
    
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
    # since we constructed a model with scaled training data,
    # we must provide scaled feature locations to predict    
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
    
    
    # ---------------------------------------------------------
    #  Plotting
    # ---------------------------------------------------------
    # plot the estimated and truth surface, evaluate rms error
    
    print 'Plot Response Surface ...'
    
    # unzip data for plotting a gridded surface
    fi = xi*0. # estimated targets
    ft = xi*0. # truth targets
    it = 0
    for ix in range(nx):
        for iy in range(nx):
            fi[ix,iy] = FI[it]
            ft[ix,iy] = FT[it]
            it = it+1
    
    # start the plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    # plot the training data
    pnts = ax.plot(XS[:,0],XS[:,1],FS[:,0],'r+',mew=3,ms=15)
    
    # plot the estimated response surface
    surf = ax.plot_surface(xi,yi,fi, cmap=cm.jet, rstride=1,cstride=1, linewidth=0, antialiased=False)
    
    # plot the truth response surface
    #truh = ax.plot_surface(xi,yi,ft, cmap=cm.autumn, rstride=1,cstride=1, linewidth=0, antialiased=False)
    
    # show the plot
    plt.show()
    
    
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


