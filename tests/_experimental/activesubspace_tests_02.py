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
from VyPy.regression import gpr, active_subspace
from VyPy import optimize as opt

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
    The_Func = composite_function
    The_Con  = composite_constraint
    
    # hypercube bounds
    ND = 4 # dimensions
    XB = np.array( [[-2.,2.]]*ND )
    
    # ---------------------------------------------------------
    #  Training Data
    # ---------------------------------------------------------
    # Select training data randomly with Latin Hypercube Sampling
    
    # number of samples
    ns = 250 
    ## ns = 10  # try for lower model accuracy
    
    # perform sampling with latin hypercube
    XS = VyPy.sampling.lhc_uniform(XB,ns)
    
    # evaluate function and gradients
    FS,DFS = The_Func(XS)
    CS,DCS = The_Con(XS)
    
    
    # ---------------------------------------------------------
    #  Machine Learning
    # ---------------------------------------------------------
    
    # number of active domain dimensions
    N_AS  = ND  # max number
    d_Lim = -3. # eigenvalue limit (log10)
    
    #Model = gpr.library.Gaussian(XB,XS,FS,DFS)
    Model   = active_subspace.build_surrogate(XS,FS,DFS,XB,d_Lim,N_AS,probNze=-2.0)
    Model_C = active_subspace.build_surrogate(XS,CS,DCS,XB,d_Lim,N_AS,probNze=-2.0)
    
    # pull function handles for plotting and evaluating
    g_x = Model.g_x          # the surrogate
    #g_x = Model.predict_YI
    f_x = lambda(Z): The_Func(Z)[0] # the truth function
    
    c_x = Model_C.g_x
    
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
    Scaling = Model.M_Y.Scaling # careful, this is in the active subspace
    #Scaling = Model.Scaling
    
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
    #  Optimization
    # ---------------------------------------------------------
    
    print 'Optimize in Active Subspace ...'
    
    problem = opt.Problem()
    
    var = opt.Variable()
    var.tag = 'x'
    var.initial = np.array([[0.0] * ND])
    var.bounds = XB.T
    problem.variables.append(var)
    
    obj = opt.Objective()
    #obj.evaluator = lambda X: {'f' : f_x(X['x'])+np.linalg.norm(X['x'],axis=1)/10.}
    obj.evaluator = lambda X: {'f' : g_x(X['x'])+np.linalg.norm(X['x'],axis=1)/10.}
    obj.tag = 'f'
    problem.objectives.append(obj)
    
    con = opt.Constraint()
    #con.evaluator = lambda X: {'c' : The_Con(X['x'])[0]}
    con.evaluator = lambda X: {'c' : c_x(X['x'])}
    con.edge = -2.0
    con.sense = '='
    con.tag = 'c'
    problem.constraints.append(con)
    
    driver = opt.drivers.CMA_ES(0)
    
    result = driver.run(problem)
    
    for r in result: print r
    
    Xmin = result[1]['x']
    
    
    # ---------------------------------------------------------
    #  Plotting
    # ---------------------------------------------------------
    # plot the estimated and truth surface, evaluate rms error
    
    
    plt.figure(0)
    plt.plot(Model.d,'bo-')
    plt.plot(Model_C.d,'ro-')
    plt.plot([0,ND-1],[10**d_Lim,10**d_Lim],'k--')
    plt.gca().set_yscale("log", nonposy='clip')
    plt.title('Eigenvalue Powers')
    
    print 'Plot Response Surface ...'
    
    # center point
    #X0 = [1.0] * ND
    X0 = Xmin
    
    ## rosenbrock local minimum for 4 <= dim <= 7
    #X0[0] = -1.
    
    # plot spider legs
    fig = plt.figure(1)
    ax = VyPy.plotting.spider_axis(fig,X0,XB)
    VyPy.plotting.spider_trace(ax,g_x,X0,XB,100,'b-',lw=2,label='Fit')
    VyPy.plotting.spider_trace(ax,f_x,X0,XB,100,'r-',lw=2,label='Truth')
    ax.legend()
    ax.set_zlabel('F')
    
    
    # in active domain
    U = Model.U
    Y0 = active_subspace.project.simple(X0,U)
    g_y = Model.g_y
    YB = Model.YB
    
    # plot spider legs
    fig = plt.figure(2)
    ax = VyPy.plotting.spider_axis(fig,Y0,YB)
    VyPy.plotting.spider_trace(ax,g_y,Y0,YB,100,'b-',lw=2,label='Fit')
    ax.legend()
    ax.set_zlabel('F')
    
    # in active domain
    U = Model_C.U
    Y0 = active_subspace.project.simple(X0,U)
    g_y = Model_C.g_y
    YB = Model_C.YB
    
    # plot spider legs
    fig = plt.figure(3)
    ax = VyPy.plotting.spider_axis(fig,Y0,YB)
    VyPy.plotting.spider_trace(ax,g_y,Y0,YB,100,'b-',lw=2,label='Fit')
    ax.legend()
    ax.set_zlabel('C')
    
    plt.draw(); 
    plt.show(block=True)    
    
    
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
    scl = 1.0
    sgn = 1.0
    X = X * scl
    D = X.shape[1]
    Y  = sgn*( 10.*D + np.sum( 10*X**2. - 10.*np.cos(2.*pi*X) , 1 ) );
    DY = sgn*( 2.*X + 20.*pi*np.sin(2.*pi*X) ) * scl;
    Y = atleast_2d(Y,'col')
    return Y,DY

# --- Parabolic Function ---
def parabolic_function(X):
    X = atleast_2d(X)
    D = X.shape[1]
    C = np.ones([1,D]) * 10
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

def rotation_function(X):
    X = atleast_2d(X) + 0.0
    N,D = X.shape
    C = np.ones([1,D]) * 1/D
    #C = np.array([ np.arange(D)+1. ])
    Y  = np.dot( X , C.T  )
    I = np.ones([N,D])
    DY = C * I
    Y = atleast_2d(Y,'col')
    return Y,DY

def composite_function(X):
    X = atleast_2d(X)
    N,D = X.shape
    Y = np.zeros([N,1])
    DY = np.zeros([N,D])
    
    #y,dy = hyperplane_function(X)
    #Y  += y
    #DY += dy
    
    k = D
    f,df = rotation_function(X[:,0:k])
    g,dg = rastrigin_function(f)
    Y += g
    DY[:,0:k] += dg * df

    return Y,DY
    
def rotation_function_2(X):
    X = atleast_2d(X) + 0.0
    N,D = X.shape
    C = np.ones([1,D]) * 1/D
    C[0,0:D/2] = -C[0,0:D/2]
    #C = np.array([ np.arange(D)+1. ])
    Y  = np.dot( X , C.T  )
    I = np.ones([N,D])
    DY = C * I
    Y = atleast_2d(Y,'col')
    return Y,DY    
    
def composite_constraint(X):
    X = atleast_2d(X)
    N,D = X.shape
    Y = np.zeros([N,1])
    DY = np.zeros([N,D])
    
    y,dy = hyperplane_function(X)
    Y  += y
    DY += dy
    
    f,df = rotation_function_2(X)
    g,dg = parabolic_function(f)
    Y += g
    DY += dg * df

    return Y,DY


# -------------------------------------------------------------
#  Start Main
# -------------------------------------------------------------

if __name__=='__main__':
    main()


