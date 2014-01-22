
import time, os, gc, sys, copy
import cPickle as pickle

import numpy as np
from numpy import pi

import pylab as plt

import VyPy
from VyPy.regression import gpr
from VyPy.tools import check_array

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt


def main():
    
    T0 = time.time()

    #fit_2D()
    fit_ND()
    #opt_ND()
    #fit_VF()
    #fit_VFND()
    
    T1 = time.time()
     
    print 'Total Time: %.4f' % (T1-T0)
    

def fit_2D():
    
    # ---------------------------------------------------------
    #  Setup
    # ---------------------------------------------------------
    
    The_Func = Rosenbrock_Function
    
    XB = np.array([ [-2. , 2. ] ,
                    [-2. , 2. ] ])    
    
    # truth
    nx = 20 # grid number of points per dimension
    XT = np.linspace(XB[0,0], XB[0,1], nx)
    YT = np.linspace(XB[1,0], XB[1,1], nx)
    XT,YT = np.meshgrid(XT,YT)
    ZT = XT*0.
    for ix in range(nx):
        for iy in range(nx):    
            ZT[ix,iy] = The_Func([ XT[ix,iy], YT[ix,iy] ])[0][0,0]
    
    # training
    nt = 20 # number of samples
    X = ViPy.tools.LHC_uniform(XB,nt)
    Y,DY = The_Func(X)
    
    # zip data to evaluate
    XI = np.zeros([nx*nx,2])
    it = 0
    for ix in range(nx):
        for iy in range(nx):
            XI[it,:] = [XT[ix,iy],YT[ix,iy]]
            it = it+1
            
    # ---------------------------------------------------------
    #  Machine Learning
    # ---------------------------------------------------------
    
    # Training
    Train = gpr.Training(XB,X,Y,DY)
    #Train = VyPy.sbo.Training(XB,X,Y,DY=None)
    
    # Scaling
    Scaling = gpr.Scaling.Training(Train)
    Train = Scaling.set_scaling(Train)
    
    # Learning
    Kernel = gpr.Kernel.Gaussian(Train)
    Model  = gpr.Modeling(Kernel)
    Model.learn()

    # Evaluate
    XI_scl = Scaling.set_scaling(XI,'X')
    The_Data = Model.evaluate(XI_scl)
    The_Data = Scaling.unset_scaling(The_Data)
    
    # ---------------------------------------------------------
    #  Post Processing
    # ---------------------------------------------------------    
    
    # unzip
    YI = The_Data.YI
    XP = XT; YP = YT; ZP = YT*0.
    it = 0
    for ix in range(nx):
        for iy in range(nx):
            ZP[ix,iy] = YI[it]
            it = it+1
    
    # plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    pnts = ax.plot(X[:,0],X[:,1],Y[:,0],'r+',mew=3,ms=15)
    surf = ax.plot_surface(XP,YP,ZP, cmap=cm.autumn, rstride=1,cstride=1, linewidth=0, antialiased=False)
    truh = ax.plot_surface(XT,YT,ZT, cmap=cm.jet, rstride=1,cstride=1, linewidth=0, antialiased=False)
    plt.draw()
    
    plt.show()
    
#: def fit_2D()


def fit_ND():
    
    # ---------------------------------------------------------
    #  Setup
    # ---------------------------------------------------------
    
    The_Func = Rosenbrock_Function
    ND = 3 # dimensions
    XB = np.array( [[-2.,2.]]*ND )
    
    # training
    nt = 60
    X = VyPy.sampling.lhc_uniform(XB,nt)
    Y,DY = The_Func(X)
    
    
    # ---------------------------------------------------------
    #  Machine Learning
    # ---------------------------------------------------------
    
    # Training
    Train = gpr.Training(XB,X,Y,DY=None)
    
    # Scaling
    Scaling = gpr.Scaling.Training(Train)
    Train = Scaling.set_scaling(Train)
    
    # Learning
    Kernel = gpr.Kernel.Gaussian(Train)
    Infer  = gpr.Inference(Kernel)
    Learn  = gpr.Learning(Infer)
    Model  = gpr.Modeling(Learn)
    Model.learn()
    
    
    # ---------------------------------------------------------
    #  Post Processing
    # ---------------------------------------------------------    
    
    # functions to plot, in scaled space
    f1 = lambda(Z): Model.predict(Z).YI                               
    f2 = lambda(Z): Scaling.Y_set( The_Func( Scaling.X_unset(Z) )[0] ) 
    # center point, scaled space
    x0 = Scaling.X_set( [1.0] * ND )
    # plot bounds, scaled space
    xb = Train.XB
    
    # plot
    fig = plt.figure(1)
    
    ax = VyPy.plotting.spider_axis(fig,x0,xb)
    VyPy.plotting.spider_trace(ax,f1,x0,xb,20,'b-',lw=2,label='Fit')
    VyPy.plotting.spider_trace(ax,f2,x0,xb,20,'r-',lw=2,label='Truth')
    
    ax.legend()
    ax.set_zlabel('Y (scaled)')
    
    plt.draw(); plt.show()
    
#: def fit_ND()


def opt_ND():

    # ---------------------------------------------------------
    #  Setup
    # ---------------------------------------------------------
    
    The_Func = Rosenbrock_Function
    ND = 4 # dimensions
    XB = np.array( [[-2.,2.]]*ND )
    
    # training
    nt = 50
    X = VyPy.sbo.tools.LHC_uniform(XB,nt)
    Y,DY = The_Func(X)
    
    
    # ---------------------------------------------------------
    #  Machine Learning
    # ---------------------------------------------------------
    print "Learning ... "
    
    # Training
    Train = VyPy.sbo.Training(XB,X,Y,DY)
    
    # Scaling
    Scaling = VyPy.sbo.Scaling.Training(Train)
    Train = Scaling.set_scaling(Train)
    
    # Learning
    Kernel = VyPy.sbo.Kernels.Gaussian(Train)
    Model  = VyPy.sbo.Modeling(Kernel)
    Model.learn()
    
    print ""


    # ---------------------------------------------------------
    #  Optimization
    # ---------------------------------------------------------
    print "Optimization ..."
    
    # functions
    Opt_Func = Model.pyopt_function
    Opt_Grad = Model.pyopt_gradient
    
    # problem setup
    Opt_Prob = pyOpt.Optimization('YI Minimization',Opt_Func)
    
    # variables and bounds
    for ix in range(ND):
        Opt_Prob.addVar('X%i'%ix,'c',lower=Train.XB[ix,0],upper=Train.XB[ix,1],value=0.)
        
    # objective name
    Opt_Prob.addObj('Estimated Objective')
    
    # global optimization
    print 'Global Optimization (ALPSO)'
    The_Optimizer = pyOpt.ALPSO(pll_type='MP')
    The_Optimizer.setOption('fileout',0)
    The_Optimizer.setOption('maxOuterIter',2)
    The_Optimizer.setOption('stopCriteria',0) # by maxits
    The_Optimizer.setOption('SwarmSize',ND*10)
    The_Optimizer(Opt_Prob) # runs the optimization
    
    # local optimization
    print 'Local Optimization (SLSQP)'
    The_Optimizer = pyOpt.SLSQP()
    The_Optimizer.setOption('IPRINT',-1)
    The_Optimizer.setOption('ACC',1e-10)
    [YI_min,X_min,Info] = The_Optimizer(Opt_Prob.solution(0),sens_type=Opt_Grad) # starts from last solution
    
    # report
    print "YImin = %.4f" % Scaling.unset_scaling(YI_min,'YI')
    print "XImin = %s"   % Scaling.unset_scaling(X_min,'X')
    
    
    # ---------------------------------------------------------
    #  Post Processing
    # ---------------------------------------------------------    
    
    # functions to plot, in scaled space
    f1 = lambda(Z): Model.evaluate(Z).YI                               
    f2 = lambda(Z): Scaling.Y_set( The_Func( Scaling.X_unset(Z) )[0] ) 
    # center point, scaled space
    x0 = X_min
    # plot bounds, scaled space
    xb = Train.XB
    
    # plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')    
    VyPy.sbo.tools.plot_spider(ax,[f1,f2],x0,xb)
    
    # labels
    ax.legend(['Fit','Truth'])
    ax.set_zlabel('Y (scaled)')
    
    # show
    plt.draw(); plt.show()
    
#: def opt_ND()


def fit_VF():
    
    # ---------------------------------------------------------
    #  Setup
    # ---------------------------------------------------------    
        
    # high fidelity first
    def Func_FI1(X):
        Y = X*X + 0.2*np.sin(2*pi*X)
        DY = 2.0*X + 0.2* 2*pi * np.cos(2*pi*X)
        return Y,DY
    def Func_FI2(X):
        Y = X*X
        DY = 2.0*X
        return Y,DY
        
    # bounds
    XB = np.array([ [-2,2] ])
    
    # truth    
    XT = np.array([ np.linspace(XB[0,0],XB[0,1],100) ]).T
    YT1 = Func_FI1(XT)[0]
    YT2 = Func_FI2(XT)[0]
    
    # training
    nt_1 = 4
    nt_2 = 20
    
    X1  = np.array([ np.linspace(XB[0,0],XB[0,1],nt_1) ]).T 
    X2  = np.array([ np.linspace(XB[0,0],XB[0,1],nt_2) ]).T

    Y1,DY1 = Func_FI1(X1)
    Y2,DY2 = Func_FI2(X2)    
    
    
    # ---------------------------------------------------------
    #  Machine Learning
    # ---------------------------------------------------------
    
    # Training
    Trains = VyPy.sbo.VF_Training()
    Trains['FI0'] = VyPy.sbo.Training(XB,X1,Y1)
    Trains['FI1'] = VyPy.sbo.Training(XB,X2,Y2)
    
    # Scaling
    Scaling = VyPy.sbo.Scaling.Training(Trains[1])
    Trains[0] = Scaling.set_scaling(Trains[0])
    Trains[1] = Scaling.set_scaling(Trains[1])
    
    # Learning
    Kernel = VyPy.sbo.Kernels.Gaussian_VF(Trains)
    Model  = VyPy.sbo.Modeling(Kernel)
    Model.learn()

    # Evaluate
    XI_scl = Scaling.set_scaling(XT,'X')
    The_Data = Model.evaluate(XI_scl)
    The_Data = Scaling.unset_scaling(The_Data)
    YI_VF = The_Data.YI
    
    
    # ---------------------------------------------------------
    #  Verification
    # ---------------------------------------------------------
    
    # High Fidelity Only
    Kernel = VyPy.sbo.Kernels.Gaussian(Trains[0])
    Model  = VyPy.sbo.Modeling(Kernel)
    Model.precalc()

    # Evaluate
    The_Data = Model.evaluate(XI_scl)
    The_Data = Scaling.unset_scaling(The_Data)
    YI_SF = The_Data.YI
    
    
    # ---------------------------------------------------------
    #  Post Processing
    # ---------------------------------------------------------    
    
    plt.figure(1)
    
    # labels
    plt.plot([np.nan],[0], '-' , color='b', lw=2)
    plt.plot([np.nan],[0], '-' , color='r', lw=2)
    plt.plot([np.nan],[0], '--', color='y', lw=1)
    plt.plot([np.nan],[0], '-' , color='k', lw=2)
    plt.legend(['Fidelity 1','Fidelity 2','SF Surrogate','VF Surrogate'])
    plt.ylabel('Y')
    plt.xlabel('X')    
    
    # truth
    plt.plot(XT,YT1, '-', color='b', lw=2)
    plt.plot(XT,YT2, '-', color='r', lw=2)
    plt.plot(X1,Y1 , '+', color='b', mew=3, ms=10)
    plt.plot(X2,Y2 , '+', color='r', mew=3, ms=10)    
    
    # predicted
    plt.plot(XT,YI_SF, '--', color='y', lw=1)
    plt.plot(XT,YI_VF, '-', color='k', lw=2)
    
    plt.show()
    
    
#: def fit_VF_1D()


def fit_VFND():
    
    # ---------------------------------------------------------
    #  Setup
    # ---------------------------------------------------------    
        
    # high fidelity first
    def Func_FI1(X):
        return Rosenbrock_Function(X)[0]
    def Func_FI2(X):
        return Rosenbrock_Function(X)[0] + Hyperplane_Function(X)[0]*50.0 
        #return Rosenbrock_Function(X)[0] * Hyperplane_Function(X)[0]/10.0 # works bad
        
    # bounds
    ND = 3
    XB = np.array( [[-2,2]]*ND )
    
    # training
    nt_1 = 20
    nt_2 = ND*60
    
    X1  = VyPy.sbo.tools.LHC_uniform(XB,nt_1) 
    X2  = VyPy.sbo.tools.LHC_uniform(XB,nt_2)
    
    Y1 = Func_FI1(X1)
    Y2 = Func_FI2(X2)    
    
    
    # ---------------------------------------------------------
    #  Machine Learning
    # ---------------------------------------------------------
    
    # Training
    Trains = VyPy.sbo.VF_Training()
    Trains['FI0'] = VyPy.sbo.Training(XB,X1,Y1)
    Trains['FI1'] = VyPy.sbo.Training(XB,X2,Y2)
    
    # Scaling
    Scaling = VyPy.sbo.Scaling.Training(Trains[1])
    Trains[0] = Scaling.set_scaling(Trains[0])
    Trains[1] = Scaling.set_scaling(Trains[1])
    
    # Learning
    Kernel = VyPy.sbo.Kernels.Gaussian_VF(Trains)
    Model  = VyPy.sbo.Modeling(Kernel)
    #Model.learn()
    
    # Function Handle
    Model.precalc()
    Func_VF = lambda(Z): Model.evaluate(Z).YI
    
    
    # ---------------------------------------------------------
    #  Verification
    # ---------------------------------------------------------
    
    # High Fidelity Only
    Train0  = copy.deepcopy(Trains[0])
    Kernel0 = VyPy.sbo.Kernels.Gaussian(Train0)
    Model0  = VyPy.sbo.Modeling(Kernel0)
    #Model0.learn()
    
    # Function Handle
    Model0.precalc()
    Func_SF = lambda(Z): Model0.evaluate(Z).YI
    
    
    # ---------------------------------------------------------
    #  Post Processing
    # ---------------------------------------------------------
    
    # truth functions in scaled space
    Func_FI1_Scl = lambda(Z): Scaling.Y_set( Func_FI1( Scaling.X_unset(Z) ) ) 
    Func_FI2_Scl = lambda(Z): Scaling.Y_set( Func_FI2( Scaling.X_unset(Z) ) ) 
    
    # errors
    print "Errors ..."
    NS = 100*ND
    Xs_err = VyPy.sbo.tools.LHC_uniform(Trains.XB,NS,None,3)
    YT = Func_FI1_Scl(Xs_err)
    YI_SF = Func_SF(Xs_err)
    YI_VF = Func_VF(Xs_err)
    
    Err_SF = np.sqrt(np.mean( (YT-YI_SF)**2 ))
    Err_VF = np.sqrt(np.mean( (YT-YI_VF)**2 ))
    
    print "Err_SF = %.2f %%" % (Err_SF*100.)
    print "Err_VF = %.2f %%" % (Err_VF*100.)

    # center point, scaled space
    x0 = Scaling.X_set( [1.0] * ND )
    # plot bounds, scaled space
    xb = Trains.XB
    
    # plot
    fig = plt.figure(1)
    
    ax = VyPy.sbo.tools.plot_spider_axis(fig,x0,xb)
    VyPy.sbo.tools.plot_spider_trace(ax,Func_FI1_Scl,x0,xb,'b-' ,lw=2,label='Fidelity 1')
    VyPy.sbo.tools.plot_spider_trace(ax,Func_FI2_Scl,x0,xb,'r-' ,lw=2,label='Fidelity 2')
    VyPy.sbo.tools.plot_spider_trace(ax,Func_SF     ,x0,xb,'y--',lw=1,label='SF Surrogate')
    VyPy.sbo.tools.plot_spider_trace(ax,Func_VF     ,x0,xb,'k-' ,lw=2,label='HF Surrogate')
    
    ax.legend()
    ax.set_zlabel('Y (scaled)')
    
    plt.draw(); plt.show()    
    
#: def fit_VFND()

# -------------------------------------------------------------
#  Test Functions
# -------------------------------------------------------------

def Rosenbrock_Function(X):
    X = check_array(X)
    D = X.shape[1]
    Y = 0.
    DY = X*0.
    for I in range(D):
        if I < D-1:
            Y = Y + 100.*( X[:,I+1]-X[:,I]**2. )**2. + ( 1-X[:,I] )**2.
            DY[:,I] = DY[:,I] - 400.*( X[:,I+1]-X[:,I]**2. ) * X[:,I] - 2.*( 1.-X[:,I] )
        if I>0:
            DY[:,I] = DY[:,I] + 200.*( X[:,I]-X[:,I-1]**2. )
    Y = check_array(Y,'col')
    return Y,DY

def Rastrigin_Function(X):
    X = check_array(X)
    scl = 1./2.5
    sgn = 1.0
    X = X * scl
    D = X.shape[1]
    Y  = sgn*( 10.*D + np.sum( X**2. - 10.*np.cos(2.*pi*X) , 1 ) );
    DY = sgn*( 2.*X + 20.*pi*np.sin(2.*pi*X) ) * scl;
    Y = check_array(Y,'col')
    return Y,DY

def Parabolic_Function(X):
    X = check_array(X)
    D = X.shape[1]
    C = np.ones([1,D])
    #C = np.array([ np.arange(D)+1. ])
    Y  = np.dot( X**2. , C.T  ) - 10.
    DY = 2.*X*C
    Y = check_array(Y,'col')
    return Y,DY
    
def Hyperplane_Function(X):
    X = check_array(X) + 0.5
    N,D = X.shape
    C = np.array([ np.arange(D)+1. ])
    I = np.ones([N,D])
    Y  = np.dot( X , C.T  )
    DY = C * I
    Y = check_array(Y,'col')
    return Y,DY

if __name__=='__main__':
        
    profile = False
        
    if not profile:
        main()
    else:
        profile_file = 'log_Profile.out'
        
        import cProfile
        cProfile.run('import package_tests_01; package_tests_01.main()', profile_file)
        
        import pstats
        p = pstats.Stats(profile_file)
        p.sort_stats('time').print_stats(20)    



## GRAVETART



## ---------------------------------------------------------
##  Post Processing
## ---------------------------------------------------------    

## Evaluate
#XI_scl = Scaling.set_scaling(XT,'X')
#YI_VF = Func_VF(XI_scl)
#YI_VF = Scaling.unset_scaling(YI_VF,'Y')    
#YI_SF = Func_SF(XI_scl)
#YI_SF = Scaling.unset_scaling(YI_SF,'Y')  

#plt.figure(1)

## labels
#plt.plot([np.nan],[0], '-' , color='b', lw=2)
#plt.plot([np.nan],[0], '-' , color='r', lw=2)
#plt.plot([np.nan],[0], '--', color='y', lw=1)
#plt.plot([np.nan],[0], '-' , color='k', lw=2)
#plt.legend(['Fidelity 1','Fidelity 2','SF Surrogate','VF Surrogate'])
#plt.ylabel('Y')
#plt.xlabel('X')    

## truth
#plt.plot(XT,YT1, '-', color='b', lw=2)
#plt.plot(XT,YT2, '-', color='r', lw=2)
#plt.plot(X1,Y1 , '+', color='b', mew=3, ms=10)
#plt.plot(X2,Y2 , '+', color='r', mew=3, ms=10)    

## predicted
#plt.plot(XT,YI_SF, '--', color='y', lw=1)
#plt.plot(XT,YI_VF, '-', color='k', lw=2)

#plt.show()
