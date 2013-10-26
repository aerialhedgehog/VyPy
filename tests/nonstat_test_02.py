
import time, os, sys, copy

import numpy as np

import pylab as plt

import VyPy

def main():
    
    fit_1D()
    
    return


def fit_1D():
    
    # ---------------------------------------------------------
    #  Sampling
    # ---------------------------------------------------------
    
    XS,YS,DYS = training_data()
    XB = [[min(XS),max(XS)]]
    
    # ---------------------------------------------------------
    #  Machine Learning
    # ---------------------------------------------------------
    
    # Training
    Train = VyPy.sbo.Training(XB,XS,YS,None)
    
    # Scaling
    Scaling = VyPy.sbo.Scaling.Training(Train)
    Train = Scaling.set_scaling(Train)
    
    # Length Scaling
    #Length = length_scaling
    Length = lambda(Z): length_scaling(Scaling.X_unset(Z))
    
    # Model
    #Kernel = VyPy.sbo.Kernels.Gaussian(Train)
    Kernel = VyPy.sbo.Kernels.Gaussian_NS(Train,Length)
    
    #Kernel.Hypers.sig_f   = -0.1
    #Kernel.Hypers.len_s   = -0.4
    #Kernel.Hypers.sig_ny  = -4.0
    #Kernel.Hypers.sig_ndy = -4.0
    
    Model  = VyPy.sbo.Modeling(Kernel)
    
    # Learning
    Model.learn()

            
    # ---------------------------------------------------------
    #  Post Processing
    # ---------------------------------------------------------    
    
    # plot sites
    XP = np.array([ np.linspace(XB[0][0],XB[0][1],200) ]).T
    
    # functions, in not scaled space        
    The_Data = Model.evaluate( Scaling.X_set(XP) )
    The_Data = Scaling.unset_scaling(The_Data)
    YP = The_Data.YI
    DYP = The_Data.DYI
    
    # plot
    plt.figure(1)
    plt.plot(XP,YP,'b-')
    plt.plot(XS,YS,'r+')
    
    # plot
    plt.figure(2)
    plt.plot(XP,DYP,'b-')
    plt.plot(XS,DYS,'r+')    
    
    plt.figure(3)
    plt.plot(XP,length_scaling(XP),'b-')
    plt.show()
    
    plt.show()
    
    return


import scipy.interpolate
interpolate = scipy.interpolate

l_guesses = np.array([0.95, 0.10, 0.20, 0.50, 1.0])    
x_guesses = np.array([0.00, 0.08, 0.11, 0.20, 1.0 ]) * 10.    

interpolator = interpolate.pchip(x_guesses, l_guesses)

def length_scaling(xs):
    
    xs = VyPy.sbo.tools.check_array(xs)
    #ys = np.zeros([xs.shape[0],1])
    #for i,x in enumerate(xs):
        #ys[i] = interpolator(x)
        
    ys = np.array([ interpolator(xs[:,0]) ]).T
    #ys = np.ones_like(xs)
        
    return ys


def training_data():

    X = np.array([
        [  0. ],
        [  0.3],
        [  0.7],
        [  0.9],
        [  1.2],
        [  1.5],
        [  2. ],
        [  2.5],
        [  3. ],
        [  4. ],
        [  6. ],
        [  8. ],
        [ 10. ],
    ])
    
    
    Y = np.array([
        [-0.03222723],
        [-0.03222746],
        [-0.007998  ],
        [ 0.003999  ],
        [-0.03599099],
        [-0.03293293],
        [-0.01717217],
        [-0.00752753],
        [ 0.00094094],
        [ 0.00940941],
        [ 0.01411411],
        [ 0.01693694],
        [ 0.01928929],
    ])
    
    DY = np.array([
        [-0.00564939],
        [ 0.01507649],
        [ 0.12407742],
        [-0.11633803],
        [ 0.04211901],
        [ 0.01023362],
        [ 0.0315054 ],
        [ 0.01544723],
        [ 0.01524186],
        [ 0.00428248],
        [ 0.00141053],
        [ 0.00135261],
        [ 0.00094123],
    ])
    
    return X,Y,DY

if __name__ == '__main__':
    main()