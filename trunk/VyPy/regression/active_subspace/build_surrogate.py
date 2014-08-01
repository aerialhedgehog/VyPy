
import numpy as np

from VyPy.data import obunch

import learn as as_learn
import project as as_project

def build_surrogate(X,F,DF,XB,dLim=-2,nAS=None,**hypers):
    
    from VyPy.regression import active_subspace, gpr
    
    # learn the active subpace by gradients
    print 'Learn Active Subspaces ...'
    W,d = as_learn.gradient(DF)
    
    # pick number of dimensions
    nLim = np.sum( d > 10**dLim )
    if nAS is None:
        nAS = nLim
    else:
        if nAS > nLim:
            nAS = nLim
    
    print '  Number of dimensions = %i' % nAS
    
    # down select the number of dimensions
    U = W[:,0:nAS]
    
    # forward map training data to active space
    Y = as_project.simple(X,U)
    DFDY = as_project.simple(DF,U)
    YB = np.vstack([ np.min(Y,axis=0) , 
                     np.max(Y,axis=0) ]).T * 1.3
    
    # build the surrogate
    M_Y = gpr.library.Gaussian(YB,Y,F,DFDY, **hypers )
    #M_Y = gpr.library.Gaussian(YB,Y,F,None, **hypers )
    
    # convenience function handles
    g_y = M_Y.predict_YI
    g_x = Active_Subspace_Surrogate(g_y,U)
    
    # pack results
    results = obunch()
    results.tag = 'active subspace model'
    results.XB  = XB
    results.X   = X
    results.F   = F
    results.DF  = DF
    
    results.W   = W
    results.d   = d
    results.U   = U
    results.Y   = Y
    results.YB  = YB
    
    results.M_Y = M_Y
    results.g_x = g_x
    results.g_y = g_y
    
    return results
    
class Active_Subspace_Surrogate(object):
    def __init__(self,g_y,U):
        self.g_y = g_y
        self.U = U
        
    def __call__(self,X):
        Y = as_project.simple(X,self.U)
        G = self.g_y(Y)
        return G