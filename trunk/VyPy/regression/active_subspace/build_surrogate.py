
import numpy as np

from VyPy.data import obunch

import learn as as_learn
import project as as_project

def build_surrogate(X,F,DF,XB,XC=None,dLim=-2,nAS=None,**hypers):
    
    from VyPy.regression import active_subspace, gpr
    
    if XC is None:
        XC = XB[None,0,:] * 0.0
    
    # learn the active subpace by gradients
    print 'Learn Active Subspaces ...'
    W,d = as_learn.gradient(DF)
    
    # pick number of dimensions
    nLim = np.sum( (d/np.linalg.norm(d)) > 10**dLim )
    if nAS is None:
        nAS = nLim
    else:
        if nAS > nLim:
            nAS = nLim
    
    print '  Number of dimensions = %i' % nAS
    
    # down select the number of dimensions
    U = W[:,0:nAS]
    
    # forward map training data to active space
    Y = as_project.simple(X-XC,U)
    DFDY = as_project.simple(DF,U)
    YB = np.vstack([ np.min(Y,axis=0) , 
                     np.max(Y,axis=0) ]).T * 1.3
    
    # build the surrogate
    #M_Y = gpr.library.Gaussian(YB,Y,F,DFDY, **hypers )
    M_Y = gpr.library.Gaussian(YB,Y,F,None, **hypers )
    
    # convenience function handles
    AS_Model = Active_Subspace_Surrogate()
    
        
    # pack results
    AS_Model.tag = 'active subspace model'
    AS_Model.XB  = XB
    AS_Model.XC  = XC
    AS_Model.X   = X
    AS_Model.F   = F
    AS_Model.DF  = DF
    
    AS_Model.W   = W
    AS_Model.d   = d
    AS_Model.U   = U
    AS_Model.Y   = Y
    AS_Model.YB  = YB
    
    AS_Model.M_Y = M_Y
    
    return AS_Model
    
    
class Active_Subspace_Surrogate(obunch):
    
    def __init__(self):
        self.M_Y = None
        self.U   = None
        self.XC  = None

    def g_y(self,Y):
        return self.M_Y.predict_YI(Y)
        
    def g_x(self,X):
        Y = as_project.simple(X-self.XC,self.U)
        return self.g_y(Y)