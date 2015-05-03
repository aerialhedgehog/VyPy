
import numpy as np

from VyPy.data import obunch

import learn as as_learn
import project as as_project

def build_surrogate(X,F,DF,XB,XC=None,
                    dLim=-2,nAS=None,
                    gpr_grads=True,**hypers):
    
    from VyPy.regression import active_subspace, gpr
    
    if XC is None:
        XC = XB[None,:,0] * 0.0
        
    training_fullspace = gpr.training.Training(XB,X,F,DF)
    scaling_fullspace = gpr.scaling.Linear(training_fullspace)
    
    # build scaled space
    X_SCL  = scaling_fullspace.X.set_scaling(X)
    F_SCL  = scaling_fullspace.Y.set_scaling(F)
    DF_SCL = scaling_fullspace.DY.set_scaling(DF)
    XB_SCL = scaling_fullspace.XB.set_scaling(XB)
    XC_SCL = scaling_fullspace.X.set_scaling(XC)
    
    # learn the active subpace by gradients
    print 'Learn Active Subspaces ...'
    W,d = as_learn.gradient(DF_SCL) # embeded in scaled space
    
    # pick number of dimensions
    #nLim = np.sum( (d/np.linalg.norm(d)) > 10**dLim )
    nLim = np.sum( d > 10**dLim )
    if nAS is None:
        nAS = nLim
    else:
        if nAS > nLim:
            nAS = nLim
    
    print '  Number of dimensions = %i' % nAS
    
    # down select the number of dimensions
    U = W[:,0:nAS] # active      # embeded in scaled space
    V = W[:,nAS:]  # inactive    # embeded in scaled space
    
    # forward map training data to active space
    Y = as_project.simple(X_SCL-XC_SCL,U)  # embeded in scaled space
    DFDY = as_project.simple(DF_SCL,U)     # embeded in scaled space
    YB = np.vstack([ np.min(Y,axis=0) ,    # embeded in scaled space
                     np.max(Y,axis=0) ]).T
    
    # build the surrogate
    if gpr_grads:
        M_Y = gpr.library.Gaussian(YB,Y,F_SCL,DFDY, **hypers ) # embeded in scaled X
    else:
        M_Y = gpr.library.Gaussian(YB,Y,F_SCL,None, **hypers ) # embeded in scaled X
    
    # convenience function handles
    AS_Model = Active_Subspace_Surrogate() # embeded in scaled space!!!
    
        
    # pack results
    AS_Model.tag = 'active subspace model'
    AS_Model.XB  = XB
    AS_Model.XC  = XC
    AS_Model.X   = X
    AS_Model.F   = F
    AS_Model.DF  = DF
    
    AS_Model.W   = W
    AS_Model.d   = d
    AS_Model.n   = nAS
    AS_Model.U   = U
    AS_Model.V   = V
    AS_Model.Y   = Y
    AS_Model.YB  = YB
    
    AS_Model.M_Y = M_Y
    AS_Model.SCL_X = scaling_fullspace
    
    return AS_Model
    
    
class Active_Subspace_Surrogate(obunch):
    
    def __init__(self):
        self.M_Y = None
        self.SCL_X = None
        self.U   = None
        self.V   = None
        self.d   = None
        self.XC  = None
        
        self.y_last = None
        self.p_last = None
        
    def predict_y(self,Y):
        #if not self.y_last is None and np.sum(Y-self.y_last)==0.:
            #P = self.p_last
        #else:
        P = self.M_Y.predict(Y)
        self.p_last = P
        self.y_last = Y
        return P
        
    def g_y(self,Y):
        G_SCL = self.predict_y(Y).YI
        G = self.SCL_X.Y.unset_scaling(G_SCL)
        return G
        
    def g_x(self,X):
        Y = self.y(X)
        G = self.g_y(Y)
        return G
    
    def s_y(self,Y):
        S_SCL = self.predict_y(Y).CovYI
        return S_SCL
    
    def s_x(self,X):
        Y = self.y(X)
        S_SCL = self.s_y(Y)
        return S_SCL
    
    def y(self,X):
        X_SCL  = self.SCL_X.X.set_scaling(X)    # embed in scaled space
        XC_SCL = self.SCL_X.X.set_scaling(self.XC)   
        Y = as_project.simple(X_SCL-XC_SCL,self.U)
        return Y
    
    def z(self,X):
        X_SCL  = self.SCL_X.X.set_scaling(X)
        XC_SCL = self.SCL_X.X.set_scaling(self.XC)   # embed in scaled space
        Z = as_project.simple(X_SCL-XC_SCL,self.V)
        return Z

    def y_dist(self,X):
        Y = self.y(X)
        Y_dist = np.sqrt( np.sum(Y**2,axis=1) )[:,None]
        return Y_dist
    
    def y_dist2(self,Y):
        Y_dist = np.sqrt( np.sum(Y**2,axis=1) )[:,None]
        return Y_dist    
    
    def z_dist(self,X):
        Z = self.z(X)
        Z_dist = np.sqrt( np.sum(Z**2,axis=1) )[:,None]
        return Z_dist
    
    def yb_con(self,X):
        Y = self.y(X)
        YB = self.YB
        dY = np.vstack([-Y.T+(YB[:,0,None]),
                         Y.T-(YB[:,1,None])])
        return dY
    
    def z_norm(self,X):
        
        from numpy import dot
        
        d = self.d
        V = self.V
        #y_scl = self.SCL_X.Y.scale
        
        nz = V.shape[1]
        dz = d[-nz:]
        #Lz = np.diag(dz)
        Lz = np.sqrt( np.diag(dz) )
        
        z = self.z(X)
        
        z_norm = dot(z,dot(Lz,z.T)) * 0.1
        #z_norm = dot(z,dot(Lz,z.T))/nz * 0.1
        #z_norm = np.sqrt( dot(z,dot(Lz,z.T)) ) *0.1
        #z_norm = dot(z,dot(Lz,z.T)) * y_scl
        
        return z_norm