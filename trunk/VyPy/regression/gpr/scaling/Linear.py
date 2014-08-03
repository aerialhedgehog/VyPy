# scaling functions

import numpy as np
import copy
from VyPy.data.scaling import ScalingBunch
from VyPy.data.scaling import Linear as LinearFunction


class Linear(ScalingBunch):
    
    def __init__(self,Train):
        
        self.calc_scaling(Train)
        
        return
    
    def calc_scaling(self,Train):
        
        # unpack
        X  = Train.X
        Y  = Train.Y
        DY = Train.DY
        XB = Train.XB
        _,nx = X.shape
        
        # calculate scaling parameters
        Xref  = np.array([ XB[:,1] - XB[:,0] , 
                           XB[:,0]           ])
        Yref  = np.array([ np.max(Y,0)-np.min(Y,0) , 
                           np.min(Y,0)             ])
        DYref = np.array([ Yref[0]/Xref[0,:] , 
                           np.zeros([nx]) ])

        # build scaling functions
        XB_scaling = LinearFunction( Xref[0,:,None] , Xref[1,:,None] )
        X_scaling  = LinearFunction( Xref[None,0,:] , Xref[None,1,:] )
        Y_scaling  = LinearFunction( Yref[0], Yref[1] )
        DY_scaling = LinearFunction( DYref[None,0,:]  )
        
        # set scaling data keys
        self['XB']     = XB_scaling
        self['X']      = X_scaling
        self['Y']      = Y_scaling
        self['DY']     = DY_scaling
        self['XI']     = X_scaling
        self['YI']     = Y_scaling
        self['DYI']    = DY_scaling
        self['CovYI']  = Y_scaling
        self['CovDYI'] = DY_scaling
        
        return
    
    #: def calc_scaling()
    
    def wrap_function(self,function):
        return Scaled_Function(self,function)
    
    
class Scaled_Function(object):
    def __init__(self,Scaling,function):
        self.Scaling = Scaling
        self.function = function
        
    def __call__(self,X):
        
        Scaling = self.Scaling
        function = self.function
        
        X = Scaling.X.unset_scaling(X)
        Y = function(X)
        Y = Scaling.Y.set_scaling(Y)
        
        return Y
    
    
    
    #def center_ci(self):
        #''' Translate Y's scaling function to satisfy 
            #C*(X)=0 when C(X)=0
        #'''
        
        ## current scaling functions
        #Y_set   = self.Y_set
        #Y_unset = self.Y_unset
        ## rename
        #self.C_set   = Y_set
        #self.C_unset = Y_unset
        ## center scaled data on 0.0
        #C_set   = lambda(Z): Y_set(Z) - Y_set(0.0)
        #C_unset = lambda(Z): Y_unset( Z + Y_set(0.0) )
        ## store
        #self.Y_set   = C_set
        #self.Y_unset = C_unset
        
        #return
    
    ##: def center_ci()
    
    #def uncenter_ci(self):
        
        #self.Y_set   = self.C_set
        #self.Y_unset = self.C_unset
        
        #return
        
    ##: def uncenter_ci()
 