
import sys, copy, weakref,gc

import numpy as np
import scipy as sp

from VyPy.data import IndexableBunch
from VyPy.tools import atleast_2d_col, atleast_2d_row

class Training(IndexableBunch):  
    ''' '''
    # i like trains!
    
    def __init__(self,XB,X,Y,DY=None):
        
        # make sure all are 2D arrays
        XB = atleast_2d_row(XB)
        X  = atleast_2d_row(X)
        Y  = atleast_2d_col(Y)
        
        # sizes
        ntx ,nx  = X.shape
        nty ,ny  = Y.shape
        assert ntx == nty        , 'different training data X and targets Y'
        assert XB.shape[0] == nx , 'different training data X and bounds XB dimension'
        assert ny == 1           , 'training targets Y must be a column vector'
        
        # handle optional gradients
        if DY is None:
            DY = np.empty([0,nx])
            ntdy,ndy = DY.shape
        else:
            DY = atleast_2d_row(DY)
            ntdy,ndy = DY.shape
            assert ntx == ntdy  , 'different training data X and target gradients DY'
            
        # make sure data is of type float
        X  = X. astype(float)
        Y  = Y. astype(float)
        DY = DY.astype(float)
        XB = XB.astype(float)
    
        # store data
        self['X']  = X
        self['Y']  = Y
        self['DY'] = DY
        self['XB'] = XB
        
        #self.Scaling = None
        

    def best(self):
        ''' find best training data
        '''
        
        # unpack
        Train = self
        X  = Train.X
        Y  = Train.Y
        DY = Train.DY
        NT = X.shape[0]
        
        # minimum's index
        I_min = np.argmin( Y )
        
        # minimum data
        X_min  = X[I_min,:]
        Y_min  = Y[I_min]
        DY_min = DY[I_min,:]
        
        # pack data
        The_Data = IndexableBunch()
        The_Data['I']  = I_min
        The_Data['X']  = X_min
        The_Data['Y']  = Y_min
        The_Data['DY'] = DY_min
            
        return The_Data
    
    #: def best()
    
    def append(self,X_new,Y_new,DY_new=None):
        ''' add data
        '''
        
        # add data
        self.X  = np.vstack([ self.X  , X_new  ])
        self.Y  = np.vstack([ self.Y  , Y_new  ])
        if not DY_new is None:
            self.DY = np.vstack([ self.DY , DY_new ])
        
        return
    
    #: def append()
    

#: class Training()
