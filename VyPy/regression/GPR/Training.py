
import sys, copy, weakref,gc

import numpy as np
import scipy as sp

from VyPy.data import IndexableBunch
from VyPy.tools import check_array

def Training(XB,X,Y,DY=None):
    """ Class Factory for Training Data """
    
    NewCls = TrainingBunch()
    
    # make sure all are 2D arrays
    XB = check_array(XB)
    X  = check_array(X)
    Y  = check_array(Y,'col')
    
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
        DY = check_array(DY)
        ntdy,ndy = DY.shape
        assert ntx == ntdy  , 'different training data X and target gradients DY'
        
    # make sure data is of type float
    X  = X. astype(float)
    Y  = Y. astype(float)
    DY = DY.astype(float)
    XB = XB.astype(float)

    # load Training Class
    NewCls['X']  = X
    NewCls['Y']  = Y
    NewCls['DY'] = DY
    NewCls['XB'] = XB
        
    return NewCls

class TrainingBunch(IndexableBunch):  
    ''' '''
    # i like trains!

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


def VF_Training(Trains=None):
    
    Train = VF_TrainingBunch()
    
    if not Trains is None:
        Train.update(Trains)
    
    return Train

class VF_TrainingBunch(IndexableBunch):
    
    # --------------------------------------------------------
    #   Train.X
    # --------------------------------------------------------   
    @property
    def X(self):
        Xs = []
        for Train in self.values():
            Xs.append(Train.X)
        Xs = np.vstack(Xs)
        return Xs
    @X.setter
    def X(self,val):
        x1 = 0
        for Train in self.values():
            nx = Train.X.shape[0]
            x2 = x1 + nx
            Train.X = val[x1:x2,:]
            x1 = x2
        assert x2 == val.shape[0]
        return
    
    # --------------------------------------------------------
    #   Train.Y
    # --------------------------------------------------------
    @property
    def Y(self):
        Ys = []
        for Train in self.values():
            Ys.append(Train.Y)
        Ys = np.vstack(Ys)
        return Ys
    @Y.setter
    def Y(self,val):
        x1 = 0
        for Train in self.values():
            nx = Train.Y.shape[0]
            x2 = x1 + nx
            Train.Y = val[x1:x2,:]
            x1 = x2
        assert x2 == val.shape[0]
        return
    
    # --------------------------------------------------------
    #   Train.DY
    # --------------------------------------------------------
    @property
    def DY(self):
        DYs = []
        for Train in self.values():
            DYs.append(Train.DY)
        DYs = np.vstack(DYs)
        return DYs
    @DY.setter
    def DY(self,val):
        x1 = 0
        for Train in self.values():
            nx = Train.DY.shape[0]
            x2 = x1 + nx
            Train.DY = val[x1:x2,:]
            x1 = x2
        assert x2 == val.shape[0]
        return
    
    # --------------------------------------------------------
    #   Train.XB
    # --------------------------------------------------------
    @property
    def XB(self):
        XBs = self[0].XB
        for Train in self.values():
            assert (Train.XB == XBs).all()
        return XBs
    @XB.setter
    def XB(self,val):
        for Train in self.values():
            Train.XB = val
        return
    
    def best(self):
        raise NotImplementedError
    
    def append(self):
        raise NotImplementedError
        