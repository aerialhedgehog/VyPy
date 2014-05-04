
import sys, copy, weakref,gc

import numpy as np
import scipy as sp

from Training import Training

from VyPy.data import IndexableBunch
from VyPy.tools import atleast_2d_col, atleast_2d_row

class Training_VF(Training):
    
    def __init__(self,Trains=None):
        if not Trains is None:
            self.update(Trains)
        
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
        