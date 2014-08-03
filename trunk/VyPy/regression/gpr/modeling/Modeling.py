
import sys, copy

import numpy as np
import scipy as sp
import scipy.linalg

from VyPy.exceptions import EvaluationFailure
from VyPy.data import IndexableDict
from VyPy.tools import vector_distance, atleast_2d

class Modeling(object):
    
    def __init__(self,Learn,Scaling=None):
        
        # pack
        self.Learn   = Learn
        self.Infer   = Learn.Infer
        self.Kernel  = Learn.Kernel
        self.Train   = Learn.Train
        self.Hypers  = Learn.Hypers
        self.Scaling = Scaling
        
        # useful data
        self._current = False

        # try to precalc Kernel
        ##self.safe_precalc()
        
        return
    
    #: def __init__()

    
    def predict(self,XI):
        
        if not self.Scaling is None:
            XI = self.Scaling.X.set_scaling(XI)
            
        # will skip if current
        self.safe_precalc()
        
        # prediction
        data = self.Infer.predict(XI)
        
        if not self.Scaling is None:
            data = self.Scaling.unset_scaling(data)                
        
        return data
    
    def predict_YI(self,XI):
        return self.predict(XI).YI
    
    def safe_precalc(self):
        if self._current:
            return
        
        try:
            self.precalc()
            
        except EvaluationFailure:
            try:
                self.learn()
            except EvaluationFailure:
                raise EvaluationFailure , 'could not precalculate Kernel'
        
        self._current = True
        
        return 
    
    def precalc(self):
        self.Infer.precalc()
        
    def learn(self):
        self.Learn.learn()
        
     
#: class Modeling()