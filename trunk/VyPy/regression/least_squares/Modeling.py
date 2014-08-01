
import sys, copy, weakref,gc

import numpy as np
import scipy as sp
import scipy.linalg

from VyPy.exceptions import EvaluationFailure
from VyPy.data import IndexableDict
from VyPy.tools import vector_distance, atleast_2d

class Modeling(object):
    
    def __init__(self,Train):
        
        self.Train = Train
    
    def learn(self):
        raise NotImplementedError
    
    def predict(self,XI):
        raise NotImplementedError
    
