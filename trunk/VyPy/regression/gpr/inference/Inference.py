
import numpy as np
import scipy as sp

from VyPy import tools
from VyPy.exceptions import EvaluationFailure
from VyPy.tools import atleast_2d

class Inference(object):
    
    def __init__(self,Kernel):
        
        self.Kernel = Kernel
        self.Train  = Kernel.Train
        
        return
    
    def __call__(self,XI):
        return self.predict(XI)
    
    def precalc(self):
        return
    
    def predict(self,XI):
        ''' Evaluate GPR model at XI
        '''
        
        raise NotImplementedError
        
        # unpack
        Kernel = self.Kernel
        XI     = atleast_2d(XI)
        
        # process
        ## CODE
        
        # results
        YI_solve   = 0 # predicted output
        CovI_solve = 0 # predicted covariance
        
        # pack up outputs 
        try:
            data = Kernel.pack_outputs(XI,YI_solve,CovI_solve)
        except NotImplementedError:
            data = [YI_solve,CovI_solve]
        
        return data
    
    #: def predict()
    