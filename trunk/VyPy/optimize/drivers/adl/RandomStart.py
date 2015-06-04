
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import VyPy
from VyPy.data import ibunch
from VyPy.optimize.drivers import Driver
import numpy as np
from time import time
from VyPy.exceptions import MaxEvaluations

try:
    import scipy
    import scipy.optimize  
except ImportError:
    pass


# ----------------------------------------------------------------------
#   Sequential Least Squares Quadratic Programming
# ----------------------------------------------------------------------

class RandomStart(Driver):
    def __init__(self):
        ''' see 
        '''
        
        Driver.__init__(self)
        
        self.driver = None
        self.number_starts = 10
        
    def run(self,problem):
        
        NS = self.number_starts
        
        bounds = problem.variables.bounds()
        XB = np.hstack(bounds).T
        XS = VyPy.sampling.lhc_uniform(XB,NS)
        
        best_result = None
        best_obj = np.inf
        obj_scale = problem.objectives[0].scale
        
        for X in XS:
            
            V = problem.variables.unpack_array(X)
            
            problem.variables.set(initials=V.values())
            
            result = self.driver.run(problem)    
            
            obj = result.objectives[0] / obj_scale
             
            if best_result is None:
                best_result = result
            elif obj<best_obj:
                best_obj = obj
                best_result = result
                
        return best_result
                