
from ScalingFunction import ScalingFunction

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import numpy as np


# ----------------------------------------------------------------------
#   Logarithmic Scaling Function
# ----------------------------------------------------------------------

class Logarithmic(ScalingFunction):
    
    def __init__(self,scale=1.0,base=10.0):
        """ o / scl ==> np.log_base(other/scale)
            o * scl ==> (base**other) * scale
            
            base defualt to 10.0
            base could be numpy.e for example
        """
        self.scale = scale
        self.base   = base
        
    def set_scaling(self,other):
        return np.log10(other/self.scale)/np.log10(self.base)
    def unset_scaling(self,other):
        return (self.base**other)*self.scale
    
    
    
    
class Gradient(Logarithmic):    
    
    def __init__(self,scale_x=1.0,scale_f=1.0,base=10.0):
        
        self.scale_x = scale_x
        self.scale_f = scale_f
        self.base   = base    
    
    def set_scaling(self,grad,func):
        return 1. / ( np.log(self.base) * func ) * grad * self.scale_x
    
    def unset_scaling(self,grad,func):
        return grad * ( np.log(self.base) / self.scale_x ) \
                    * ( (self.base**func) * self.scale_f )
        
        
    
    
# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
    
if __name__ == '__main__':    
    
    import numpy as np
    
    s = Logarithmic(2.0,10.0)
    
    a = 20.0
    b = np.array([20,200,6000.])
    
    print a
    print b
    
    a = a / s    
    b = b / s
    
    print a
    print b
    
    a = a * s    
    b = b * s    
    
    print a
    print b