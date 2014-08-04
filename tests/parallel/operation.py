

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------  

import numpy as np
import os, sys, shutil, time

import VyPy
from VyPy import parallel as para


# ----------------------------------------------------------------------        
#   Main
# ----------------------------------------------------------------------  

def main():
    
    function = test_func
    print function
    
    resource = para.Resource.ComputeCores(max_cores=4)
    print resource
    
    gate = resource.gate(default=2)
    print gate
    
    function = para.Operation(function,gate)
    print function
    print ''
    
    print 'call operation'
    result = function(10.)
    print 'result:',result
    

# ----------------------------------------------------------------------        
#   Test Function
# ----------------------------------------------------------------------  

def test_func(x):
    y = x*2.
    print 'function wait ...'
    sys.stdout.flush()
    time.sleep(1.0)
    print 'x:',x
    print 'y:',y
    print 'function done'
    return y


# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------  

if __name__ == '__main__':
    main()