

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------  

import numpy as np
import os, sys, shutil, time

import VyPy
from VyPy import parallel as para

tic = time.time()


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
    
    function = para.Remember(function)
    print function    
    
    function = para.MultiTask(function,copies=4)
    print function

    print 'call multitask, 1 job'
    result = function(1.)
    print result
    print ''
    
    print 'call multitask, 3 jobs'
    result = function([1.,2.,3.])
    print result
    print ''
    


def test_func(x):
    
    print 'function start'
    print 't = %.4f' % (time.time()-tic)
    print 'x =',x
    sys.stdout.flush()
    
    y = x*2.
    print 'function wait ...'
    sys.stdout.flush()
    time.sleep(1.0)
    
    print 'function done'
    print 't = %.4f' % (time.time()-tic)
    print 'y =',y
    
    return y


# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------  

if __name__ == '__main__':
    main()
