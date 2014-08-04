

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
        
    function = para.Operation(function)
    print function
    
    function = para.Remember(function)
    print function  
    print ''
    
    print 'call remember'
    result = function(10.)
    print 'result:',result
    print ''
    
    print 'call remember'
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