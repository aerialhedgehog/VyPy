
# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------  

from VyPy.data import HashedDict
import pickle
from copy import deepcopy
from time import time, sleep
import numpy as np

# ----------------------------------------------------------------------        
#   Main
# ----------------------------------------------------------------------  

def main():

    # --------------------------------------------------------
    #   Initialize
    # --------------------------------------------------------
    
    cache = HashedDict()

    
    # --------------------------------------------------------
    #   Load up data
    # --------------------------------------------------------
    
    cache['a'] = 1      # normal dictionary keys are strings
    cache[[1,2,3]] = 2  # HashedDict accepts lists for example
    cache[[1,2,5]] = 5    
    
    funny_key = object()
    
    cache[[6,2,5]] = HashedDict()  # sub-dictionary
    cache[[6,2,5]][funny_key] = 77


    # --------------------------------------------------------    
    #   Printing
    # --------------------------------------------------------
    
    print '>>> print cache'
    print cache
    
    print '>>> print cache[[1,2,3]]'
    print cache[[1,2,3]] 
    print ''

    print '>>> print cache[(1,2,3)]'
    print cache[(1,2,3)] 
    print ''
    
    print 'should be True:' , cache.has_key([1,2,3])
    assert cache.has_key([1,2,3])
    
    print 'should be True:' , [1,2,3] in cache
    assert [1,2,3] in cache
    
    del cache[[1,2,3]]
    print 'should be False:' , cache.has_key([1,2,3])
    assert not cache.has_key([1,2,3])
    print ''
    
    
    # --------------------------------------------------------
    #   Pickling test
    # --------------------------------------------------------
    
    print '>>> pickle.dumps()'
    d = pickle.dumps(cache)
    print '>>> pickle.loads()'
    p = pickle.loads(d)
    print ''
    
    print '>>> print p'
    print p    

    print 'should be True:' , [1,2,5] in p    
    assert [1,2,5] in p  
    
    # beware after pickling some objects...
    print 'should be False:' , funny_key in p[[6,2,5]]
    assert not funny_key in p[[6,2,5]]
    print ''
    
    
    # --------------------------------------------------------
    #   Access Speed test
    # --------------------------------------------------------
    print 'Access speed test...'
    
    # accessing bunch
    t0 = time()
    for i in range(int(1e5)):
        v = cache[[6,2,5]][funny_key]
    t1 = time()-t0
    
    # a test dictionary
    z = dict()
    z['t'] = dict()
    z['t']['i'] = 0
    
    # accessing a normal dictionary
    t0 = time()
    for i in range(int(1e5)):
        v = z['t']['i']
    t2 = time()-t0
    
    # results
    print 'HashedDict: %.6f s' % (t1)
    print 'dict:       %.6f s' % (t2)    
    assert (t1-t2)/t2 < 60.0
    print ''
    
    
    # --------------------------------------------------------
    #   Assignment Speed test
    # --------------------------------------------------------
    print 'Assignment speed test...'
    
    # accessing bunch
    t0 = time()
    for i in range(int(1e5)):
        v = cache[[6,2,5]][funny_key] = 10
    t1 = time()-t0
    
    # accessing a normal dictionary
    t0 = time()
    for i in range(int(1e5)):
        z['t']['i'] = 10
    t2 = time()-t0
    
    # results
    print 'HashedDict: %.6f s' % (t1)
    print 'dict:       %.6f s' % (t2)   
    assert (t1-t2)/t2 < 60.0
    print ''
    
    
# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------  

if __name__ == '__main__':
    main()