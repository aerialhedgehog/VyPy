
# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------  

from VyPy.data import IndexableBunch, Property
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
    
    o = IndexableBunch()

    
    # --------------------------------------------------------
    #   Load up data
    # --------------------------------------------------------
    
    o['x'] = 'hello'  # dictionary style
    o.y    = 1        # attribute style
    o['z'] = None
    
    o[2]   = [3,4,5]  # index style access or reassignment 
                      # (must name the item with a string)
    
    o.t = IndexableBunch()
    o.t['h'] = 20
    o.t.i = (1,2,3)
    

    # --------------------------------------------------------    
    #   Printing
    # --------------------------------------------------------
    
    print '>>> print o'
    print o
    
    
    print '>>> print o[0]'
    print o[0]    
    print ''
    
    # --------------------------------------------------------
    #   Pickling test
    # --------------------------------------------------------
    
    print '>>> pickle.dumps()'
    d = pickle.dumps(o)
    print '>>> pickle.loads()'
    p = pickle.loads(d)
    print ''
    
    print '>>> print p'
    print p    
    
    
    # --------------------------------------------------------
    #   The update function
    # --------------------------------------------------------
    
    o.t[0] = 'changed'
    p.update(o)

    print "should be 'changed':" , p.t.h
    assert p.t.h == 'changed'
    print ''
    
    
    # --------------------------------------------------------
    #   Access Speed test
    # --------------------------------------------------------
    print 'Access speed test...'
    
    # accessing bunch
    t0 = time()
    for i in range(int(1e6)):
        v = o.t.i
    t1 = time()-t0
    
    # accessing a simpler bunch
    class SimpleBunch:
        pass
    z = SimpleBunch()
    z.t = SimpleBunch
    z.t.i = 0
    
    t0 = time()
    for i in range(int(1e6)):
        v = z.t.i
    t2 = time()-t0
    
    # results
    print 'OrderedBunch: %.6f s' % (t1)
    print 'SimpleBunch:  %.6f s' % (t2)    
    assert (t1-t2)/t2 < 0.5
    print ''
    
    
    # --------------------------------------------------------
    #   Assignment Speed test
    # --------------------------------------------------------
    print 'Assignment speed test...'
    
    # accessing bunch
    t0 = time()
    for i in range(int(1e6)):
        o.t.i = 10
    t1 = time()-t0
    
    # accessing a simpler bunch
    t0 = time()
    for i in range(int(1e6)):
        z.t.i = 10
    t2 = time()-t0
    
    # results
    print 'OrderedBunch: %.6f s' % (t1)
    print 'SimpleBunch:  %.6f s' % (t2)    
    assert (t1-t2)/t2 < 5.0
    print ''
    
    
# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------  

if __name__ == '__main__':
    main()