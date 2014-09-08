
# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------  

from VyPy.data import Bunch, Property
import pickle
from time import time, sleep

# ----------------------------------------------------------------------        
#   Main
# ----------------------------------------------------------------------  

def main():

    # --------------------------------------------------------
    #   Initialize
    # --------------------------------------------------------
    
    o = Bunch()

    
    # --------------------------------------------------------
    #   Load up data
    # --------------------------------------------------------
    
    o['x'] = 'hello'      # dictionary style
    o.y    = 1            # attribute style
    o['z'] = [3,4,5]
    
    o.t = Bunch()         # sub-bunch
    o.t['h'] = 20
    o.t.i = (1,2,3)
    
    
    # --------------------------------------------------------
    #   Attach a callable object
    # --------------------------------------------------------
    
    o.f = Callable(test_function,o)


    # --------------------------------------------------------    
    #   Printing
    # --------------------------------------------------------
    
    print '>>> print o.keys()'
    print o.keys()
    print ''
    
    print '>>> print o'
    print o
    
    print '>>> print o.f()'
    print o.f()
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
    print "should be true:" , p.f.d is p
    assert p.f.d is p
    
    
    # --------------------------------------------------------
    #   The update function
    # --------------------------------------------------------

    o.t['h'] = 'changed'
    p.update(o)
    
    print "should be 'changed':" , p.t.h
    assert p.t.h == 'changed'
    assert p.f.d.t.h == 'changed'
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
    print 'Bunch:       %.6f s' % (t1)
    print 'SimpleBunch: %.6f s' % (t2)    
    assert (t1-t2)/t2 < 0.3
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
    print 'Bunch:       %.6f s' % (t1)
    print 'SimpleBunch: %.6f s' % (t2)    
    assert (t1-t2)/t2 < 0.3
    print ''
    
    
# ----------------------------------------------------------------------        
#   Callable Object 
# ----------------------------------------------------------------------  

# has a hidden property
# works like a decorator

class Callable(Bunch):
    d = Property('d')
    def __init__(self,f,d):
        self.f = f
        self.d = d
    def __call__(self,*x):
        return self.f(self.d,*x)

# ----------------------------------------------------------------------        
#   Test Function
# ----------------------------------------------------------------------  

# to work in the callable object

def test_function(c):
    return c.x    

    
# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------  

if __name__ == '__main__':
    main()