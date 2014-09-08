
# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------  

from VyPy.data import Object
import pickle
from copy import deepcopy
from time import time, sleep
import numpy as np

        
# ----------------------------------------------------------------------        
#   Test Object
# ----------------------------------------------------------------------  
    
class TestObject(Object):
    def __init__(self):
        self.x = None


# ----------------------------------------------------------------------        
#   Test Descriptor
# ----------------------------------------------------------------------  

class TestDescriptor(object):
    def __init__(self,x):
        self.x = x
    
    def __get__(self,obj,kls=None):
        print '__get__:' , self.x
        return self.x
    
    def __set__(self,obj,val):
        print '__set__:' , val
        self.x = val


# ----------------------------------------------------------------------        
#   Main
# ----------------------------------------------------------------------  

def main():

    # --------------------------------------------------------
    #   Initialize
    # --------------------------------------------------------
    
    # the object
    o = TestObject()
    
    # the descriptor
    d = TestDescriptor([1,2,3])

    # assign the descriptor
    # this is different from the traditional python descriptor,
    # since we are late-binding the descritor
    o.x = d
    
    # the cost shows up in performance, access and assignment
    # is much slower (~3x)

    # --------------------------------------------------------    
    #   Access
    # --------------------------------------------------------    
    
    print ">>> o.x"
    print o.x
    print ""
    
    # --------------------------------------------------------    
    #   Assignment
    # --------------------------------------------------------        
    
    print ">>> o.x = [3,4,5]"
    o.x = [3,4,5]
    print ""
    
    
    # --------------------------------------------------------
    #   Pickling test
    # --------------------------------------------------------
    
    print '>>> pickle.dumps()'
    s = pickle.dumps(o)
    print '>>> pickle.loads()'
    p = pickle.loads(s)
    print ''
    
    print '>>> print p.x'
    print p.x
    print ''
    
    # --------------------------------------------------------
    #   Access Speed test
    # --------------------------------------------------------
    print 'Access speed test...'
    
    class SimpleDescriptor(object):
        def __init__(self,x):
            self.x = x
        def __get__(self,obj,kls=None):
            return self.x
        def __set__(self,obj,val):
            self.x = val        
    
    o.y = SimpleDescriptor(10)
    
    # accessing bunch
    t0 = time()
    for i in range(int(1e6)):
        v = o.y
    t1 = time()-t0
    
    # accessing a simpler bunch
    class SimpleBunch(object):
        t = SimpleDescriptor(10)
        
    z = SimpleBunch()
    z.t = 10
    
    t0 = time()
    for i in range(int(1e6)):
        v = z.t
    t2 = time()-t0
    
    # results
    print 'Object:       %.6f s' % (t1)
    print 'SimpleBunch:  %.6f s' % (t2)    
    assert (t1-t2)/t2 < 3.0
    print ''
    
    
    # --------------------------------------------------------
    #   Assignment Speed test
    # --------------------------------------------------------
    print 'Assignment speed test...'
    
    # accessing bunch
    t0 = time()
    for i in range(int(1e6)):
        o.y = 10
    t1 = time()-t0
    
    # accessing a simpler bunch
    t0 = time()
    for i in range(int(1e6)):
        z.t = 10
    t2 = time()-t0
    
    # results
    print 'Object:       %.6f s' % (t1)
    print 'SimpleBunch:  %.6f s' % (t2)    
    assert (t1-t2)/t2 < 3.0
    print ''
       
    
# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------  

if __name__ == '__main__':
    main()