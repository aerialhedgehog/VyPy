
# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------  

from VyPy.data import OrderedDict, Property
import pickle
from time import time, sleep
from copy import deepcopy
import numpy as np

# ----------------------------------------------------------------------        
#   Main
# ----------------------------------------------------------------------  

def main():

    # --------------------------------------------------------
    #   Initialize
    # --------------------------------------------------------
    
    o = OrderedDict()

    
    # --------------------------------------------------------
    #   Load up data
    # --------------------------------------------------------
    
    o['x'] = 'hello'       # dictionary style
    o['y'] = 1           
    o['z'] = [3,4,5]
    
    o['t'] = OrderedDict() # sub-dictionary
    o['t']['h'] = 20
    o['t']['i'] = (1,2,3)
    
    
    # --------------------------------------------------------
    #   Attach a callable object
    # --------------------------------------------------------
    
    #o['f'] = Callable(test_function,o)


    # --------------------------------------------------------    
    #   Printing
    # --------------------------------------------------------
    
    print '>>> print o.keys()'
    print o.keys()
    print ''
    
    print '>>> print o'
    print o
    
    #print ">>> print o['f']()"
    #print o['f']()
    #print ''
    
    
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
    #print "should be true:" , p['f'].d is p
    #assert p['f'].d is p
    
    
    # --------------------------------------------------------
    #   The update function
    # --------------------------------------------------------
    
    o['t']['h'] = 'changed'
    p.update(o)
    
    print "should be 'changed':" , p['t']['h']
    assert p['t']['h'] == 'changed'
    #assert p['f'].d['t']['h'] == 'changed'
    print ''
    
    
    # --------------------------------------------------------
    #   Array Manipulation
    # --------------------------------------------------------

    # an ordered dictionary of floats
    a = OrderedDict()
    a['f'] = 1
    a['g'] = 2
    a['b'] = OrderedDict()
    a['b']['h'] = np.array([1,2,3])
    a['n'] = 'strings ignored'
    
    print '>>> print a'
    print a
    print ''
    
    # dump the numerical data to an array
    print '>>> a.pack_array()'
    c = a.pack_array()
    print c
    print ''
    
    # modify array
    print '>>> modify c[2]'
    c[2] = 25
    print c
    print ''
    
    # repack dictionary
    a.unpack_array(c)
    print '>>> a.unpack_array(c)'
    print a
    print ''    

    # make a copy
    b = deepcopy(a)
    
    # a method to do recursivlely on both a and b
    def method(self,other):
        try:    return self-other  
        except: return None       # ignore strings or failed operations
    
    d = a.do_recursive(method,b)
    
    print ">>> recursive a-b"
    print d
    print ''
    
    return # `````!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # --------------------------------------------------------
    #   Access Speed test
    # --------------------------------------------------------
    print 'Access speed test...'
    
    # accessing bunch
    t0 = time()
    for i in range(int(1e6)):
        v = o['t']['i']
    t1 = time()-t0
    
    # a test dictionary
    z = dict()
    z['t'] = dict()
    z['t']['i'] = 0
    
    # accessing a normal dictionary
    t0 = time()
    for i in range(int(1e6)):
        v = z['t']['i']
    t2 = time()-t0
    
    # results
    print 'OrderedDict: %.6f s' % (t1)
    print 'dict:        %.6f s' % (t2)    
    assert (t1-t2)/t2 < 0.5
    print ''
    
    
    # --------------------------------------------------------
    #   Assignment Speed test
    # --------------------------------------------------------
    print 'Assignment speed test...'
    
    # accessing bunch
    t0 = time()
    for i in range(int(1e6)):
        o['t']['i'] = 10
    t1 = time()-t0
        
    # accessing a normal dictionary
    t0 = time()
    for i in range(int(1e6)):
        z['t']['i'] = 10
    t2 = time()-t0
    
    # results
    print 'OrderedDict: %.6f s' % (t1)
    print 'dict:        %.6f s' % (t2)       
    assert (t1-t2)/t2 < 5.0
    print ''
    
    
## ----------------------------------------------------------------------        
##   Callable Object 
## ----------------------------------------------------------------------  

## has a hidden property
## works like a decorator

#class Callable(OrderedDict):
    #def __init__(self,f,d):
        #self['f'] = f
        #self.d    = d
    #def __call__(self,*x):
        #return self['f'](self.d,*x)

## ----------------------------------------------------------------------        
##   Test Function
## ----------------------------------------------------------------------  

## to work in the callable object

#def test_function(c):
    #return c['x']

    
# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------  

if __name__ == '__main__':
    main()