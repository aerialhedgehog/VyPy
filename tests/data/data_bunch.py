
# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------  

from VyPy.data import DataBunch
import pickle
from copy import deepcopy
from time import time, sleep
import numpy as np


# ----------------------------------------------------------------------        
#   Floor Data Type
# ----------------------------------------------------------------------  

class Floor(DataBunch):
    
    def __defaults__(self):
        
        self.pattern = 'checkered'
        self.value   = 100.
        
        #self.is_carpet = True  # this should be handled with a subclass
        
    def method(self,factor):
        self.value = self.value * factor
        return factor
        

# ----------------------------------------------------------------------        
#   Carpet Floor Data Type
# ----------------------------------------------------------------------  

class Carpet(Floor):
    def __defaults__(self):
        
        # self.pattern = 'checkered'     # will be set by Floor's defaults
        self.value = 200                 # overrides Floor's defaults
        self.shaggyness = 9
        

# ----------------------------------------------------------------------        
#   Main
# ----------------------------------------------------------------------  

def main():

    # --------------------------------------------------------
    #   Initialize
    # --------------------------------------------------------
    
    carpet = Carpet()


    # --------------------------------------------------------    
    #   Printing
    # --------------------------------------------------------
    
    print '>>> print carpet'
    print carpet
    
    print '>>> print carpet.value'      # attribute style
    print carpet.value 
    print ''
    
    print ">>> carpet.method(10)"
    print carpet.method(10)
    print ''
    
    print ">>> print carpet['value']"   # dictionary style
    print carpet['value']
    print ''    
    
    print ">>> isinstance(carpet,Floor)"
    print isinstance(carpet,Floor)


    # --------------------------------------------------------
    #   Load up more data
    # --------------------------------------------------------
    
    carpet['x'] = 'hello'  # dictionary style
    carpet.y    = 1        # attribute style
    carpet['z'] = None
    
    carpet[2]   = [3,4,5]  # index style access or reassignment 
                           # (must name the item with a string)
    
    carpet.t = DataBunch()  # sub data, no type
    carpet.t['h'] = 20
    carpet.t.i = (1,2,3)
    
    print '>>> print carpet'
    print carpet
    
    # the data is ordered
    print '>>> iterate carpet.items()'
    for k,v in carpet.items():
        if isinstance(v,DataBunch): v = 'DataBunch()'
        print k , ':' , v
    print ''
    
    
    # --------------------------------------------------------
    #   Pickling test
    # --------------------------------------------------------
    
    print '>>> pickle.dumps()'
    d = pickle.dumps(carpet)
    print '>>> pickle.loads()'
    p = pickle.loads(d)
    print ''
    
    print '>>> print p'
    print p    
    
    
    # --------------------------------------------------------
    #   Access Speed test
    # --------------------------------------------------------
    print 'Access speed test...'
    
    # accessing bunch
    t0 = time()
    for i in range(int(1e6)):
        v = carpet.t.i
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
    print 'DataBunch:    %.6f s' % (t1)
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
        carpet.t.i = 10
    t1 = time()-t0
    
    # accessing a simpler bunch
    t0 = time()
    for i in range(int(1e6)):
        z.t.i = 10
    t2 = time()-t0
    
    # results
    print 'DataBunch:    %.6f s' % (t1)
    print 'SimpleBunch:  %.6f s' % (t2)    
    assert (t1-t2)/t2 < 5.0
    print ''
    
    
# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------  

if __name__ == '__main__':
    main()