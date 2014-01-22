#!/usr/bin/env python

""" Bunch is a subclass of dict with attribute-style access.
    
    It is safe to import * from this module:
    
        __all__ = ('Bunch', 'bunchify','unbunchify')
    
    un/bunchify provide dictionary conversion; Bunches can also be
    converted via Bunch.to/fromDict().
    
    original source:
    https://pypi.python.org/pypi/bunch
"""

from Bunch import Bunch
from OrderedDict import OrderedDict

class OrderedBunch(OrderedDict,Bunch):
    """ An ordered dictionary that provides attribute-style access.
    """
    pass


if __name__ == '__main__':
    
    class TestDescriptor(object):
        def __init__(self,x):
            self.x = x
        
        def __get__(self,obj,kls=None):
            print '__get__'
            print type(obj), type(self)
            print self.x
            return self.x
        
        def __set__(self,obj,val):
            print '__set__'
            print type(obj), type(self)
            print val
            self.x = val
        
    class TestObject(OrderedBunch):
        pass
    
    o = TestObject()
    o['x'] = TestDescriptor([1,2,3])
    o.y = 1
    for i in range(10):
        o['x%i'%i] = 'yo'
    
    print ''
    print o['x']
    print o.y
    
    print ''
    o['x'] = [3,4,5]
            
            
    import pickle
        
    d = pickle.dumps(o)
    p = pickle.loads(d)
    
    print ''
    print p['x']
    print p