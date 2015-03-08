#!/usr/bin/env python


from Dict import Dict

#from lib.python_Bunch import Bunch as _Bunch
from lib.cython_Bunch import Bunch as _Bunch

#try:
    #from lib.cython_Bunch import Bunch as _Bunch
#except ImportError:
    #from lib.python_Bunch import Bunch as _Bunch

class Bunch(_Bunch,Dict):
    
    update   = Dict.update
    __repr__ = Dict.__repr__
    __str__  = Dict.__str__


if __name__ == '__main__':
    
    o = Bunch()
    o['x'] = 'hello'
    o.y = 1
    o['z'] = [3,4,5]
    o.t = Bunch()
    o.t['h'] = 20
    o.t.i = (1,2,3)
    
    print o

    import pickle

    d = pickle.dumps(o)
    p = pickle.loads(d)
    
    print ''
    print p    
    
    o.t['h'] = 'changed'
    p.update(o)
    p['t'].update(o)

    print ''
    print p
    
    #class TestDescriptor(object):
        #def __init__(self,x):
            #self.x = x
        
        #def __get__(self,obj,kls=None):
            #print '__get__'
            #print type(obj), type(self)
            #print self.x
            #return self.x
        
        #def __set__(self,obj,val):
            #print '__set__'
            #print type(obj), type(self)
            #print val
            #self.x = val
        
    #class TestObject(Bunch):
        #pass
    
    #o = TestObject()
    #o['x'] = TestDescriptor([1,2,3])
    #o['y'] = 1
    
    #print ''
    #print o['x']
    #print o.y
    #print o
    
    #print ''
    #o['x'] = [3,4,5]
            
            
    #import pickle
        
    #d = pickle.dumps(o)
    #p = pickle.loads(d)
    
    #print ''
    #print p['x']
    #print p.y