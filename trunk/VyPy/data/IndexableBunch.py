
from Bunch import Bunch
from IndexableDict import IndexableDict

class IndexableBunch(IndexableDict,Bunch):
    """ An ordered indexable dictionary that provides attribute-style access.
    """
    pass


if __name__ == '__main__':
    
    o = IndexableBunch()
    o['x'] = 'hello'
    o.y = 1
    o['z'] = [3,4,5]
    o.t = IndexableBunch()
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
    p.t.update(o)

    print ''
    print p[3]
    
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
        
    #class TestObject(IndexableBunch):
        #def __init__(self,c):
            #self.c = c
    
    #o = TestObject([1,2,3.5,7])
    #o['x'] = TestDescriptor([1,2,3])
    #o.y = 1
    #for i in range(10):
        #o['x%i'%i] = 'yo'
    
    #print ''
    #print o['x']
    #print o.y
    #print o.c
    
    #print ''
    #o['x'] = [3,4,5]
            
            
    #import pickle
        
    #d = pickle.dumps(o)
    #p = pickle.loads(d)
    
    #print ''
    #print p['x']
    #print p[1]
    #print p
    #print p.c