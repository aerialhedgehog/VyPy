#!/usr/bin/env python

""" Bunch is a subclass of dict with attribute-style access.
    
    It is safe to import * from this module:
    
        __all__ = ('Bunch', 'bunchify','unbunchify')
    
    un/bunchify provide dictionary conversion; Bunches can also be
    converted via Bunch.to/fromDict().
    
    original source:
    https://pypi.python.org/pypi/bunch
"""

from Dict import Dict

class Bunch(Dict):
    """ A dictionary that provides attribute-style access.
    """
    
    def __contains__(self, k):
        try:
            return hasattr(self, k) or dict.__contains__(self, k)
        except:
            return False
        
    # only called if k not found in normal places 
    def __getattr__(self, k):
        """ Gets key if it exists, otherwise throws AttributeError.
        """
        try:
            # Throws exception if not in prototype chain
            return super(Bunch,self).__getattribute__(k)
        except AttributeError:
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)
    
    def __setattr__(self, k, v):
        """ Sets attribute k if it exists, otherwise sets key k. A KeyError
            raised by set-item (only likely if you subclass Bunch) will 
            propagate as an AttributeError instead.
        """
        try:
            # Throws exception if not in prototype chain
            super(Bunch,self).__getattribute__(k)
        except AttributeError:
            try:
                self[k] = v
            except:
                raise AttributeError(k)
        else:
            super(Bunch,self).__setattr__(k, v)
    
    def __delattr__(self, k):
        """ Deletes attribute k if it exists, otherwise deletes key k. A KeyError
            raised by deleting the key--such as when the key is missing--will
            propagate as an AttributeError instead.
        """
        try:
            # Throws exception if not in prototype chain
            super(Bunch,self).__getattribute__(k)
        except AttributeError:
            try:
                del self[k]
            except KeyError:
                raise AttributeError(k)
        else:
            super(Bunch,self).__delattr__(k)
        

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