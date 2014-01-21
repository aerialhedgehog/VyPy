
from Object import Object

class Dict(Object,dict):
    
    # implement descriptor protocol for items
    def __getitem__(self,k):
        try:
            return super(Dict,self).__getitem__(k).__get__(self,type(self))
        except:
            return super(Dict,self).__getitem__(k)
    
    def __setitem__(self,k,v):
        try:
            super(Dict,self).__getitem__(k).__set__(self,v)
        except:
            super(Dict,self).__setitem__(k,v)
    
    def __delitem__(self,k):
        try:
            super(Dict,self).__getitem__(k).__del__(self)
        except:
            super(Dict,self).__delitem__(k)
            
    # prettier printing
    def __repr__(self):
        """ Invertible* string-form of a Bunch.
        """
        keys = self.keys()
        args = ', '.join(['%s=%r' % (key, self[key]) for key in keys])
        return '%s(%s)' % (self.__class__.__name__, args)
    
    def __str__(self):
        """ String-form of a OrderedBunch.
        """
        args = '\n'.join(['%s : %s' % (k,v) for k,v in self.iteritems()])
        return args
                


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
        
    class TestObject(Dict):
        def __init__(self,c):
            self.c = c
    
    o = TestObject(555)
    o['x'] = TestDescriptor([1,2,3])
    o['y'] = 1
    o.desc = TestDescriptor([5,7,8])
    
    print ''
    print o['x']
    print o['y']
    print o.desc
    print o.c
    
    print ''
    o['x'] = [3,4,5]
            
            
    import pickle
        
    d = pickle.dumps(o)
    p = pickle.loads(d)
    
    print ''
    print p['x']
    print p['y']
    print o.c
    print o.desc