
from IndexableDict import IndexableDict
    
from make_hashable import make_hashable


class HashedDict(IndexableDict):
    
    def __getitem__(self,k):
        _k = make_hashable(k) 
        return super(HashedDict,self).__getitem__(_k)
        #raise KeyError , ('Key does not exist: %s' % k)
        
    def __setitem__(self,k,v):
        _k = make_hashable(k)
        super(HashedDict,self).__setitem__(_k,v)
        
    def __delitem__(self,k):
        _k = make_hashable(k)
        super(HashedDict,self).__delitem__(_k)
        
    def __contains__(self,k):
        _k = make_hashable(k)
        return super(HashedDict,self).__contains__(_k)
    
    def has_key(self,k):
        _k = make_hashable(k)
        return super(HashedDict,self).has_key(_k)
    
    




if __name__ == '__main__':
    
    # tests
    print 'Testing make_hash.py'
    cache = HashedDict()
    cache['a'] = 1
    cache[[1,2,3]] = 2
    print 'should be True:' , cache.has_key([1,2,3])
    print 'should be True:' , [1,2,3] in cache
    del cache[[1,2,3]]
    print 'should be False:' , cache.has_key([1,2,3])
    
        
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
        
    class TestObject(HashedDict):
        def __init__(self,c):
            self.c = c
    
    o = TestObject('456')
    o['x'] = TestDescriptor([1,2,3])
    o['y'] = TestDescriptor([4,3,5])
    o.desc = TestDescriptor([5,7,8])
    
    print ''
    print o
    print o.desc
    print o.c
    
    print ''
    o['x'] = [3,4,5]
    
    print ''
    print o[0]
            
            
    import pickle
        
    d = pickle.dumps(o)
    p = pickle.loads(d)
    
    print ''
    print p[1]
    print p.c
    print p.desc
    
    