
from OrderedDict import OrderedDict
    
from make_hashable import make_hashable


class HashedDict(OrderedDict):
    def __getitem__(self,k):
        _k = make_hashable(k)
        try:
            return OrderedDict.__getitem__(self,_k)
        except KeyError:
            raise KeyError , ('Key does not exist: %s' % k)
    
    def __setitem__(self,k,v):
        _k = make_hashable(k)
        OrderedDict.__setitem__(self,_k,v)
    def __delitem__(self,k):
        _k = make_hashable(k)
        OrderedDict.__delitem__(self,_k)
    def __contains__(self,k):
        _k = make_hashable(k)
        return OrderedDict.__contains__(self,_k)
    def has_key(self,k):
        _k = make_hashable(k)
        return OrderedDict.has_key(self,_k)
    
    




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
    
        