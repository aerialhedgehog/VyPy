
import copy
from collections import OrderedDict
DictProxyType = type(object.__dict__)

try:
    import numpy as np
    array_type  = np.ndarray
    matrix_type = np.matrixlib.defmatrix.matrix
    numpy_isloaded = True
except ImportError:
    numpy_isloaded = False
    
    
    

def make_hash(o):
    """ Makes a hash from a dictionary, list, tuple or set to any level, that 
        contains only other hashable types (including any lists, tuples, sets, and
        dictionaries).  This allows these types to be used as keys for dictionaries,
        for example.
    
        In the case where other kinds of objects (like classes) need 
        to be hashed, pass in a collection of object attributes that are pertinent. 
    
        For example, a class can be hashed in this fashion:  
            make_hash([cls.__dict__, cls.__name__])
  
        A function can be hashed like so:
            make_hash([fn.__dict__, fn.__code__])
      
        original source: http://stackoverflow.com/a/8714242
    """
    
    # ints
    if isinstance(o,int):
        return o
    
    # classes and functions
    if type(o) == DictProxyType:
        o2 = {}
        for k,v in o.items():
            if not k.startswith("__"):
                o2[k] = v
        o = o2  
        
    # numpy arrays and matrices
    elif numpy_isloaded and isinstance(o,(array_type,matrix_type)):
        o = o.tolist()
    
    # sets, tuples, lists 
    if isinstance(o, (set, tuple, list)):
        return hash(tuple([make_hash(e) for e in o]))
    
    # dictionaries
    elif isinstance(o,dict):
        new_o = copy.deepcopy(o)
        for k,v in new_o.items():
            new_o[k] = make_hash(v)
        return hash(tuple(frozenset(new_o.items())))    
    
    # unhandled types
    else:
        return hash(o)
  
#: def make_hash()

def make_hashable(o):
    """ Makes a hashable item from a dictionary, list, tuple or set to any level, that 
        contains only other hashable types (including any lists, tuples, sets, and
        dictionaries).  This allows these types to be used as keys for dictionaries,
        for example.
    
        In the case where other kinds of objects (like classes) need 
        to be hashed, pass in a collection of object attributes that are pertinent. 
    
        For example, a class can be hashed in this fashion:  
            make_hash([cls.__dict__, cls.__name__])
  
        A function can be hashed like so:
            make_hash([fn.__dict__, fn.__code__])
      
        original source: http://stackoverflow.com/a/8714242
    """
    
    # already hashable
    if isinstance(o,(int,float,str,set,tuple)):
        return o
    
    # classes and functions
    if type(o) == DictProxyType:
        o2 = {}
        for k,v in o.items():
            if not k.startswith("__"):
                o2[k] = v
        o = o2  
        
    # numpy arrays and matrices
    elif numpy_isloaded and isinstance(o,(array_type,matrix_type)):
        o = o.tolist()
    
    # lists 
    if isinstance(o, list):
        return hash(tuple([make_hashable(e) for e in o]))
    
    # dictionaries
    elif isinstance(o,dict):
        new_o = copy.deepcopy(o)
        for k,v in new_o.items():
            new_o[k] = make_hashable(v)
        return tuple(new_o.items())
        #return tuple(frozenset(new_o.items()))
    
    # unhandled types
    else:
        return o
  
#: def make_hash()

class hashed_dict(OrderedDict):
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
    cache = hashed_dict()
    cache['a'] = 1
    cache[[1,2,3]] = 2
    print 'should be True:' , cache.has_key([1,2,3])
    print 'should be True:' , [1,2,3] in cache
    del cache[[1,2,3]]
    print 'should be False:' , cache.has_key([1,2,3])
    
        