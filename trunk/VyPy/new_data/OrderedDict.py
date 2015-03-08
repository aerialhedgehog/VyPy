
from Dict import Dict

#from lib.python_OrderedDict import OrderedDict as _OrderedDict
#from lib.cython_OrderedDict import OrderedDict as _OrderedDict
from cyordereddict import OrderedDict as _OrderedDict

#try:
    #from lib.cython_OrderedDict import OrderedDict as _OrderedDict
#except ImportError:
    #from lib.python_OrderedDict import OrderedDict as _OrderedDict

#try:
    #from cyordereddict import OrderedDict as _OrderedDict
#except ImportError:
    #from _OrderedDict import OrderedDict as _OrderedDict


class OrderedDict(_OrderedDict,Dict):
    
    update   = Dict.update
    __repr__ = Dict.__repr__
    __str__  = Dict.__str__
    
    
    
    #def __init__(self, items=None, **kwds):
        #'''Initialize an ordered dictionary.  Signature is the same as for
           #regular dictionaries, but keyword arguments are not recommended
           #because their insertion order is arbitrary.
        #'''
                    
        #def append_value(key,value):               
            #self[key] = value            
        
        ## a dictionary
        #if hasattr(items, 'iterkeys'):
            #for key in items.iterkeys():
                #append_value(key,items[key])

        #elif hasattr(items, 'keys'):
            #for key in items.keys():
                #append_value(key,items[key])
                
        ## items lists
        #elif items:
            #for key, value in items:
                #append_value(key,value)
                
        ## key words
        #for key, value in kwds.iteritems():
            #append_value(key,value)    
            
            
    def pack_array(self,output='vector'):
        """ OrderedDict.pack_array(output='vector')
            maps the data dict to a 1D vector or 2D column array
            
            Inputs - 
                output - either 'vector' (default), or 'array'
                         chooses whether to output a 1d vector or 
                         2d column array
            Outputs - 
                array - the packed array
                
            Assumptions - 
                will only pack int, float, np.array and np.matrix (max rank 2)
                if using output = 'matrix', all data values must have 
                same length (if 1D) or number of rows (if 2D), otherwise is skipped
        
        """
        
        # dont require dict to have numpy
        import numpy as np
        from VyPy.tools.arrays import atleast_2d_col, array_type, matrix_type
        
        # check output type
        if not output in ('vector','array'): raise Exception , 'output type must be "vector" or "array"'        
        vector = output == 'vector'
        
        # list to pre-dump array elements
        M = []
        
        # valid types for output
        valid_types = ( int, float,
                        array_type,
                        matrix_type )
        
        # initialize array row size (for array output)
        size = [False]
        
        # the packing function
        def do_pack(D):
            for v in D.itervalues():
                # type checking
                if isinstance( v, OrderedDict ): 
                    do_pack(v) # recursion!
                    continue
                elif not isinstance( v, valid_types ): continue
                elif np.ndim(v) > 2: continue
                # make column vectors
                v = atleast_2d_col(v)
                # handle output type
                if vector:
                    # unravel into 1d vector
                    v = v.ravel(order='F')
                else:
                    # check array size
                    size[0] = size[0] or v.shape[0] # updates size once on first array
                    if v.shape[0] != size[0]: 
                        #warn ('array size mismatch, skipping. all values in data must have same number of rows for array packing',RuntimeWarning)
                        continue
                # dump to list
                M.append(v)
            #: for each value
        #: def do_pack()
        
        # do the packing
        do_pack(self)
        
        # pack into final array
        if M:
            M = np.hstack(M)
        else:
            # empty result
            if vector:
                M = np.array([])
            else:
                M = np.array([[]])
        
        # done!
        return M
    
    def unpack_array(self,M):
        """ OrderedDict.unpack_array(array)
            unpacks an input 1d vector or 2d column array into the data dictionary
                following the same order that it was unpacked
            important that the structure of the data dictionary, and the shapes
                of the contained values are the same as the data from which the 
                array was packed
        
            Inputs:
                 array - either a 1D vector or 2D column array
                 
            Outputs:
                 a reference to self, updates self in place
                 
        """
        
        # dont require dict to have numpy
        import numpy as np
        from VyPy.tools.arrays import atleast_2d_col, array_type, matrix_type
        
        # check input type
        vector = np.ndim(M) == 1
        
        # valid types for output
        valid_types = ( int, float,
                        array_type,
                        matrix_type )
        
        # counter for unpacking
        _index = [0]
        
        # the unpacking function
        def do_unpack(D):
            for k,v in D.iteritems():
                
                # type checking
                if isinstance(v,OrderedDict): 
                    do_unpack(v) # recursion!
                    continue
                elif not isinstance(v,valid_types): continue
                
                # get this value's rank
                rank = np.ndim(v)
                
                # get unpack index
                index = _index[0]                
                
                # skip if too big
                if rank > 2: 
                    continue
                
                # scalars
                elif rank == 0:
                    if vector:
                        D[k] = M[index]
                        index += 1
                    else:#array
                        continue
                        #raise RuntimeError , 'array size mismatch, all values in data must have same number of rows for array unpacking'
                    
                # 1d vectors
                elif rank == 1:
                    n = len(v)
                    if vector:
                        D[k][:] = M[index:(index+n)]
                        index += n
                    else:#array
                        D[k][:] = M[:,index]
                        index += 1
                    
                # 2d arrays
                elif rank == 2:
                    n,m = v.shape
                    if vector:
                        D[k][:,:] = np.reshape( M[index:(index+(n*m))] ,[n,m], order='F')
                        index += n*m 
                    else:#array
                        D[k][:,:] = M[:,index:(index+m)]
                        index += m
                
                #: switch rank
                
                _index[0] = index

            #: for each itme
        #: def do_unpack()
        
        # do the unpack
        do_unpack(self)
         
        # check
        if not M.shape[-1] == _index[0]: 
            raise IndexError , 'did not unpack all values'
         
        # done!
        return self
    
    def do_recursive(self,method,other=None,default=None):
        
        # result data structure
        klass = self.__class__
        from VyPy.data import DataBunch
        if isinstance(klass,DataBunch):
            klass = DataBunch
        result = klass()
                
        # the update function
        def do_operation(A,B,C):
            for k,a in A.iteritems():
                if isinstance(B,OrderedDict):
                    if B.has_key(k):
                        b = B[k]
                    else: 
                        continue
                else:
                    b = B
                # recursion
                if isinstance(a,OrderedDict):
                    c = klass()
                    C[k] = c
                    do_operation(a,b,c)
                # method
                else:
                    if b is None:
                        c = method(a)
                    else:
                        c = method(a,b)
                    if not c is None:
                        C[k] = c
                #: if type
            #: for each key,value
        #: def do_operation()
        
        # do the update!
        do_operation(self,other,result)    
        
        return result
    
    
    def __reduce__(self):
        """Return state information for pickling"""
        items = [( k, OrderedDict.__getitem__(self,k) ) for k in OrderedDict.iterkeys(self)]
        inst_dict = vars(self).copy()
        for k in vars(OrderedDict()):
            inst_dict.pop(k, None)
        return (_reconstructor, (self.__class__,items,), inst_dict)
    
# for rebuilding dictionaries with attributes
def _reconstructor(klass,items):
    self = OrderedDict.__new__(klass)
    OrderedDict.__init__(self,items)
    return self


if __name__ == '__main__':
    
    import string
    k = string.lowercase
    v = range(0,26)
    i = zip(k,v)
    
    q = OrderedDict(i)
    print q
    
    o = OrderedDict()
    o['x'] = 'hello'
    o['y'] = 1
    o['z'] = [3,4,5]
    o['t'] = OrderedDict()
    o['t']['h'] = 20
    o['t']['i'] = (1,2,3)
    
    print isinstance(o,Dict)
    
    print o
    
    import pickle
    
    d = pickle.dumps(o)
    p = pickle.loads(d)
    
    print ''
    print p
    
    print isinstance(o,Dict)
    
    o['t']['h'] = 'changed'
    p.update(o)
    p['t'].update(o)
    
    print p.keys()

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
        
    #class TestObject(OrderedDict):
        #pass
        ##def __init__(self,c):
            ##self.c = c
    
    #o = TestObject()
    #o['x'] = TestDescriptor([1,2,3])
    #o['y'] = TestDescriptor([4,3,5])
    #for i in range(10):
        #o['x%i'%i] = 'yo'    
    
    #print ''
    #print o
    ##print o.c
    
    #print ''
    #o['x'] = [3,4,5]
            
    #print ''
    #print 'pickle'
    #import pickle
        
    #d = pickle.dumps(o)
    #p = pickle.loads(d)
    
    #print ''
    #print p
    ##print p.c