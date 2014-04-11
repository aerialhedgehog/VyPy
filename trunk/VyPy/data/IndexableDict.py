

from OrderedDict import OrderedDict

class IndexableDict(OrderedDict):
    """ An OrderedDict with list-style access 
    """
        
    def __getitem__(self,k):
        if isinstance(k,int):
            try:           
                return self[ self.keys()[k] ]
            except IndexError:
                raise IndexError(k)
        else:
            return super(IndexableDict,self).__getitem__(k)
    
    def __setitem__(self,k,v):
        if isinstance(k,int):
            try:
                self[ self.keys()[k] ] = v
            except IndexError:
                raise IndexError(k)
        else:
            super(IndexableDict,self).__setitem__(k,v)
    
    def __delitem__(self,k):
        if isinstance(k,int):
            try:
                del self[ self.keys()[k] ]
            except IndexError:
                raise IndexError(k)
        else:
            super(IndexableDict,self).__delitem__(k)
    
    # iterate on values, not keys
    def __iter__(self):
        return super(IndexableDict,self).itervalues()

    def index(self,key):
        if isinstance(key,int):
            index = range(len(self))[key]
        else:
            index = self.keys().index(key)
        return index  
    
    def insert(self,index,key,value):
        """ IndexableDict.insert(index,key,value)
            insert key and value before index
            index can be an integer or key name
        """
        # potentially expensive....
        # clears dictionary in process...
        
        # original length
        len_self = len(self)
        
        # add to dictionary
        self[key] = value
        
        # find insert index number
        index = self.index(index)
        
        # done if insert index is greater than list length
        if index >= len_self: return
        
        # insert into index list
        indeces = range(0,len_self) 
        indeces.insert(index,len_self) # 0-based indexing ...
        
        # repack dictionary
        keys = self.keys()
        values = self.values()
        self.clear()
        for i in indeces: self[keys[i]] = values[i]
    
    def swap(self,key1,key2):
        """ IndexableDict.swap(key1,key2)
            swap key locations 
        """
        # potentially expensive....
        # clears dictionary in process...
        
        # get swapping indeces
        index1 = self.index(key1)
        index2 = self.index(key2)
        
        # list of all indeces
        indeces = range(0,len(self))
        
        # replace index1 with index2
        indeces.insert(index1,index2)
        del(indeces[index1+1])
        
        # replace index2 with index1
        indeces.insert(index2,index1)
        del(indeces[index2+1])
        
        # repack dictionary
        keys = self.keys()
        values = self.values()
        self.clear()
        for i in indeces: self[keys[i]] = values[i]     
        
        
        
        
if __name__ == '__main__':
    
    o = IndexableDict()
    o['x'] = 'hello'
    o['y'] = 1
    o['z'] = [3,4,5]
    o['t'] = IndexableDict()
    o['t']['h'] = 20
    o['t']['i'] = (1,2,3)

    print o

    import pickle

    d = pickle.dumps(o)
    p = pickle.loads(d)
    
    print ''
    print p    
    
    o['t']['h'] = 'changed'
    p.update(o)
    p['t'].update(o)

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
        
    #class TestObject(IndexableDict):
        #def __init__(self,c):
            #self.c = c
    
    #o = TestObject(555)
    #o['x'] = TestDescriptor([1,2,3])
    #o['y'] = TestDescriptor([4,3,5])
    
    #print ''
    #print o
    #print o.c
    
    #print ''
    #o['x'] = [3,4,5]
    
    #print ''
    #print o[0]
            
            
    #import pickle
        
    #d = pickle.dumps(o)
    #p = pickle.loads(d)
    
    #print ''
    #print p
    #print p[0]
    #print p.c