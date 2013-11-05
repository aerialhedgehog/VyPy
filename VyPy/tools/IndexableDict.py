

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
            