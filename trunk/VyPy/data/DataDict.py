
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from IndexableBunch import IndexableBunch

import types
from copy            import deepcopy
from warnings        import warn


# ----------------------------------------------------------------------
#  DataDict
# ----------------------------------------------------------------------

class DataDict(IndexableBunch):
    """ DataDict()
        
        a dict-type container with attribute, item and index style access
        initializes with default attributes
        will recursively search for defaults of base classes
        current class defaults overide base class defaults

        Methods:
            __defaults__(self)      : sets the defaults of 
            find_instances(datatype)
    """
    
    def __defaults__(self):
        pass    
    
    def __new__(cls,*args,**kwarg):
        """ supress use of args or kwarg for defaulting
        """
        
        # initialize data, no inputs
        self = IndexableBunch.__new__(cls)
        IndexableBunch.__init__(self)
        
        # get base class list
        klasses = self.get_bases()
                
        # fill in defaults trunk to leaf
        for klass in klasses[::-1]:
            klass.__defaults__(self)
        
        ## ensure local copies
        #for k,v in self.iteritems():
            #self[k] = deepcopy(v)
            
        return self
    
    def __init__(self,*args,**kwarg):
        """ initializes DataDict()
        """
        
        # handle input data (ala class factory)
        input_data = IndexableBunch(*args,**kwarg)
        
        # update this data with inputs
        self.update(input_data)
        
        # call over-ridable post-initialition setup
        self.__check__()
        
    #: def __init__()
    
    def __check__(self):
        """ 
        """
        pass
    
    
    def __setitem__(self,k,v):
        # attach all functions as static methods
        if isinstance(v,types.FunctionType):
            v = staticmethod(v)
        IndexableBunch.__setitem__(self,k,v)
        
    def __str__(self,indent=''):
        new_indent = '  '
        args = ''
        
        # trunk data name
        if not indent:
            args += self.dataname()  + '\n'
        else:
            args += ''
            
        args += IndexableBunch.__str__(self,indent)
        
        return args
        
    def __repr__(self):
        return self.__str__()
    
    def find_instances(self,data_type):
        """ DataDict.find_instances(data_type)
            
            searches DataDict() for instances of given data_type
            
            Inputs:
                data_type  - a class type, for example type(myData)
                
            Outputs:
                data - DataDict() of the discovered data
        """
        
        output = DataDict()
        for key,value in self.iteritems():
            if isinstance(value,type):
                output[key] = value
        return output
    
    def get_bases(self):
        """ find all DataDict() base classes, return in a list """
        klass = self.__class__
        klasses = []
        while klass:
            if issubclass(klass,DataDict): 
                klasses.append(klass)
                klass = klass.__base__
            else:
                klass = None
        if not klasses: # empty list
            raise TypeError , 'class %s is not of type DataDict()' % self.__class__
        return klasses
    
    def typestring(self):
        # build typestring
        typestring = str(type(self)).split("'")[1]
        typestring = typestring.split('.')
        if typestring[-1] == typestring[-2]:
            del typestring[-1]
        typestring = '.'.join(typestring) 
        return typestring
    
    def dataname(self):
        return "<data object '" + self.typestring() + "'>"


    
# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------        

    
if __name__ == '__main__':
    

    d = DataDict()
    d.tag = 'data name'
    d['value'] = 132
    d.options = DataDict()
    d.options.field = 'of greens'
    d.options.half  = 0.5
    print d
    
    import numpy as np
    ones = np.ones([10,1])
        
    m = DataDict()
    m.tag = 'numerical data'
    m.hieght = ones * 1.
    m.rates = DataDict()
    m.rates.angle  = ones * 3.14
    m.rates.slope  = ones * 20.
    m.rates.special = 'nope'
    m.value = 1.0
    
    print m
    
    V = m.pack_array('vector')
    M = m.pack_array('array')
    
    print V
    print M
    
    V = V*10
    M = M-10
    
    print m.unpack_array(V)
    print m.unpack_array(M)
    