
from cpython.dict cimport PyDict_DelItem, PyDict_SetItem, PyDict_GetItem, PyDict_Contains
from cpython.object cimport PyObject_HasAttr, PyObject_GetAttr, PyObject_SetAttr, PyObject_DelAttr

cdef class Bunch(dict):
    """ A dictionary that provides attribute-style access.
    """
    
    def __contains__(self, k):
        try:
            return PyObject_HasAttr(self, k) or PyDict_Contains(self, k)
        except:
            return False
        
    # only called if k not found in normal places 
    def __getattr__(self, k):
        """ Gets key if it exists, otherwise throws AttributeError.
        """
        if PyDict_Contains(self, k):
            return dict.__getitem__(self,k)
        else:
            return super(Bunch,self).__getattribute__(k)
    
    def __setattr__(self, k, v):
        """ Sets attribute k if it exists, otherwise sets key k. A KeyError
            raised by set-item (only likely if you subclass Bunch) will 
            propagate as an AttributeError instead.
        """
        if not PyObject_HasAttr(self, k) or PyDict_Contains(self, k):
            PyDict_SetItem(self,k,v)
        else:
            PyObject_SetAttr(self,k, v)
    
    def __delattr__(self, k):
        """ Deletes attribute k if it exists, otherwise deletes key k. A KeyError
            raised by deleting the key--such as when the key is missing--will
            propagate as an AttributeError instead.
        """
        if not PyObject_HasAttr(self, k) or PyDict_Contains(self, k):
            PyDict_DelItem(self,k)
        else:
            PyObject_DelAttr(self,k)

    def copy(self):
        return self.__class__(self)
        
    def __reduce__(self):
        items = self.items()
        return self.__class__, (items,)
        

#from cpython.dict cimport PyDict_DelItem, PyDict_SetItem, PyDict_GetItem, PyDict_Contains
#from cpython.object cimport PyObject_HasAttr, PyObject_GetAttr, PyObject_SetAttr, PyObject_DelAttr


#cdef class Bunch(dict):
    #""" A dictionary that provides attribute-style access.
    #"""
    
    #def __contains__(self, k):
        #try:
            #return PyObject_HasAttr(self, k) or PyDict_Contains(self, k)
        #except:
            #return False
        
    ## only called if k not found in normal places 
    #def __getattr__(self, k):
        #""" Gets key if it exists, otherwise throws AttributeError.
        #"""
        #try:
            ## Throws exception if not in prototype chain
            #return super(Bunch,self).__getattribute__(k)
            ##return PyObject_GetAttr(self,k)
        #except AttributeError:
            #try:
                #return self[k]
                ##return PyDict_GetItem(self,k)
            #except KeyError:
                #raise AttributeError(k)
    
    #def __setattr__(self, k, v):
        #""" Sets attribute k if it exists, otherwise sets key k. A KeyError
            #raised by set-item (only likely if you subclass Bunch) will 
            #propagate as an AttributeError instead.
        #"""
        #try:
            ## Throws exception if not in prototype chain
            #super(Bunch,self).__getattribute__(k)
            ##PyObject_GetAttr(self,k)
        #except AttributeError:
            #try:
                #PyDict_SetItem(self, k, v)
            #except:
                #raise AttributeError(k)
        #else:
            #super(Bunch,self).__setattr__(k, v)
    
    #def __delattr__(self, k):
        #""" Deletes attribute k if it exists, otherwise deletes key k. A KeyError
            #raised by deleting the key--such as when the key is missing--will
            #propagate as an AttributeError instead.
        #"""
        #try:
            ## Throws exception if not in prototype chain
            #super(Bunch,self).__getattribute__(k)
            ##PyObject_GetAttr(self, k)
        #except AttributeError:
            #try:
                #PyDict_DelItem(self, k)
            #except KeyError:
                #raise AttributeError(k)
        #else:
            #PyObject_DelAttr(self, k)
        

    #def copy(self):
        #return self.__class__(self)
        
    #def __reduce__(self):
        #items = self.items()
        #return self.__class__, (items,)

