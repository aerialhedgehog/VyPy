
from VyPy.tools.data_io import load_data, save_data
from VyPy.tools import HashedDict, make_hashable

import os, sys, time


# ----------------------------------------------------------------------
#   Remember
# ----------------------------------------------------------------------
class Remember(object):
    
    def __init__( self, function, filename='', write_freq=1, name='Remember'):
           
        # store
        self.function   = function
        self.filename   = filename
        self.write_freq = write_freq
        self.name       = name
        
        # initialize cache from file
        if filename and os.path.exists(filename) and os.path.isfile(filename):
            self.load_cache()
            
        # initialize new cache
        else:
            self.__cache__ = HashedDict()
        
        return
        
    def __func__(self,inputs):
        outputs = self.function(inputs)
        return outputs
        
    def __call__(self,inputs):
            
        # hashable type for cache
        _inputs = make_hashable(inputs)
                
        # check cache
        if self.__cache__.has_key(_inputs): 
            print 'PULLED FROM CACHE'
            outputs = self.__cache__[_inputs]
        
        # evalute function
        else:
            outputs = self.__func__(inputs)
            self.__cache__[_inputs] = outputs
        
        #: if cached
        
        # save cache
        if self.filename and len(self.__cache__)%self.write_freq == 0:
            self.save_cache()
        
        # done
        return outputs
    
    def load_cache(self):
        if not self.filename:
            raise Exception , 'no filename for loading cache'
        self.__cache__ = load_data(filename)
    
    def save_cache(self):
        if not self.filename:
            raise Exception , 'no filename for saving cache'
        save_data(self.__cache__,self.filename)
    













## ----------------------------------------------------------------------
##   Remember
## ----------------------------------------------------------------------
#class Remember(object):
    
    #def __init__( self, function, filename='', write_freq=1, name='Remember'):
           
        ## store
        #self.function   = function
        #self.filename   = filename
        #self.write_freq = write_freq
        #self.name       = name
        
        ## initialize cache from file
        #if filename and os.path.exists(filename) and os.path.isfile(filename):
            #self.load_cache()
            
        ## initialize new cache
        #else:
            #self.__cache__ = HashedDict()
        
        #return
        
    #def __func__(self,inputs):
        #outputs = self.function(inputs)
        #return outputs
        
    #def __call__(self,x):
        
        ## check for multiple inputs
        #if isinstance(x,(list,tuple)): 
            #xx = x   # multiple inputs
            #ff = [None] * len(xx)
            #f = [None]
        #elif numpy_isloaded and isinstance(x,(array_type,matrix_type)):
            #xx = x   # multiple inputs
            #ff = np.zeros_like(x)
            #f = np.array([None])
        #else:
            #xx = [x] # single input
            #ff = [None]
            #f = None
            
        ## initialize list of hashables for cache
        #_xx = [None]*len(xx)

        ## list of inputs to send to function
        #xo, _xo = [], []
            
        ## process input
        #for i,x in enumerate(xx):
            
            ## hashable type for cache
            #_x = make_hashable(x)
            #_xx[i] = _x
                
            ## check cache
            #if self.__cache__.has_key(_x): 
                #print 'PULLED FROM CACHE'
                #continue
                
            ## remember to call
            #xo.append(x)
            #_xo.append(_x)
            
        ## call function for un-cached inputs
        #if xo:
            ## multiple inputs
            #if isinstance(self.function,MultiTask):
                #fo = self.__func__(xo)
                #for _x,f in zip(_xo,fo):
                    #self.__cache__[_x] = f
            ## single inputs 
            #else:
                #for x,_x in zip(xo,_xo):
                    #f = self.__func__(x)
                    #self.__cache__[_x] = f
        ##: if uncached evals
        
        ## pull output 
        #for i,_x in enumerate(_xx):
            #ff[i] = self.__cache__[_x]
            
        ## format for output
        #if f is None: f = ff[0] # single input
        #else:         f = ff    # multiple input
        
        ## save cache
        #if self.filename and len(self.__cache__)%self.write_freq == 0:
            #self.save_cache()
        
        ## done
        #return f
    
    #def load_cache(self):
        #if not self.filename:
            #raise Exception , 'no filename for loading cache'
        #self.__cache__ = load_data(filename)
    
    #def save_cache(self):
        #if not self.filename:
            #raise Exception , 'no filename for saving cache'
        #save_data(self.__cache__,self.filename)
    
