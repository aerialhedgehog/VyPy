# scaling functions

import numpy as np
import copy
from VyPy.tools.IndexableBunch import IndexableBunch


class Training(object):
    
    def __init__(self,Train):
        
        self.calc_scaling(Train)
        
        return
    
    
    def calc_scaling(self,Train):
        
        # unpack
        X  = Train.X
        Y  = Train.Y
        DY = Train.DY
        XB = Train.XB
        _,nx = X.shape
        
        # calc scaling parameters
        Xref = np.array([ (XB[:,1] - XB[:,0]).T , 
                          (XB[:,0]          ).T ])
        
        Yref  = np.array([ np.max(Y,0)-np.min(Y,0) , 
                           np.min(Y,0)             ])
        
        DYref = np.array([ Yref[0]/Xref[0,:] , 
                           np.zeros([nx]) ])

        # update scaling parameters
        Ref = IndexableBunch()
        Ref.Xref  = Xref
        Ref.Yref  = Yref
        Ref.DYref = DYref
        self.Ref = Ref
        
        # define scaling and unscaling functions
        self.X_set    = lambda(Z): (Z - Ref['Xref'][1,:]) / Ref['Xref'][0,:]
        self.Y_set    = lambda(Z): (Z - Ref['Yref'][1,:]) / Ref['Yref'][0,:]
        self.DY_set   = lambda(Z): (Z - Ref['DYref'][1,:])/ Ref['DYref'][0,:]
        self.XB_set   = lambda(Z): ( (Z.T -  Ref['Xref'][1,:])/Ref['Xref'][0,:] ).T
        self.X_unset  = lambda(Z):  Z * Ref['Xref'][0,:]  + Ref['Xref'][1,:]
        self.Y_unset  = lambda(Z):  Z * Ref['Yref'][0,:]  + Ref['Yref'][1,:]
        self.DY_unset = lambda(Z):  Z * Ref['DYref'][0,:] + Ref['DYref'][1,:]
        self.XB_unset = lambda(Z): ( Z.T * Ref['Xref'][0,:] + Ref['Xref'][1,:] ).T
        
        # map of data keys to scaling function names
        self.scale_map = { 'XB'     : 'XB' ,
                           'X'      : 'X'  ,
                           'Y'      : 'Y'  ,
                           'DY'     : 'DY' ,
                           'XI'     : 'X'  ,
                           'YI'     : 'Y'  ,
                           'DYI'    : 'DY' ,
                           'CovYI'  : 'Y'  ,
                           'CovDYI' : 'DY'  }
        
        return
    
    #: def calc_scaling()
    
    def center_ci(self):
        ''' Translate Y's scaling function to satisfy 
            C*(X)=0 when C(X)=0
        '''
        
        # current scaling functions
        Y_set   = self.Y_set
        Y_unset = self.Y_unset
        # rename
        self.C_set   = Y_set
        self.C_unset = Y_unset
        # center scaled data on 0.0
        C_set   = lambda(Z): Y_set(Z) - Y_set(0.0)
        C_unset = lambda(Z): Y_unset( Z + Y_set(0.0) )
        # store
        self.Y_set   = C_set
        self.Y_unset = C_unset
        
        return
    
    #: def center_ci()
    
    def uncenter_ci(self):
        
        self.Y_set   = self.C_set
        self.Y_unset = self.C_unset
        
        return
        
    #: def uncenter_ci()
    
    def set_scaling(self,Data,key=None):
        
        Data = copy.deepcopy(Data)
        
        if key is None:
            for key in Data.keys():
                if self.scale_map.has_key(key):
                    func_key = self.scale_map[key] + '_set'
                    set_func = self.__dict__[ func_key  ]
                    Data[key] = set_func( Data[key] )
                    
        else: 
            func_key = self.scale_map[key] + '_set'
            set_func = self.__dict__[ func_key ]
            Data = set_func( Data )
        
        return Data
    
    #: def set_scaling()
    
    def unset_scaling(self,Data,key=None):
        
        Data = copy.deepcopy(Data)
        
        if key is None:
            for key in Data.keys():
                if self.scale_map.has_key(key):
                    func_key = self.scale_map[key] + '_unset'
                    unset_func = self.__dict__[ func_key  ]
                    Data[key] = unset_func( Data[key] )
                    
        else: 
            func_key = self.scale_map[key] + '_unset'
            unset_func = self.__dict__[ func_key ]
            Data = unset_func( Data )
        
        return Data
    
    #: def unset_scaling()
            
        
#: class Training()
        

def no_scaling(Z): return Z

class f_set(object):
    def __init__(self,params,key):
        self.params = params
        self.key = key
    def __call__(self,Z):
        Zref = self.params[self.key]
        return (Z - Zref[1,:]) / Zref[0,:]
class f_unset(object):
    def __init__(self,params,key):
        self.params = params
        self.key = key
    def __call__(self,Z):
        Zref = self.params[self.key]
        return Z * Zref[0,:]  + Zref[1,:]

class fT_set(object):
    def __init__(self,params,key):
        self.params = params
        self.key = key
    def __call__(self,Z):
        Zref = self.params[self.key]
        return ( (Z.T - Zref[1,:])/Zref[0,:] ).T
class fT_unset(object):
    def __init__(self,params,key):
        self.params = params
        self.key = key
    def __call__(self,Z):
        Zref = self.params[self.key]
        return ( Z.T * Zref[0,:] + Zref[1,:] ).T

class c_set(object):
    def __init__(self,f_set,f_unset):
        self.f_set   = f_set
        self.f_unset = f_unset
    def __call__(self,Z):
        f_set   = self.f_set
        #f_unset = self.f_unset
        return f_set(Z) - f_set(0.0)
class c_unset(object):
    def __init__(self,f_set,f_unset):
        self.f_set   = f_set
        self.f_unset = f_unset
    def __call__(self,Z):
        f_set   = self.f_set
        f_unset = self.f_unset
        return f_unset( Z + f_set(0.0) )

class s_set(object):
    def __init__(self,params,key):
        self.params = params
        self.key = key
    def __call__(self,Z):
        Zref = self.params[self.key]
        return Z / Zref[0,:]
class s_unset(object):
    def __init__(self,params,key):
        self.params = params
        self.key = key
    def __call__(self,Z):
        Zref = self.params[self.key]
        return Z * Zref[0,:] 
    
class l_set(object):
    def __init__(self,params,key):
        self.params = params
        self.key = key
    def __call__(self,Z):
        Zref = self.params[self.key]
        return Z / np.mean( Zref[0,:] )
class l_unset(object):
    def __init__(self,params,key):
        self.params = params
        self.key = key
    def __call__(self,Z):
        Zref = self.params[self.key]
        return Z * np.mean( Zref[0,:] )
    
class logS_set(object):
    def __init__(self,params,key):
        self.params = params
        self.key = key
    def __call__(self,Z):
        Zref = self.params[self.key]
        return Z - np.log10( Zref[0,:] )
class logS_unset(object):
    def __init__(self,params,key):
        self.params = params
        self.key = key
    def __call__(self,Z):
        Zref = self.params[self.key]
        return Z + np.log10( Zref[0,:] )


    




## GRAVETARK


class Scalable_Value(object):
    
    _scaled = False
    
    def __init__(self,value,f_set,f_unset,scaled=False):
        self.value   = copy.deepcopy( value )
        self.f_set   = f_set
        self.f_unset = f_unset
        self._scaled = scaled
        return
    
    def set_scaling(self):
        if self._scaled: return
        self.value = self.f_set( self.value )
        self._scaled = True
        return self.value
    
    def unset_scaling(self):
        if not self._scaled: return
        self.value = self.f_unset( self.value )
        self._scaled = False
        return self.value
    
    def get_value(self):
        return self.value
        
    def __str__(self):
        return self.value.__str__()    
    def __repr__(self):
        return self.value.__repr__()
    
#: class Scalable_Value()

def Scalable_Data(data=None,params=None):
    """ Factory Function """
    
    NewCls = Scalable_Data_Class()
    
    if not data is None:
               
        # error checking
        scaled = None
        for key,value in data.iteritems():
            assert isinstance(value,Scalable_Value) , 'input data must be type Scalable_Value'
            if scaled is None:
                scaled = value._scaled
            else:
                assert scaled == value._scaled , 'values must be all scaled or all unscaled'
        
        # set scaling
        NewCls._scaled = scaled
        
        # set data
        NewCls.update(data)
            
    if not params is None:
        NewCls._params = params
        
    return NewCls

class Scalable_Data_Class(IndexableBunch):
    
    _params = None
    _scaled = None
    
    def __init__(self,*args,**kwarg):
        super(Scalable_Data_Class,self).__init__(*args,**kwarg)
        self._params = IndexableBunch()
        self._scaled = False
    
    def set_scaling(self):
        for key,value in self.iteritems():
            value.set_scaling()
        self._scaled = True
        return
    
    def unset_scaling(self):
        for key,value in self.iteritems():
            value.unset_scaling()
        self._scaled = False
        return
    
    def __getattr__(self,k):
        if self._initialized and self.has_key(k):
            return self[k].value
        else:
            self.getitem(k)
    
    def __setattr__(self,k,v):
        if self._initialized and self.has_key(k):
            self[k].value = v
        else:
            self.setitem(k,v)
            
    def getitem(self,k):
        return IndexableBunch.__getattr__(self,k)
    def setitem(self,k,v):
        IndexableBunch.__setattr__(self,k,v)
        
    
#: class Scalable_Data()

