
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import os, shutil, sys, copy

from VyPy.tools.arrays import array_type, matrix_type
from VyPy.data import IndexableDict, Object
idict = IndexableDict

iterable_type = (list,tuple,array_type,matrix_type)

# ----------------------------------------------------------------------
#   Variable
# ----------------------------------------------------------------------
class Variable(object):
    
    Container = None # linked at end of module
    
    def __init__( self, tag, initial=0.0,
                  bounds=(1.e-100,1.e+100), scale=1.0 ):
        self.tag     = tag
        self.initial = initial
        self.bounds  = bounds
        self.scale   = scale
    
    def __repr__(self):
        return '<Variable %s>' % self.tag

# ----------------------------------------------------------------------
#   Scaled Variable
# ----------------------------------------------------------------------
class ScaledVariable(object):

    Container = None # linked at end of module
    
    def __init__( self, tag, initial=0.0,
                  bounds=(1.e-100,1.e+100), scale=1.0 ):
        scale = float(scale)
        self.tag     = tag
        self.initial = initial*scale
        self.bounds  = tuple([ b*scale for b in bounds ])
        self.scale   = scale
    
    def __repr__(self):
        return '<ScaledVariable %s>' % self.tag


# ----------------------------------------------------------------------
#   Variables
# ----------------------------------------------------------------------
class Variables(IndexableDict):
    
    def __init__(self):
        self.scaled = ScaledVariables(self)
    
    def __set__(self,problem,arg_list):
        self.clear()
        self.extend(arg_list)

    def append( self, tag, initial=0.0,
                bounds=(1.e-100,1.e+100), scale=1.0 ):

        tag = self.next_key(tag)
        
        variable   = Variable(tag,initial,bounds,scale)
        scaled_var = ScaledVariable(tag,initial,bounds,scale)
                
        self[tag]        = variable
        self.scaled[tag] = scaled_var
        
        return
    
    def extend(self,arg_list):
        for args in arg_list:
            self.append(*args)
    
    def pack(self,values):
        """ vars = Variables.pack(vals)
            pack a list of values into an ordered dictionary 
            of variables
        """
        
        # Pack Variables
        variables = idict()
        
        for tag,val in zip(self.tags(),values):
            variables[tag] = val
        
        return variables
    
    def unpack(self,variables):
        """ values = Variables.pack(vars)
            unpack an ordered dictionary of variables into
            a list of values
        """
        
        if isinstance(variables,dict):
            # Unpack Variables
            values = [ variables[tag] for tag in self.tags() ]
            
        elif isinstance(variables,iterable_type):
            # already unpacked
            values = variables
                    
        else:
            raise Exception, 'could not unpack variables: %s' % variables
                
        return values
    
    def tags(self):
        return self.keys()
    def initials(self):
        return [ var.initial for var in self.values() ]
    def bounds(self):
        [ var.bounds for var in self.values() ]
    def scales(self):
        return [ var.scale for var in self.values() ]
    
    def set(self,initials=None,bounds=None,scales=None):
        if initials:
            for i,(v,s) in enumerate(zip(initials,self.scales())):
                self[i].initial = v
                self.scaled[i].initial = v*s
        if scales:
            for i,s in enumerate(scales):
                self[i].scale = s
                self[i].scaled.scale = s
        if bounds:
            for i,(bnd,s) in enumerate(zip(bounds,self.scales())):
                self[i].bounds = b
                self[i].scaled.bounds = [ v*s for v in b ]
    

     
# ----------------------------------------------------------------------
#   Helper Class - Scaled Variables
# ----------------------------------------------------------------------
        
class ScaledVariables(Variables):
    
    def __init__(self,variables):
        self.unscaled = variables
     
    def pack(self,values):
        """ vars = Variables.pack(vals)
            pack a list of values into an ordered dictionary 
            of variables
        """
        scales = self.scales()
        values = [ v/s for v,s in zip(values,scales) ]
        variables = Variables.pack(self,values)
        return variables
    
    def unpack(self,variables):
        """ values = Variables.pack(vars)
            unpack an ordered dictionary of variables into
            a list of values
        """
        scales = self.scales()
        values = Variables.unpack(self,variables)
        values = [ v*s for v,s in zip(values,scales) ]
        return values
    
    def __set__(self,*args):
        ''' not used '''
        raise AttributeError('__set__')
    def append(self,*args):
        ''' not used '''
        raise AttributeError('append')
    def extend(self,*args):
        ''' not used '''
        raise AttributeError('extend')
             
    def scales(self):
        return [ var.scale for var in self.values() ]
    def initials(self):
        return [ var.initial for var in self.values() ]
    def bounds(self):
        return [ var.bounds for var in self.values() ]
    
    def set(self,initials=None,bounds=None,scales=None):
        if initials:
            for i,(v,s) in enumerate(zip(initials,self.scales)):
                self[i].initial = v/s
                self.scaled[i].initial = v
        if bounds:
            for i,(bnd,s) in enumerate(zip(bounds,self.scales)):
                self[i].bounds = [ v/s for v in bnd ]
                self[i].scaled.bounds = bnd
        if scales:
            for i,s in enumerate(scales):
                self[i].scale = s
                self[i].scaled.scale = s                
    
    
# ----------------------------------------------------------------------
#   Container Linking
# ----------------------------------------------------------------------
Variable.Container = Variables
ScaledVariable.Container = ScaledVariable