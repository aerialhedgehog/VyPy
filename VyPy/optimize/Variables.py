
import os, shutil, sys, copy

from VyPy.tools.arrays import array_type, matrix_type
from VyPy.tools import OrderedDict as odict

iterable_type = (list,tuple,array_type,matrix_type)

# ----------------------------------------------------------------------
#   Variables
# ----------------------------------------------------------------------
class Variables(object):
    def __init__(self,inputs,scale=1.0):
        
        names,initial,bounds,_ = self.process_inputs(inputs)
        
        self.names   = names
        self.initial = initial
        self.bounds  = bounds
        
        self.scaled  = ScaledVariables(inputs)

        return
    
    def process_inputs(self,inputs):
        
        names   = []
        initial = []
        bounds  = []
        scales  = []
        
        default_bnd = [-1e100,1e100]
        
        for i,vals in enumerate( inputs ):
            
            if not isinstance(vals,(list,tuple)):
                # init only
                init = vals
                initial.append(init)
                
            else:
                # ['name' , vals , scale]
                if len(vals) == 3:
                    name,vals,scale = vals
                    names.append(name)
                    scales.append(scale)
                    
                # ['name' , vals ]
                elif len(vals) == 2:
                    name,vals = vals
                    scale = 1.0
                    names.append(name)
                    scales.append(scale)
                    
                # no name or scale
                else:
                    name = 'Var_%i' % i
                    scale = 1.0
                    names.append(name)
                    scales.append(scale)
                
                # vals = init only
                if not isinstance(vals,(list,tuple)):
                    init = vals
                    initial.append(init)
                    bounds.append(default_bnd)
                    
                # vals = (init)
                elif len(vals) == 1:
                    init = vals[0]
                    initial.append(init)
                    bounds.append(default_bnd)
                
                # vals = (lb,x0,ub)
                else:
                    lb,init,ub = vals
                    bound = [lb,ub]
                    initial.append(init)
                    bounds.append(bound)
            
            #: if named
            
        #: for each var
            
        return names,initial,bounds,scales
    
    def unpack(self,values):
        
        if self.names:
            # Unpack Variables
            names = self.names
            variables = odict()
            
            for name,val in zip(names,values):
                variables[name] = val
                
        elif isinstance(variables,iterable_type):
            variables = values
            
        else:
            raise Exception, 'could not unpack values: %s' % values
                
        return variables        
    
    def pack(self,variables):
        
        if self.names:
            names = self.names
            
            # Pack Variables
            values = [ variables[n] for n in names ]
            
        elif isinstance(variables,iterable_type):
            values = variables
                    
        else:
            raise Exception, 'could not unpack variables: %s' % variables
                
        return values
    
    def get_scaled(self,values):
        return self.scaled.get_scaled(values)
    def get_unscaled(self,values):
        return self.scaled.get_unscaled(values)
    
    def update_initial(self,variables):
        x0 = self.pack(variables)
        x0_scl = self.scaled.pack(variables)
        
        for i,(v,vs) in enumerate(zip(x0,x0_scl)):
            self.initial[i] = v
            self.scaled.initial[i] = vs
        
        return
     
# ----------------------------------------------------------------------
#   Helper Class - Scaled Variables
# ----------------------------------------------------------------------
        
class ScaledVariables(Variables):
    def __init__(self,inputs):
        
        names,initial,bounds,scales = self.process_inputs(inputs)
        
        for i in range(len(names)):
            scale = scales[i]
            initial[i] = initial[i]*scale
            bounds[i] = [ b*scale for b in bounds[i] ]
            
        self.names   = names
        self.initial = initial
        self.bounds  = bounds
        self.scales  = scales
        
        return
    
    def get_scaled(self,values):
        assert len(values) == len(self.initial) , 'wrong number of values'
        
        if isinstance(values,iterable_type):
            values = [ v/s for v,s in zip(values,self.scales) ]
            
        elif isinstance(values,dict):
            values = copy.deepcopy(values)
            for k,v,s in zip(values.keys(),values.values(),self.scales):
                values[k] = v/s
                
        else:
            raise Exception , 'could not unzip values'
        return values
        
    def get_unscaled(self,values):
        assert len(values) == len(self.initial) , 'wrong number of values'
        if isinstance(values,iterable_type):
            values = [ v*s for v,s in zip(values,self.scales) ]
            
        elif isinstance(values,dict):
            values = copy.deepcopy(values)
            for k,v,s in zip(values.keys(),values.values(),self.scales):
                values[k] = v*s
                
        else:
            raise Exception , 'could not unzip values'
        
        return values
        
    def unpack(self,values):
        scales = self.scales
        values = [ v/s for v,s in zip(values,scales) ]
        variables = Variables.unpack(self,values)
        return variables
    
    def pack(self,variables):
        scales = self.scales
        values = Variables.pack(self,variables)
        values = [ v*s for v,s in zip(values,scales) ]
        return values
  