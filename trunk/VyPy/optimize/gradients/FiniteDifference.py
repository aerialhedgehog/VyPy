
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from VyPy.data import OrderedBunch as obunch
from VyPy.tools import atleast_2d_row, atleast_2d_col, array_type

from copy import deepcopy
import numpy as np
        
# ----------------------------------------------------------------------
#   Finite Difference Gradient
# ----------------------------------------------------------------------

class FiniteDifference(object):
    
    def __init__(self,function,step=1e-6):
        
        self._function = function
        self.step = step
        
        return
    
    def function(self,variables_list):
        
        nx = len(variables_list)
        
        results_list = [0]*nx
        
        for i,variables in enumerate(variables_list):
            
            results_list[i] = self._function(variables)
            
        return results_list
        
        
    def __call__(self,variables):
        
        step = self.step
        
        variables = deepcopy(variables)
        
        if not isinstance(variables,obunch):
            variables = obunch(variables)
            
        # arrayify variables
        values_init = variables.pack_array('vector')
        values_init = atleast_2d_row(values_init)
        nx = values_init.shape[1]
        
        # prepare step
        if not isinstance(step,(array_type,list,tuple)):
            step = [step] * nx
        step = atleast_2d_row(step)
        if not step.shape[0] == 1:
            step = step.T
        
        values_run = np.vstack([ values_init , 
                                 np.tile(values_init,[nx,1]) + np.diag(step[0]) ])
        variables_run = [ deepcopy(variables).unpack_array(val) for val in values_run ]
            
        # run the function
        #for i in variables_run: print i
        results = self.function(variables_run)
        #for i in results: print i
        
        results_values = np.vstack([ atleast_2d_row( res.pack_array('vector') ) for res in results ])
        
        # pack results
        gradients_values = ( results_values[1:,:] - results_values[None,0,:] ) / step.T
        gradients_values = np.ravel( gradients_values )
        
        variables = variables.unpack_array( values_init[0] * 0.0 )
        
        gradients = deepcopy(results[0])
        if not isinstance(gradients,obunch):
            gradients = obunch(gradients)
            
        for i,g in gradients.items():
            if isinstance(g,array_type):
                g = atleast_2d_col(g)
            gradients[i] = deepcopy(variables)
            for j,v in gradients[i].items():
                if isinstance(v,array_type):
                    v = atleast_2d_row(v)
                gradients[i][j] = g * v
        
        gradients.unpack_array(gradients_values)
        
        return gradients
        