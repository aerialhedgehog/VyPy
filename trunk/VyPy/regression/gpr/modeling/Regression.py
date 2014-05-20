
import sys, copy, weakref,gc

import numpy as np
import scipy as sp
import scipy.linalg

from Modeling import Modeling

from VyPy.exceptions import EvaluationFailure
from VyPy.data import IndexableDict
from VyPy.tools import vector_distance, atleast_2d

class Regression(Modeling):
    
    def __init__(self,Learn,Scaling=None):
        
        Modeling.__init__(self,Learn,Scaling)
        
        # reinterpolated model
        self.RIE = {}
        self.RI = None
        
        # try to precalc Kernel
        ##self.safe_precalc()
        
        return
    
    #: def __init__()
            
    def reinterpolate(self):
            
        # new noise param (log10)
        sig_nR = -6.0
        
        # copy model data
        Learn = copy.deepcopy( self.Learn )
        
        new_Model = Modeling( Learn )
        new_Model.safe_precalc()
        
        # kernel output data names
        self.RIE = IndexableDict()
                
        # evaluate training data with model
        X = self.Train.X
        The_Data = new_Model.predict(X)
        
        try:
            The_Data.keys()
        except KeyError:
            raise Exception, 'Kernel must return pack_outputs() to use Modeling.reinterpolate()'
        
        # operate on each training output
        for output in Train.keys():
            
            # check if returned by data
            if output == 'X':
                continue
            
            # pull old data
            D = Train[output]
            if D.shape[0] == 0: continue
            
            # pull new data
            output_i = output + 'I'
            DI = The_Data[output_i]
            
            # calculate errors
            RIE = np.log10( np.mean( np.abs( DI  - D ) ) )
            self.RIE[output] = RIE
            print 'RIE %s: %.4f' % (output,RIE)
            
            # reassign reinterpolated redata
            Train.setitem(output,The_Data[output_i])
            
        #: for each output    
        
        # reinterpolated implies no noise
        new_Model.Hypers.sig_ny  = sig_nR
        new_Model.Hypers.sig_ndy = sig_nR
        
        # store
        self.RI = new_Model
                
        return
    
    #: def reinterpolate()   

     
#: class Modeling()