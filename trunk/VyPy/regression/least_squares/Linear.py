
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import numpy as np
import scipy as sp

from VyPy.exceptions import EvaluationFailure
from VyPy.tools import vector_distance, atleast_2d_row

from Modeling import Modeling


# ----------------------------------------------------------------------
#   Linear Regression Model
# ----------------------------------------------------------------------

class Linear(Modeling):
    
    def __init__(self,Train):
        
        self.Train = Train
        
        # coefficients
        self.b = None
        self.c = None
        
        
    def learn(self):
        """
            regresses data with functions and gradients (if available)
            to this form:
                f = c + b_i*x_i
            
        """
        
        # unpack
        X = self.Train.X
        Y = self.Train.Y
        DY = self.Train.DY
        
        NX ,ND = X.shape
        NDY,_  = DY.shape
        
        print 'Build Information Matricies ...'
        
        # functions
        ay0 = np.array([[1.]]*NX)
        ay1 = X

        # basis matrix, functions
        Ay  = np.hstack([ay0,ay1])
        
        # arrays for the least squares regression
        At = Ay
        Yt = Y
        
        # gradients
        if False: # NDY:
            ad0 = np.array([[0.]]*NX*ND)
            ad1 = np.tile( np.eye(ND) , [NX,1]  )

            Ad = np.hstack([ad0,ad1])
            
            # add to arrays for least squares regression
            At = np.vstack([At,Ad])
            Yt = np.vstack([Yt, np.ravel(DY.T)[:,None]])
        
        print 'Least Squares Solve ...'
        B = sp.linalg.lstsq(At,Yt)[0]        
        
        # unpack data
        c = B[0,0]
        b = B[1:]
        
        # store results
        self.c = c
        self.b = b
        
        print ''
        
    
    def predict(self,XI):
        
        NX ,ND = XI.shape
        
        a0 = np.array([[1.]]*NX)
        a1 = XI
        
        A  = np.hstack([a0,a1])
        
        b0 = self.c
        b1 = self.b
        B  = np.vstack([b0,b1])
        
        YI = np.dot(A,B)      
        
        # TODO: DYI
        
        return YI
        
        
# ----------------------------------------------------------------------
#   Unit Test
# ----------------------------------------------------------------------
# See tests/regression/linear.py