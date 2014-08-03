
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import time, os, gc, sys, copy

import numpy as np

import VyPy
import VyPy.optimize as opt
from VyPy.data import obunch

import warnings
warnings.simplefilter('error',RuntimeWarning)


# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():
    
    setup_problem()
    

# ----------------------------------------------------------------------
#   An Evaluator
# ----------------------------------------------------------------------

class Test_Evaluator(opt.Evaluator):
    
    def function(self,inputs):
        
        # unpack inputs
        x1 = inputs.x1
        x2 = inputs['x2']
        x3 = inputs.x3
        
        # the math
        f = x1**2 + x2**2 + np.sum(x3)**2
        c = x1 + x2 + x3[0]
        c2 = x3 - 1.
        
        # pack outputs
        outputs = obunch()
        outputs.f  = f
        outputs.c  = c
        outputs.c2 = c2
        
        return outputs

# ----------------------------------------------------------------------
#   Setup an Optimization Problem
# ----------------------------------------------------------------------

def setup_problem():
    
    # initialize the problem
    problem = opt.Problem()
    
    # setup variables, list style
    problem.variables = [
    #   [ 'tag' , x0, (lb ,ub) , scl ],
        [ 'x1'  , 0., (-2.,2.) , 1.0 ],
        [ 'x2'  , 0., (-2.,2.) , 1.0 ], 
    ]
    
    # setup variables, arrays
    var = opt.Variable()
    var.tag     = 'x3'
    var.initial = np.array([10., 10., 10.])
    ub = np.array([30.]*3)
    lb = np.array([-1.]*3)
    var.bounds  = (lb,ub)
    var.scale   = opt.scaling.Linear(scale=4.0,center=10.0)
    problem.variables.append(var)
    
    # initialize evaluator
    test_eval = Test_Evaluator()
    
    # setup objective
    problem.objectives = [
    #   [ func   , 'tag', scl ],
        [ test_eval, 'f', 1.0 ],
    ]
    
    # setup constraint, list style
    problem.constraints = [
    #   [ func , ('tag' ,'><=', val), scl] ,
        [ test_eval, ('c','=',1.), 1.0 ],
    ]
    
    # setup constraint, array style
    con = opt.Equality()
    con.evaluator = test_eval
    con.tag       = 'c2'
    con.sense     = '='
    con.edge      = np.array([3.,3.,3.])
    problem.constraints.append(con)
  
    # print
    print problem  
    
    # expected answer
    truth = obunch()
    truth.variables  = obunch()
    truth.objectives = obunch()
    truth.equalities = obunch()
    
    truth.variables.x1 = -1.5
    truth.variables.x2 = -1.5
    truth.variables.x3 = np.array([ 4., 4., 4.])
    
    truth.objectives.f = 148.5
    
    truth.equalities.c  = 1.0
    truth.equalities.c2 = np.array([ 3., 3., 3.])
    
    problem.truth = truth    
  
    # done!
    return problem
    

# ----------------------------------------------------------------------
#   Call Main
# ----------------------------------------------------------------------

if __name__ == '__main__':
    main()