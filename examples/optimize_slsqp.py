
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import time, os, gc, sys, copy

import numpy as np

import VyPy
import VyPy.optimize as opt
from VyPy.data import obunch


# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

# This example explores how to solve an optimization problem with VyPy.
# 

def main():
    
    problem = setup_problem()
    optimize_problem(problem)
    

# ----------------------------------------------------------------------
#   The Evaluator
# ----------------------------------------------------------------------

# This section demonstrates the Evaluator() object.  Evaluators organize
# functions and gradients, inputs and outputs, for use with the VyPy 
# optimization drivers.  
#

class The_Evaluator(opt.Evaluator):
    
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
    
    def gradient(self,inputs):
        
        # unpack inputs
        x1 = inputs.x1
        x2 = inputs['x2']
        x3 = inputs.x3  
        
        # prep outputs
        grads = obunch()
        grads.f  = obunch()
        grads.c  = obunch()
        grads.c2 = obunch()
        
        # the math
        grads.f.x1 = 2*x1
        grads.f.x2 = 2*x2 
        grads.f.x3 = 2*x3
        grads.c.x1 = 1.
        grads.c.x2 = 1.
        grads.c.x3 = np.array([1.,0,0])
        grads.c2.x1 = np.zeros([3,1])
        grads.c2.x2 = np.zeros([3,1])
        grads.c2.x3 = np.eye(3)
        
        return grads
    
    def __init__(self):
        
        # cache the function and gradient evals
        self.function = VyPy.parallel.Remember(self.function)
        self.gradient = VyPy.parallel.Remember(self.gradient)    
    
    
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
    var.scale   = 1.0
    problem.variables.append(var)
    
    # initialize evaluator
    test_eval = The_Evaluator()
    
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
    con = opt.Constraint()
    con.evaluator = test_eval
    con.tag       = 'c2'
    con.sense     = '>'
    con.edge      = np.array([3.,3.,3.])
    problem.constraints.append(con)
  
    # print
    print problem  
      
    # done!
    return problem
    

# ----------------------------------------------------------------------        
#   Optimize the Problem
# ----------------------------------------------------------------------    

def optimize_problem(problem):
    
    # ------------------------------------------------------------------
    #   Setup Driver
    # ------------------------------------------------------------------    
    
    driver = opt.drivers.SLSQP()
    driver.verbose = False
    
    # ------------------------------------------------------------------
    #   Run the Problem
    # ------------------------------------------------------------------        

    results = driver.run(problem)
    
    print 'Results:'
    print results

    
    # done!
    return

# ----------------------------------------------------------------------
#   Call Main
# ----------------------------------------------------------------------

if __name__ == '__main__':
    main()