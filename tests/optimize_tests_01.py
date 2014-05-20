
import time, os, gc, sys, copy

import numpy as np

import VyPy
import VyPy.optimize as opt
from VyPy.data import obunch

import warnings
warnings.simplefilter('error',RuntimeWarning)



def main():
    
    test_1()
    
    pass


def test_func(inputs):
    x1 = inputs.x1
    x2 = inputs['x2']
    x3 = inputs.x3
    
    #print inputs
    
    f = x1**2 + x2**2 + np.sum(x3)**2
    c = x1 + x2 + x3[0]
    c2 = x3 - 1.
    
    outputs = obunch()
    outputs.f  = f
    outputs.c  = c
    outputs.c2 = c2
    
    #print outputs
    
    return outputs
    

def test_1():
    
    problem = opt.Problem()
    
    problem.variables = [
    #   [ 'tag' , x0, (lb ,ub) , scl ],
        [ 'x1'  , 0., (-2.,2.) , 1.0 ],
        [ 'x2'  , 0., (-2.,2.) , 1.0 ], 
    ] #+ \
       #[[ 'x%i' , 1., (-2.,2.) , 1.0 ]]*20 
    
    var = opt.Variable()
    var.tag     = 'x3'
    var.initial = np.array([10., 10., 10.])
    ub = np.array([30.]*3)
    lb = np.array([-1.]*3)
    var.bounds  = (lb,ub)
    var.scale   = opt.scaling.Linear(scale=4.0,center=10.0)
    problem.variables.append(var)
    
    problem.objectives = [
    #   [ func   , 'tag', scl ],
        [ test_func, 'f', 1.0 ],
    ]
    
    problem.constraints = [
    #   [ func , ('tag' ,'><=', val), scl] ,
        [ test_func, ('c','=',1.), 1.0 ],
    ]
    
    con = opt.Equality()
    con.evaluator = test_func
    con.tag       = 'c2'
    con.sense     = '='
    con.edge      = np.array([3.,3.,3.])
    problem.constraints.append(con)
    
    #driver = opt.drivers.SLSQP()
    #driver = opt.drivers.BFGS(n_eval=10000)   # doesnt respect design space bounds
    #driver = opt.drivers.COBYLA(n_eval=10000) # doesnt respect design space bounds
    driver = opt.drivers.CMA_ES(0)
    
    from time import time
    t0 = time()
    result = driver.run(problem)
    print '\nTime:' , time() - t0 , '\n'
    
    for r in result: print r
    

if __name__ == '__main__':
    main()