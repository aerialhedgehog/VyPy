
import time, os, gc, sys, copy

import numpy as np

import VyPy
import VyPy.optimize as opt

import warnings
warnings.simplefilter('error',RuntimeWarning)



def main():
    
    test_1()
    
    pass


def test_func(inputs):
    x1 = inputs.x1
    x2 = inputs['x2']
    x3 = inputs.x3
    
    print inputs
    
    f = x1**2 + x2**2 + np.sum(x3)
    c = x1 + x2 + x3[0]
    c2 = x3 - 1.
    
    outputs = {
        'f' : f,
        'c' : c,
        'c2': c2,
    }
    return outputs
    

def test_1():
    
    problem = opt.Problem()
    
    problem.variables = [
    #   [ 'tag' , x0, (lb ,ub) , scl ],
        [ 'x1'  , 1., (-2.,2.) , 1.0 ],
        [ 'x2'  , 1., (-2.,2.) , 1.0 ], 
    ] #+ \
       #[[ 'x%i' , 1., (-2.,2.) , 1.0 ]]*20 
    
    var = opt.Variable()
    var.tag     = 'x3'
    var.initial = np.array([14.,10.,20.])
    ub = np.array([30.]*3)
    lb = np.array([-1.]*3)
    var.bounds  = (lb,ub)
    #var.scale   = opt.scaling.Linear(scale=4.0,center=10.0)
    problem.variables.append(var)
    
    problem.objectives = [
    #   [ func   , 'tag', scl ],
        [ test_func, 'f', 1.0 ],
    ]
    
    problem.constraints = [
    #   [ func , ('tag' ,'><=', val), scl] ,
        [ test_func, ('c','=',1.), 1.0 ],
    ]
    
    con = opt.Inequality()
    con.evaluator = test_func
    con.tag       = 'c2'
    con.sense     = '>'
    con.edge      = np.array([3.,3.,3.])
    problem.constraints.append(con)
    
    
    problem.inequalities.keys()
    
    driver = opt.drivers.SLSQP()
    
    result = driver.run(problem)
    
    print result
    
    #problem.variables.get_scaled()
    #Evaluator.__init__()
    


if __name__ == '__main__':
    main()