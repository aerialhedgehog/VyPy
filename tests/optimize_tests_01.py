
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
    x1 = inputs['x1']
    x2 = inputs['x2']
    
    f = x1**2 + x2**2
    c = x1 + x2
    
    outputs = {
        'f' : f,
        'c' : c,
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
    
    problem.objectives = [
    #   [ func   , 'tag', scl ],
        [ test_func, 'f', 1.0 ],
    ]
    
    problem.constraints = [
    #   [ func , ('tag' ,'><=', val), scl] ,
        [ test_func, ('c','=',1.), 1.0 ],
    ]
    
    problem.inequalities.keys()
    
    driver = opt.drivers.SLSQP()
    
    result = driver.run(problem)
    
    print result
    
    #problem.variables.get_scaled()
    #Evaluator.__init__()
    


if __name__ == '__main__':
    main()