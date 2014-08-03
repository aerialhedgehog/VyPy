# example_test_script.py
# 
# Created:  Your Name, Jun 2014
# Modified:     


# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import VyPy
import numpy as np

import VyPy.optimize as opt


# ----------------------------------------------------------------------        
#   Main
# ----------------------------------------------------------------------    

def main():
    """ test the COBYLA Optimizer, with a test problem function """
    
    # ------------------------------------------------------------------
    #   Get the problem
    # ------------------------------------------------------------------
    
    from problem_function import setup_problem
    problem = setup_problem()
    
    
    # ------------------------------------------------------------------
    #   Setup Driver
    # ------------------------------------------------------------------    
    
    driver = opt.drivers.COBYLA()
    driver.max_evaluations  = 10000
    driver.verbose          = True
    
    
    # ------------------------------------------------------------------
    #   Run the Problem
    # ------------------------------------------------------------------        

    results = driver.run(problem)
    
    print 'Results:'
    print results

    
    # ------------------------------------------------------------------
    #   Check Results
    # ------------------------------------------------------------------
    
    # the expected results
    truth = problem.truth
    
    # the checking function
    def check(a,b):
        return a-b
    
    delta = truth.do_recursive(check,results)
    
    print 'Errors Against Expected:'
    print delta
    
    assert np.all( delta.pack_array() < 1e-3 )
    assert len( delta.pack_array() ) == 10
    
    # done!
    return

#: def main()


# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    
    main()
