# automatic_regression.py
# 
# Created:  Trent Lukaczyk, Aug 2014
# Modified:     

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------  
        
import VyPy
from VyPy.data import OrderedBunch
import sys, os, traceback, time
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------        
#   How This Works
# ----------------------------------------------------------------------        

# automatic_regression.py runs test scripts that exercise the code to 
# catch bugs as soon as they appear
#
# The "modules" list contains the name of the file you would like to run.
# Each test script must include a main function, this will be called by
# this automatic regression script. 
#
# For more information, see ./example_test_script.py


# ----------------------------------------------------------------------
#   The Modules to Test
# ----------------------------------------------------------------------        

modules = [
    'optimize/problem_function.py',
    'optimize/problem_evaluator.py',
    'optimize/driver_slsqp.py',
    'optimize/driver_cma.py',
    'optimize/driver_cobyla.py',
    'optimize/driver_bfgs.py',
]


# ----------------------------------------------------------------------        
#   Main
# ----------------------------------------------------------------------    

def main():
    
    # preallocate test results
    results = OrderedBunch()
    for module in modules:
        results[module] = 'Untested'
        
    sys.stdout.write('# --------------------------------------------------------------------- \n')
    sys.stdout.write('#   VyPy Automatic Regression \n')
    sys.stdout.write('#   %s \n' % time.strftime("%B %d, %Y - %H:%M:%S", time.gmtime()) )
    sys.stdout.write('# --------------------------------------------------------------------- \n')
    sys.stdout.write(' \n')
        
    # run tests
    for module in modules:
        passed = test_module(module)
        if passed:
            results[module] = 'Passed'
        else:
            results[module] = 'Failed'
    
    # final report
    sys.stdout.write('# --------------------------------------------------------------------- \n')
    sys.stdout.write('#   Final Results \n')
    sys.stdout.write('# --------------------------------------------------------------------- \n')
    sys.stdout.write('\n')
    for module,result in results.items():
        sys.stdout.write('%s - %s\n' % (result,module))
        
    return


# ----------------------------------------------------------------------        
#   Module Tester
# ----------------------------------------------------------------------   

def test_module(module_path):
    
    home_dir = os.getcwd()
    test_dir, module_name = os.path.split( os.path.abspath(module_path) )
    
    sys.stdout.write('# --------------------------------------------------------------------- \n')
    sys.stdout.write('#   Test: %s \n' % module_path)
    sys.stdout.write('# --------------------------------------------------------------------- \n')
    sys.stdout.write('\n')
    sys.stdout.flush()
    
    tic = time.time()
    
    # try the test
    try:

        # see if file exists
        os.chdir(test_dir)
        if not os.path.exists(module_name) and not os.path.isfile(module_name):
            raise ImportError, 'file %s does not exist' % module_name
        
        # add module directory
        sys.path.append(test_dir)
        
        # do the import
        name = os.path.splitext(module_name)[0]
        module = __import__(name)        
        
        # run main function
        module.main()
        
        passed = True
        
    # catch an error
    except Exception as exc:
        
        # print traceback
        sys.stderr.write( 'Test Failed: \n' )
        sys.stderr.write( traceback.format_exc() )
        sys.stderr.write( '\n' )
        sys.stderr.flush()
        
        passed = False
        
    sys.stdout.write('\n')

    # final result
    if passed:
        sys.stdout.write('# Passed: %s \n' % module_name)
    else:
        sys.stdout.write('# Failed: %s \n' % module_name)
    sys.stdout.write('# Test Duration: %.4f min \n' % ((time.time()-tic)/60) )
    sys.stdout.write('\n')
    sys.stdout.write('\n')
    
    # cleanup
    plt.close('all')
    os.chdir(home_dir)
    
    # make sure to write to stdout
    sys.stdout.flush()    
    sys.stderr.flush()
    
    return passed


# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()