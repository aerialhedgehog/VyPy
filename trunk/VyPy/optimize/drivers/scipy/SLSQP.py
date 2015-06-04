
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import VyPy
from VyPy.data import ibunch
from VyPy.optimize.drivers import Driver
import numpy as np
from time import time
from VyPy.exceptions import MaxEvaluations

try:
    import scipy
    import scipy.optimize  
except ImportError:
    pass


# ----------------------------------------------------------------------
#   Sequential Least Squares Quadratic Programming
# ----------------------------------------------------------------------

class SLSQP(Driver):
    def __init__(self):
        ''' see http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_slsqp.html for more info
        '''
        
        # import check
        import scipy.optimize  
        
        Driver.__init__(self)
        
        self.verbose            = True
        self.max_iterations     = 1000
        self.max_evaluations    = 10000
        self.objective_accuracy = None
    
    def run(self,problem):
        
        # store the problem
        self.problem = problem

        # cache
        self._current_x = None
        self._current_eval = 0        
        
        # single objective
        assert len(problem.objectives) == 1 , 'too many objectives'
        
        # optimizer
        import scipy.optimize  
        optimizer = scipy.optimize.fmin_slsqp
        
        # inputs
        func           = self.func
        x0             = problem.variables.scaled.initials_array()
        f_eqcons       = self.f_eqcons
        f_ieqcons      = self.f_ieqcons
        bounds         = problem.variables.scaled.bounds_array()
        fprime         = self.fprime
        fprime_ieqcons = self.fprime_ieqcons
        fprime_eqcons  = self.fprime_eqcons  
        iprint         = 2
        iters          = self.max_iterations
        accuracy       = self.objective_accuracy or 1e-6
        
        ## objective scaling
        #accuracy = accuracy / problem.objective.scale
        
        # printing
        if not self.verbose: iprint = 0
        
        # constraints?
        if not problem.constraints.inequalities: f_ieqcons = None
        if not problem.constraints.equalities:   f_eqcons  = None
        
        # gradients?
        dobj,dineq,deq = problem.has_gradients()
        if not dobj: fprime = None
        if not (f_ieqcons and dineq): fprime_ieqcons = None
        if not (f_eqcons  and deq)  : fprime_eqcons  = None
        
        # for catching max_evaluations
        self._current_x = x0
        
        # start timing
        tic = time()
        
        # run the optimizer
        try: # for catching custom exits
            x_min,f_min,its,imode,smode = optimizer( 
                func           = func           ,
                x0             = x0             ,
                f_eqcons       = f_eqcons       ,
                f_ieqcons      = f_ieqcons      ,
                bounds         = bounds         ,
                fprime         = fprime         ,
                fprime_ieqcons = fprime_ieqcons ,
                fprime_eqcons  = fprime_eqcons  ,
                iprint         = iprint         ,
                full_output    = True           ,
                iter           = iters          ,
                acc            = accuracy       ,
                **self.other_options.to_dict()
            )
        except MaxEvaluations:
            its = None  # can't know major iterations unless gradients are provided
            imode = 10  # custom mode number
            smode = 'Evaluation limit exceeded'
            x_min = self._current_x
            
        ## TODO - check constraints are met to tolerance, scipy doesn't do this
        
        # stop timing
        toc = time() - tic
        
        # get final variables
        vars_min = self.problem.variables.scaled.unpack_array(x_min)
        
        # pack outputs
        outputs = self.pack_outputs(vars_min)
        outputs.success               = imode == 0
        outputs.messages.exit_flag    = imode
        outputs.messages.exit_message = smode
        outputs.messages.iterations   = its
        outputs.messages.evaluations  = self._current_eval
        outputs.messages.run_time     = toc
        
        # done!
        return outputs
            
    
    def func(self,x):
        
        # check number of evaluations
        max_eval = self.max_evaluations
        if max_eval and max_eval>0 and self._current_eval >= max_eval:
            raise MaxEvaluations
        
        self._current_x = x
        self._current_eval += 1        
        
        # evaluate the objective function
        objective = self.problem.objectives[0]
        result = objective.function(x)
        result = result[0,0]
        
        return result
        
    def f_ieqcons(self,x):
        inequalities = self.problem.inequalities
        result = [ -1.*inequality.function(x) for inequality in inequalities ]
        result = np.vstack(result)            
        result = result[:,0]
        return result
    
    def f_eqcons(self,x):
        equalities = self.problem.equalities
        result = [ equality.function(x) for equality in equalities ]
        result = np.vstack(result)
        result = result[:,0]
        return result

    def fprime(self,x):
        objective = self.problem.objectives[0]
        result = objective.gradient(x)
        result = result[0,:]
        return result
    
    def fprime_ieqcons(self,x):
        inequalities = self.problem.inequalities
        result = [ -1.*inequality.gradient(x) for inequality in inequalities ]
        result = np.vstack(result)
        return result
    
    def fprime_eqcons(self,x):
        equalities = self.problem.equalities
        result = [ equality.gradient(x) for equality in equalities ]
        result = np.vstack(result)
        return result
