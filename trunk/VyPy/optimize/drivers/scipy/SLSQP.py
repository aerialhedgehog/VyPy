
from VyPy.optimize.drivers import Driver
import numpy as np

try:
    import scipy
    import scipy.optimize  
except ImportError:
    pass

# ----------------------------------------------------------------------
#   Sequential Least Squares Quadratic Programming
# ----------------------------------------------------------------------
class SLSQP(Driver):
    def __init__(self,iprint=1):
        
        import scipy.optimize  
        
        self.iprint = iprint
    
    def run(self,problem):
        
        # store the problem
        self.problem = problem
        
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
        iprint         = self.iprint
        
        # gradients?
        dobj,dineq,deq = problem.has_gradients()
        if not dobj : fprime         = None
        if not dineq: fprime_ieqcons = None
        if not deq  : fprime_eqcons  = None
        
        # run the optimizer
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
        )
        
        x_min = self.problem.variables.scaled.unpack_array(x_min)
        f_min = self.problem.objectives[0].evaluator.function(x_min)
        
        # done!
        return f_min, x_min, imode
    
    def func(self,x):
        objective = self.problem.objectives[0]
        result = objective.function(x)
        result = np.squeeze(result)
        return result
        
    def f_ieqcons(self,x):
        inequalities = self.problem.inequalities
        result = []
        for inequality in inequalities:
            res = inequality.function(x)
            res = -1 * res
            result.append(res)
        if result:
            result = np.vstack(result)
            result = np.squeeze(result)
        return result
    
    def f_eqcons(self,x):
        equalities = self.problem.equalities
        result = []
        for equality in equalities:
            res = equality.function(x)
            result.append(res)
        if result:
            result = np.vstack(result)
            result = np.squeeze(result)
        return result

    def fprime(self,x):
        objective = self.problem.objectives[0]
        result = objective.gradient(x)
        result = np.vstack(result)
        result = np.squeeze(result)
        return result
    
    def fprime_ieqcons(self,x):
        inequalities = self.problem.inequalities
        result = []
        for inequality in inequalities:
            res = inequality.gradient(x)
            res = -1 * res
            result.append(res)
        if result:
            result = np.vstack(result)
        return result
    
    def fprime_eqcons(self,x):
        equalities = self.problem.equalities
        result = []
        for equality in equalities:
            res = equality.gradient(x)
            result.append(res)
        if result:
            result = np.vstack(result)
        return result
