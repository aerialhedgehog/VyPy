
from VyPy.optimize.drivers import Driver
import numpy as np

try:
    import scipy
    import scipy.optimize  
except ImportError:
    pass

# ----------------------------------------------------------------------
#   Broyden-Fletcher-Goldfarb-Shanno Algorithm
# ----------------------------------------------------------------------
class BFGS(Driver):
    def __init__(self,iprint=1,n_eval=100):
        
        import scipy.optimize
        
        self.iprint  = iprint
        self.n_eval  = n_eval
    
    def run(self,problem):
        
        # store the problem
        self.problem = problem
        
        # single objective
        assert len(problem.objectives) == 1 , 'too many objectives'
        
        # optimizer
        import scipy.optimize
        optimizer = scipy.optimize.fmin_bfgs
        
        # inputs
        func   = self.func
        fprime = None
        x0     = problem.variables.scaled.initials_array()
        
        # run the optimizer
        x_min = optimizer( f       = func   ,
                           x0      = x0     ,
                           fprime  = fprime ,
                           maxiter = self.n_eval )
        
        x_min = self.problem.variables.scaled.unpack_array(x_min)
        f_min = self.problem.objectives[0].evaluator.function(x_min)        
        
        # done!
        return f_min, x_min, None

    def func(self,x):
        
        obj = self.objective(x)[0,0]
        cons = self.constraints(x)
        
        # penalty for constraints
        result = obj + sum( cons**2 ) * 100000.0
        
        return result
            
    def objective(self,x):
        objective = self.problem.objectives[0]
        result = objective.function(x)
        return result
        
    def constraints(self,x):
        equalities   = self.problem.equalities
        inequalities = self.problem.inequalities
        
        result = []
        
        for inequality in inequalities:
            res = inequality.function(x)
            res[res<0.0] = 0.0
            result.append(res)
        for equality in equalities:
            res = equality.function(x)
            result.append(res)
            
        if result:
            result = np.vstack(result)
            result = np.squeeze(result)
            
        return result
    

