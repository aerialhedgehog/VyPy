
from VyPy.optimize.drivers import Driver

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
        x0     = problem.variables.scaled.initial
        
        # run the optimizer
        result = optimizer( f       = func   ,
                            x0      = x0     ,
                            fprime  = fprime ,
                            maxiter = self.n_eval )
        # done!
        return result

    def func(self,x):
        
        obj  = self.objective(x)
        
        # penalty for constraints
        cons = self.constraints(x)
        cons = [c**2 for c in cons]
        
        result = obj + sum(cons)
        
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
            if res < 0.0: res = 0.0
            result.append(res)
        for equality in equalities:
            res = equality.function(x)
            result.append(res)
            
        return result
    

