
from VyPy.optimize import Driver
import numpy

try:
    import cma
    is_loaded = True
except ImportError:
    from VyPy.plugins import cma
    is_loaded = True
except ImportError:
    is_loaded = False
    
# ----------------------------------------------------------------------
#   Covariance Matrix Adaptation - Evolutionary Strategy
# ----------------------------------------------------------------------
class CMA_ES(Driver):
    def __init__(self,iprint=1., rho_scl = 0.10, n_eval=None):
        
        if not is_loaded:
            raise ImportError, 'Could not import cma, please install from: https://www.lri.fr/~hansen/cma.py'
        
        self.iprint  = iprint
        self.rho_scl = rho_scl
        self.n_eval  = n_eval or numpy.inf
    
    def run(self,problem):
        
        # store the problem
        self.problem = problem
        
        # single objective
        assert len(problem.objectives) == 1 , 'too many objectives'
        
        # optimizer
        optimizer = cma.fmin
        
        # inputs
        func   = self.func
        x0     = problem.variables.scaled.initial
        sigma0 = self.sigma0()
        bounds = problem.variables.scaled.bounds
        bounds = map(list, zip(*bounds))
        
        # run the optimizer
        result = optimizer( func      = func   ,
                            x0        = x0     ,
                            sigma0    = sigma0 ,
                            bounds    = bounds ,
                            verb_disp = self.iprint ,
                            verb_log  = 0      ,
                            verb_time = 0      ,
                            maxfevals = self.n_eval )
        
        
        x_min = result[0].tolist()
        f_min = result[1]
        
        x_min = self.problem.variables.scaled.unpack(x_min)
        f_min = self.problem.objectives[0].evaluator.function(x_min)
        
        # done!
        return f_min, x_min, result

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
    
    def sigma0(self):
        bounds = self.problem.variables.scaled.bounds
        sig0 = []
        for b in bounds:
            lo,hi = b
            s = (hi-lo)*self.rho_scl
            sig0.append(s)
        sig0 = sum(sig0)/len(sig0)
        return sig0
        

