
from VyPy.optimize.drivers import Driver
import numpy as np

try:
    import cma
except ImportError:
    pass
    
# ----------------------------------------------------------------------
#   Covariance Matrix Adaptation - Evolutionary Strategy
# ----------------------------------------------------------------------
class CMA_ES(Driver):
    def __init__(self,iprint=1, rho_scl = 0.10, n_eval=None):
        
        try:
            import cma
        except ImportError:
            raise ImportError, 'Could not import cma, please install with pip: "> pip install cma"'
        
        self.iprint  = iprint
        self.rho_scl = rho_scl
        self.n_eval  = n_eval or np.inf
    
    def run(self,problem):
        
        # store the problem
        self.problem = problem
        
        # single objective
        assert len(problem.objectives) == 1 , 'too many objectives'
        
        # optimizer
        import cma
        optimizer = cma.fmin
        
        # inputs
        func   = self.func
        x0     = problem.variables.scaled.initials_array()
        x0     = np.squeeze(x0)
        sigma0 = self.sigma0()
        bounds = problem.variables.scaled.bounds_array()
        bounds = [ bounds[:,0] , bounds[:,1] ]
        
        options = {
            'bounds'    : bounds      ,
            'verb_disp' : self.iprint ,
            'verb_log'  : 0           ,
            'verb_time' : 0           ,
            'maxfevals' : self.n_eval ,
        }
        
        # run the optimizer
        result = optimizer( 
            objective_function = func    ,
            x0                 = x0      ,
            sigma0             = sigma0  ,
            options            = options ,
        )
        
        
        x_min = result[0].tolist()
        f_min = result[1]
        
        x_min = self.problem.variables.scaled.unpack_array(x_min)
        f_min = self.problem.objectives[0].evaluator.function(x_min)
        
        # done!
        return f_min, x_min, result

    def func(self,x):
        
        obj  = self.objective(x)[0,0]
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
            if res < 0.0: res = 0.0
            result.append(res)
        for equality in equalities:
            res = equality.function(x)
            result.append(res)
            
        if result:
            result = np.vstack(result)
            result = np.squeeze(result)
            
        return result
    
    def sigma0(self):
        bounds = self.problem.variables.scaled.bounds_array()
        sig0 = np.mean( np.diff(bounds) ) * self.rho_scl
        return sig0
        

