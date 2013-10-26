
# ----------------------------------------------------------------------
#   Constrained Optimization BY Linear Approximation
# ----------------------------------------------------------------------
class COBYLA(object):
    def __init__(self,iprint=1,n_eval=100,rho_scl=0.01):
        
        import scipy.optimize        
        
        self.iprint = iprint
        self.n_eval = n_eval
        self.rho_scl = rho_scl
    
    def run(self,problem):
        
        # store the problem
        self.problem = problem
        
        # single objective
        assert len(problem.objectives) == 1 , 'too many objectives'
        
        # optimizer        
        import scipy.optimize        
        optimizer = scipy.optimize.fmin_cobyla
        
        # inputs
        func   = self.func
        x0     = self.problem.variables.scaled.initial
        cons   = self.cons()
        rhobeg = self.rhobeg()
        rhoend = [ r*0.0001 for r in rhobeg ]
        iprint = self.iprint
        
        x_min = optimizer( 
            func      = func   ,
            x0        = x0     ,
            cons      = cons   ,
            rhobeg    = rhobeg ,
            rhoend    = rhoend ,
            iprint    = iprint ,
        )
        
        # nice: store result = fmin,xmin,iterations,time to problem history

        x_min = self.problem.variables.scaled.unpack(x_min)
        f_min = self.problem.objectives[0].evaluator.function(x_min)        
        
        # done!
        return f_min, x_min, None
        
    def func(self,x):
        objective = self.problem.objectives[0]
        result = objective.function(x)
        return result        
        
    def cons(self):        
        equalities   = self.problem.equalities
        inequalities = self.problem.inequalities
        
        cons = []
        for equality in inequalities:
            func = lambda (x): -equality.function(x)
            cons.append(func)
        for inequality in equalities:
            # build equality constraint with two inequality constraints
            func = inequality.function
            cons.append(func)
            func = lambda (x): -inequality.function(x)
            cons.append(func)
            
        return cons
    
    def rhobeg(self):
        bounds = self.problem.variables.scaled.bounds
        rho = []
        for b in bounds:
            lo,hi = b
            r = (hi-lo)*self.rho_scl
            rho.append(r)
        return rho
    
    
    