
import numpy as np
import scipy as sp
import scipy.linalg
import copy

from VyPy import optimize, tools
from VyPy.exceptions import EvaluationFailure

class Learning(object):
    
    def __init__(self,Infer):
        """ GPR.Learning(Infer)
            
        """
        
        self.Infer  = Infer
        self.Kernel = Infer.Kernel
        self.Hypers = Infer.Kernel.Hypers
        self.Train  = Infer.Kernel.Train
        
        return

    def learn(self, hypers=None):
        """ Learning.learn( hypers=None )
            
            learn useful hyperparameters for the training data
        
        """
        
        print 'Hyperparameter Learing ...'
        
        Infer  = self.Infer
        Train  = self.Train
        Hypers = self.Hypers        
        
        if not hypers is None:
            Hypers.update(hypers)
              
        # setup the learning problem
        problem = optimize.Problem()
        problem = self.setup(problem)
        
        # Run Global Optimization
        print '  Global Optimization (CMA_ES)'
        driver = optimize.drivers.CMA_ES( rho_scl = 0.10  ,
                                          n_eval  = 1000 , # 1000
                                          iprint  = 0     )
        [logP_min,Hyp_min,result] = driver.run(problem)
        
        # setup next problem
        problem.variables.set(initials=Hyp_min)
        problem.objectives['logP'].scale = -1.0e-2
        
        # Run Local Refinement
        print '  Local Optimization (SLSQP)'
        driver = optimize.drivers.scipy.SLSQP( iprint = 0 )   
        [logP_min,Hyp_min,result] = driver.run(problem)
        
        #print 'Local Optimization (COBYLA)'
        #driver = opt.drivers.scipy.COBYLA()   
        #[logP_min,Hyp_min,result] = driver.run(problem)        
        
        # report
        print "  logP = " + str( logP_min )
        print "  Hyp  = " + ', '.join(['%s: %.4f'%(k,v) for k,v in Hyp_min.items()])
        
        # store
        Hyp_min = copy.deepcopy(Hyp_min)
        Hypers.update(Hyp_min)
        self.logP = logP_min
        
        try:
            Infer.precalc()
        except Exception:
            raise EvaluationFailure, 'learning failed, could not precalculate kernel'
        
        return
    
    #: def learn()
    
    def setup(self,problem):
        """ Kernel.setup(problem):
            
            setup the marginal likelihood maximization problem
            
            Inputs:
                problem - an empty VyPy Problem() class
                
            Outputs:
                None
            
        """
        
        # kernel setup and constraints
        Kernel = self.Kernel
        problem = Kernel.setup_learning(problem)
                
        return problem
    


