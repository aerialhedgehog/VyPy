
import numpy as np
import scipy as sp
import scipy.linalg
import copy

from Learning import Learning

from VyPy import optimize, tools
from VyPy.exceptions import EvaluationFailure

class Likelihood(Learning):
    
    def __init__(self,Infer):
        """ GPR.Learning(Infer)
            
        """
        
        Learning.__init__(self,Infer)
        
        self.logP   = None
        
        return
    
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
        
        ## done if objectives are setup
        #if problem.objectives: return problem
        
        # log likelihood objective function
        problem.objectives = [
        #   [ function_handle     , 'output' ,  scl ] , 
            [ self.likelihood_obj , 'logP'   , -1.0 ] , # maximize
        ]
                
        return problem
    
    def likelihood_obj(self,hypers):
        try:
            
            # unpack
            Infer  = self.Infer
            Train  = self.Train
            Kernel = self.Kernel
            Hypers = self.Hypers
            
            # update hyperparameters
            Hypers.update(hypers)
            
            # try precalculation
            L, al, Yt = Infer.precalc()
            
            # shizes
            n_x = L.shape[0]
            
            # log likelihood 
            logP = -1/2* np.dot(Yt.T,al) - np.sum( np.log( np.diag(L) ) ) - n_x/2*np.log(2*np.pi)
            logP = logP[0,0]
            
            ## likelihood gradients
            #_  ,n_x  = Train.X.shape
            #n_t,n_y  = Train.Y.shape
            #_  ,n_dy = Train.DY.shape               
            #dK1 = Kernel.grad_hypers(Train.X,n_dy,Hyp_vec)
            #al_Sym = np.dot( al, al.T )
            #for i_h in range(n_hv):
                #DlogP[0,i_h] = 0.5*np.trace( np.dot(al_Sym,dK1[:,:,i_h]) 
                #                           - np.linalg.solve(L.T,np.linalg.solve(L,dK1[:,:,i_h])) )
            
        # - successfull ouputs -------
            # save
            outputs = {
                'logP'  : logP  ,
                #'DlogP' : DlogP ,
            }
            
        # - failed ouptus -------
        except EvaluationFailure:
            outputs = {
                'logP'  : -1e10  ,
                #'DlogP' : DlogP ,
            }
        
        return outputs
        
    #: def likelihood()    


