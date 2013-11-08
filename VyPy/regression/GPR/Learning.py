
import numpy as np
import scipy as sp
import scipy.linalg
import copy

from VyPy import optimize, tools, EvaluationFailure

class Learning(object):
    
    def __init__(self,Infer):
        """ GPR.Learning(Infer)
            
        """
        
        self.Infer  = Infer
        self.Kernel = Infer.Kernel
        self.Hypers = Infer.Kernel.Hypers
        self.Train  = Infer.Kernel.Train
        
        return
    
    def setup(self,problem):
        """ Kernel.setup(problem):
            
            setup the marginal likelihood maximization problem
            
            Inputs:
                problem - an empty VyPy Problem() class
                
            Outputs:
                None
            
        """
        
        Train  = self.Train
        Hypers = self.Hypers
        
        # variables and bound constraints
        X  = Train.X
        Y  = Train.Y
        DY = Train.DY     
        
        # noise constraints
        probNze = Hypers['probNze']
        bndNze  = 2.0
        
        # feature and target ranges
        DX_min,DX_max,_ = tools.vector_distance(X);
        DY_min,DY_max,_ = tools.vector_distance(Y);
        sig_lo = np.log10(DY_min)-2.
        sig_hi = np.log10(DY_max)+2.
        len_lo = np.log10(DX_min)-2.
        len_hi = np.log10(DX_max)+1.
        
        # noise ranges
        nze_lo = max([ probNze-bndNze , -10.  ])
        nze_hi = min([ probNze+bndNze , -0.01 ])
        
        # some noise limits
        max_noise_ratio = -1.0
        min_noise_ratio = -8.0
        Kcond_limit     = -12.0         
        
        # set variables and bound constraints
        problem.variables = [
        #   ['Name'    , (lb    , x0              , ub   ), scl ] ,
            ['sig_f'   , (sig_lo,Hypers['sig_f']  ,sig_hi), 1.0 ] ,
            ['len_s'   , (len_lo,Hypers['len_s']  ,len_hi), 1.0 ] ,
            ['sig_ny'  , (nze_lo,Hypers['sig_ny'] ,nze_hi), 1.0 ] ,
            ['sig_ndy' , (nze_lo,Hypers['sig_ndy'],nze_hi), 1.0 ] ,
        ]
        
        #function = self.likelihood_obj
        #gradient = opt.drivers.FiniteDifference(function,step=1e-6)
        #objective = Evaluator(function,gradient)
        
        problem.objectives = [
        #   [ function_handle     , 'output' ,  scl ] , 
            [ self.likelihood_obj , 'logP'   , -1.0 ] , # maximize
        ]
        
        problem.constraints = [
        #   [ function_handle      , ('output'    ,'><=',           val), scl] ,
            [ self.likelihood_cons , ('nze_rat_y' ,'<', max_noise_ratio), 1. ] ,
            [ self.likelihood_cons , ('nze_rat_y' ,'>', min_noise_ratio), 1. ] ,
            [ self.likelihood_cons , ('nze_rat_dy','<', max_noise_ratio), 1. ] ,
            [ self.likelihood_cons , ('nze_rat_dy','>', min_noise_ratio), 1. ] ,
            [ self.likelihood_cons , ('rel_nze'   ,'<', 0.0            ), 1. ] ,
            [ self.likelihood_cons , ('nze_dev'   ,'<', 0.0            ), 1. ] ,
            #[ self.likelihood_cons , ('k_cond'    ,'>', Kcond_limit    ), 1. ] ,
        ]
                
        return 
    
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

    def likelihood_cons(self,hypers):
        try:
            # unpack
            Train  = self.Train
            Kernel = self.Kernel
            Hypers = Kernel.Hypers
            
            # update hyperparameters
            Hypers.update(hypers)
    
            # unpack hypers
            Hypers.update(hypers)
            sig_f   = Hypers['sig_f']
            len_s   = Hypers['len_s']
            sig_ny  = Hypers['sig_ny']
            sig_ndy = Hypers['sig_ndy']
            
            # noise ratios
            noise_ratio_y   = sig_ny  - sig_f
            noise_ratio_dy  = sig_ndy - (sig_f-len_s)
            
            # kernel condition
            #Kcond = self.Kcond # expensive...
            
        # - successfull ouputs -------
            # the constraints
            constraints = {
                'nze_rat_y'  : noise_ratio_y       , 
                'nze_rat_dy' : noise_ratio_dy      ,    
                'rel_nze'    : sig_ny - sig_ndy    ,  
                'nze_dev'    : sig_ndy- sig_ny - 1 ,  
                #'k_cond'     : np.log10(Kcond)     , # expensive
            }
            
        # - failed ouputs -------
        except EvaluationFailure:
            raise
        
        return constraints
    
    #: def likelihood_cons()

    def learn(self, hypers=None):
        """ Learning.learn( hypers=None )
            
            learn useful hyperparameters for the training data
        
        """
        
        Infer  = self.Infer
        Hypers = self.Hypers        
        
        if not hypers is None:
            Hypers.update(hypers)
              
        # setup the learning problem
        problem = optimize.Problem()
        self.setup(problem)
        problem.compile()
        
        # Run Global Optimization
        print 'Global Optimization (CMA_ES)'
        driver = optimize.drivers.CMA_ES( rho_scl = 0.10  ,
                                          n_eval  = 1000 ,
                                          iprint  = 0     )
        [logP_min,Hyp_min,result] = driver.run(problem)
        
        # setup next problem
        problem.variables.update_initial(Hyp_min)
        problem.objectives[0].scale = -1.0e-2        
        
        # Run Local Refinement
        print 'Local Optimization (SLSQP)'
        driver = optimize.drivers.scipy.SLSQP( iprint = 0 )   
        [logP_min,Hyp_min,result] = driver.run(problem)
        
        #print 'Local Optimization (COBYLA)'
        #driver = opt.drivers.scipy.COBYLA()   
        #[logP_min,Hyp_min,result] = driver.run(problem)        
        
        # report
        print "logP = " + str( logP_min )
        print "Hyp  = " + ', '.join(['%s: %.4f'%(k,v) for k,v in Hyp_min.items()])
        
        # store
        Hyp_min = copy.deepcopy(Hyp_min)
        Hypers.update(Hyp_min)
        
        try:
            Infer.precalc()
        except EvaluationFailure:
            raise EvaluationFailure, 'learning failed, could not precalculate kernel'
        
        return
    
    #: def learn()