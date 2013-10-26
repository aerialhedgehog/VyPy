
import sys, copy, weakref,gc

import numpy as np
import scipy as sp
import scipy.linalg

from VyPy import opt

from IndexableBunch import IndexableBunch
import tools
from tools import vector_distance, check_array


class Modeling(object):
    
    def __init__(self,Kernel):
        
        # pack
        self.Kernel = Kernel
        self.Train  = Kernel.Train
        self.Hypers = Kernel.Hypers
        
        # useful data
        self.L     = []
        self.Yt    = []
        self.logP  = -1.e10
        self.DlogP = []
        self.Kcond = []
        self.RIE   = {}
        
        # reinterpolated model
        self.RI = None

        # try to precalc L
        self.safe_precalc()
        
        return
    
    #: def __init__()

    def precalc(self):
        ''' precalculate cholesky decomposition of K1
        '''
        # todo: check for changes to hypers, training
        
        # unpack
        Kernel = self.Kernel
        
        # evaluate first kernel subblock
        K1 = Kernel.K1()
        
        # build full training vector  
        Yt = Kernel.Yt()
        
        # try to solve 
        try:
            self.L  = np.linalg.cholesky( K1 )
            self.al = sp.linalg.cho_solve( (self.L,True), Yt )   # need to subtract out mean
            self.Yt = Yt 
        
        except np.linalg.LinAlgError:
            #print 'cholesky decomposition failed during precalc'
            raise
        
        return
    
    #: def precalc()
    
    def safe_precalc(self):
        """ will attempt to re-learn hyperparameters if precalc fails """
        try:
            self.precalc()
            
        except np.linalg.LinAlgError:
            print 'Cholesky decomposition failed, re-learn...' 
            self.learn()
            self.precalc()
            
        return
    
    #: def safe_precalc()
    
    def evaluate(self,XI):
        ''' Evaluate GPR fit at XI
        '''
        
        # unpack
        Kernel = self.Kernel
        L      = self.L
        al     = self.al
        Yt     = self.Yt
        XI     = check_array(XI)
                
        # covariance functions
        K3 = Kernel.K3(XI)
        diag_K4 = Kernel.diag_K4(XI)
        
        # the hard work
        v  = np.dot( L.T , sp.linalg.cho_solve( (L,True) , K3.T ) )
        
        # almost done
        YI_solve   = np.dot(K3,al) # + np.dot(R.T,B)
        CovI_solve = np.sqrt( np.abs( diag_K4 - np.array([ np.diag( np.dot(v.T,v) ) ]).T ) ) 
        # log probability?
        # lZ = -(y-mu).^2./(sn2+s2)/2 - log(2*pi*(sn2+s2))/2;    % log part function
        
        # pack up outputs 
        The_Data = Kernel.pack_outputs(XI,YI_solve,CovI_solve)
        
        return The_Data
    
    #: def evaluate()
    
    def reinterpolate(self):
            
        # new noise param (log10)
        sig_nR = -6.0
        
        # copy model data
        Train   = copy.deepcopy( self.Train   )
        Kernel  = copy.deepcopy( self.Kernel  )
        Hypers  = Kernel.Hypers # dont deepcopy
        probNze = copy.deepcopy( self.probNze )
        
        new_Model = Modeling(Kernel,probNze)
        new_Model.safe_precalc()
        
        # kernel output data names
        outputs = Kernel.outputs
        self.RIE = IndexableBunch()
                
        # evaluate training data with model
        X = Train.X
        The_Data = new_Model.evaluate(X)
        
        # operate on each training output
        for output in outputs:
            
            # pull old data
            D = Train[output]
            if D.shape[0] == 0: continue
            
            # pull new data
            output_i = output + 'I'
            DI = The_Data[output_i]
            
            # calculate errors
            RIE = np.log10( np.mean( np.abs( DI  - D ) ) )
            self.RIE[output] = RIE
            print 'RIE %s: %.4f' % (output,RIE)
            
            # reassign reinterpolated redata
            Train.setitem(output,The_Data[output_i])
            
        #: for each output    
        
        # reinterpolated implies no noise
        new_Model.Hypers.sig_ny  = sig_nR
        new_Model.Hypers.sig_ndy = sig_nR
        
        # store
        self.RI = new_Model
                
        return
    
    #: def reinterpolate()   
    
    def learn(self):
            
        # unpack
        Train  = self.Train
        Kernel = self.Kernel    
        
        problem = opt.Problem()
        
        problem.objectives = [
        #   [ function_handle , 'output' ,  scl ] , 
            [ self.likelihood , 'logP'   , -1.0 ] , # maximize
        ]
        
        Kernel.setup_learning(problem)
        
        problem.compile()
        
        # run global optimization
        print 'Global Optimization (CMA_ES)'
        driver = opt.drivers.CMA_ES( rho_scl = 0.10 ,
                                      n_eval  = 1000 ,
                                      iprint  = 0     )
        [logP_min,Hyp_min,result] = driver.run(problem)
        
        # run local refinement
        problem.variables.update_initial(Hyp_min)
        problem.objectives[0].scale = -1.0e-2        
        print 'Local Optimization (SLSQP)'
        driver = opt.drivers.scipy.SLSQP( iprint = 0 )   
        [logP_min,Hyp_min,result] = driver.run(problem)
        
        #print 'Local Optimization (COBYLA)'
        #driver = opt.drivers.scipy.COBYLA()   
        #[logP_min,Hyp_min,result] = driver.run(problem)        
        
        # report
        print "logP = " + str( logP_min )
        print "Hyp  = " + ', '.join(['%s: %.4f'%(k,v) for k,v in Hyp_min.items()])
        
        # store
        Hyp_min = copy.deepcopy(Hyp_min)
        self.Kernel.Hypers.update(Hyp_min)
        #self.Kernel.unpack_vec(Hyp_min)
        
        try:
            self.precalc()
        except sp.linalg.LinAlgError:
            raise sp.linalg.LinAlgError , 'Learning failed, cannot precalculate kernel'
            
        return
    
    #: def learn() 
    
    def likelihood(self,Hyp_dict):
        try:
            # unpack
            Train  = self.Train
            Kernel = self.Kernel
            
            # unpack hypers
            Kernel.Hypers.update( Hyp_dict )
            n_hv = len( Hyp_dict )
            
            # try precalculation
            try:
                self.precalc()
            except sp.linalg.LinAlgError:
                raise tools.Evaluation_Failure
            #: check cholesky
            
            # pull result
            L  = self.L
            al = self.al
            Yt = self.Yt
            
            # unpack training data
            X  = Train.X
            Y  = Train.Y
            DY = Train.DY
            XB = Train.XB
            
            # shizes
            n_x = L.shape[0]
            #_  ,n_x  = X.shape
            #n_t,n_y  = Y.shape
            #_  ,n_dy = DY.shape   
            
            # likelihood 
            logP = -1/2* np.dot(Yt.T,al) - np.sum( np.log( np.diag(L) ) ) - n_x/2*np.log(2*np.pi)
            logP = logP[0,0]
            
            ## likelihood gradients
            #dK1 = Kernel.grad_hypers(X,n_dy,Hyp_vec)
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
            #self.logP  = logP
            
        # - failed ouptus -------
        except tools.Evaluation_Failure:
            outputs = {
                'logP'  : -1e10  ,
                #'DlogP' : DlogP ,
            }
        
        return outputs
        
    #: def likelihood()    
    
    def likelihood_grad(self,Hyp_vec,f,g):
        
        new_hyp = self.Kernel.unpack_vec(Hyp_vec)
        
        if not new_hyp:
            self.likelihood(Hyp_vec)
        
        return -self.DlogP
    
    #: def likelihood_grad()
    
    
    def pyopt_function(self,X):
        return self.evaluate(X).YI, [], 0
    def pyopt_gradient(self,X,f_,g_):
        return self.evaluate(X).DYI, [], 0
    
#: class Modeling()