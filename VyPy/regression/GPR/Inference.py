

from VyPy import tools, EvaluationFailure

class Inference(object):
    
    def __init__(self,Kernel):
        
        self.Kernel = Kernel
        self.Train  = Kernel.Train
        
        return
    
    def __call__(self,XI):
        return self.evaluate(XI)
    

    def precalc(self):
        ''' precalculate cholesky decomposition of K1
        '''
        
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
            raise EvaluationFailure , 'cholesky decomposition failed during precalc'
        
        return L, al, Yt
    
    #: def precalc()
    
    def predict(self,XI):
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
        try:
            data = Kernel.pack_outputs(XI,YI_solve,CovI_solve)
        except NotImplementedError:
            data = [YI_solve,CovI_solve]
        
        return data
    
    #: def predict()
    