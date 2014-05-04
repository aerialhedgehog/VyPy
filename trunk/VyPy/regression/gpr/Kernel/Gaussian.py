import numpy as np
import scipy as sp

from Kernel import Kernel
from VyPy.data import IndexableBunch
from VyPy.tools import atleast_2d, vector_distance
from VyPy.exceptions import EvaluationFailure

class Gaussian(Kernel):
    ''' Gaussian Covariance Matrix with Derivatives 
        Hyp = sig_f,len_s,sig_ny,sig_ndy, all scalars
        probNze nominal noize for learning
    '''
    
    def __init__(self,Train,**hypers):
        
        super(Gaussian,self).__init__(Train,**hypers)
        
        if hypers.has_key('probNze'):
            self.Hypers['sig_ny']  = hypers['probNze']
            self.Hypers['sig_ndy'] = hypers['probNze']
        return 
    
    def default_hypers(self):
        
        # initialize hypers
        self.Hypers['sig_f']   = np.log10(1.0)
        self.Hypers['len_s']   = np.log10(0.5)
        self.Hypers['sig_ny']  = -6.0
        self.Hypers['sig_ndy'] = -6.0
        self.Hypers['probNze'] = -6.0
        
        return 
    
    def evaluate_kernel(self,P,Q,n_dp,n_dq,block):
        ''' evaluate kernel
        '''
        
        # unpack hypers
        Hyp = self.Hypers
        sig_f   = 10**Hyp['sig_f']
        len_s   = 10**Hyp['len_s']
        sig_ny  = 10**Hyp['sig_ny']
        sig_ndy = 10**Hyp['sig_ndy']
                        
        # sizes
        P = atleast_2d(P); Q = atleast_2d(Q)
        n_p,n_d = P.shape
        n_q,_   = Q.shape
        
        # noise terms
        if block in ['K1','diag_K4']:
            assert n_p == n_q , '%s should be square' % block
        #: end
        
        # diagonalized K4 block
        if block == 'diag_K4':
            diag_K = np.vstack([ sig_f**2*np.ones([n_p,1]) , 
                                 sig_f**2/len_s**2*np.ones([n_p*n_dp,1]) ])  
            return diag_K
        #: end
        
        # distances along each dimension
        d = P[:,None,:] - Q[None,:,:]
        
        # L2 distance matrix of training data
        D_2 = np.sum( d**2 , 2 )

        # initialize covariance matrix
        K = np.zeros([ n_p*(1+n_dp) , n_q*(1+n_dq) ])
        
        # Build Correlation Matrix with Derivatives!
        
        # Main subBlock - Point-Point
        # Ka  = sigf^2 * exp( -0.5/len_s^2 * (D).^2 );
        K[0:n_p,0:n_q] = sig_f**2 * np.exp( -0.5/len_s**2 * D_2 )
        
        # Derivative subBlock - Point-Deriv
        # Kb{1,iw}  = (dw)/len_s^2 .* Ka; 
        for i_w in range(1,n_dq+1):
          
            # index limits
            i_w1 = (i_w*n_q)
            i_w2 = (i_w+1)*n_q
                        
            # Covariance K (point-deriv)
            # Kb{1,iw}  = (dw)/len_s^2 .* Ka; 
            K[0:n_p,i_w1:i_w2] = (d[:,:,i_w-1])/len_s**2 * K[0:n_p,0:n_q]
            
        #: for i_w
            
        # Derivative subBlock - Deriv-Point
        for i_v in range(1,n_dp+1):
              
            # index limits
            i_v1 = (i_v*n_p)
            i_v2 = (i_v+1)*n_p
            
            if n_dq > 0:
                # index limits
                i_w1 = (i_v*n_q)
                i_w2 = (i_v+1)*n_q
              
                # Covariance K (deriv-point)
                # Kc{iw,1}  = -Kb{1,iw};
                K[i_v1:i_v2,0:n_q] = -K[0:n_p,i_w1:i_w2]
              
            else:
                # Covariance K (point-deriv)
                # Kb{1,iw}  = (dw)/len_s^2 .* Ka;  
                K[i_v1:i_v2,0:n_q] = -(d[:,:,i_v-1])/len_s**2 * K[0:n_p,0:n_q] 
                
        #: for i_v
                 
        # Derivative subBlock - Deriv-Deriv
        for i_v in range(1,n_dp+1):
            
            for i_w in range(1,n_dq+1):
          
                # index limits
                i_v1 = (i_v*n_p)
                i_v2 = (i_v+1)*n_p
                i_w1 = (i_w*n_q)
                i_w2 = (i_w+1)*n_q
                            
                # dirac delta function
                dvdw = int(i_v==i_w)
            
                # Covariance K (deriv-deriv)
                # Kd{iv,iw}  = ( dwdv/len_s^2 - (dv.*dw)/len_s^4 ) .* Ka;
                K[i_v1:i_v2,i_w1:i_w2] = ( dvdw/len_s**2 - (d[:,:,i_v-1]*d[:,:,i_w-1])/len_s**4 ) * K[0:n_p,0:n_q]
                
            #: for i_w
            
        #: for i_v
        
        # Noise
        if block=='K1':
            
            # build diagonal noise term
            diag_SN = np.hstack([ sig_ny**2 *np.ones([n_p])     , 
                                  sig_ndy**2*np.ones([n_p*n_dp]) ])                 
                   
            # Add to covariance matrix
            K = K + np.diag( diag_SN )
            
        #: if noise
            
        return K
    
    def K1(self):
        """ Kernel.K1(Train=None)
            first covariance block - training to training
            enforces square symmetric
            adds noise Kernel.Hypers.sig_ny and sig_ndy
            optional Train for different training data
        """
        
        Train = self.Train
        X  = Train.X
        DY = Train.DY
        
        n_tdy,n_dy = DY.shape
        if n_tdy==0: n_dy=0        
        
        K1 = self.evaluate_kernel(X,X,n_dy,n_dy,'K1')
        
        return K1
    
    def K3(self,XI):
        """ Kernel.K3(XI)
            third covariance block - estimate to training
            no noise added
        """
        
        Train = self.Train
        X  = Train.X
        DY = Train.DY
        
        n_t ,n_x   = X.shape
        n_tdy,n_dy = DY.shape
        if n_tdy==0: n_dy=0   
        
        K3 = self.evaluate_kernel(XI,X,n_x,n_dy,'K3')  
        
        return K3
    
    def diag_K4(self,XI):
        """ Kernel.K4(XI):
            diagonal of fourth covariance block - estimate to estimate
            enforces square symmetric
            no noise added
        """

        Train = self.Train
        X  = Train.X
        DY = Train.DY
        
        n_t ,n_x   = X.shape
        n_tdy,n_dy = DY.shape
        if n_tdy==0: n_dy=0   
        
        diag_K4 = self.evaluate_kernel(XI,XI,n_x,n_x,'diag_K4')
        
        return diag_K4
    
    def Yt(self):
        
        Train = self.Train
        Y  = Train.Y
        DY = Train.DY
        
        n_t  ,_    = Y.shape
        n_tdy,n_dy = DY.shape
        if n_tdy==0: n_dy=0
        
        Yt = np.zeros([ n_t*(1+n_dy) , 1 ])
        Yt[0:n_t,:] = Y
        for i_d in range(1,n_dy+1):
            Yt[i_d*n_t:(i_d+1)*n_t,:] = np.array([ DY[:,i_d-1] ]).T
        
        return Yt
    
    def pack_outputs(self,XI,YIt,CovIt):
        
        Train = self.Train
        X  = Train.X
        DY = Train.DY
        
        n_t ,n_x  = X.shape
        ntdy,n_dy = DY.shape
        n_i ,_    = XI.shape
        if ntdy==0: n_dy=0
        
        # pack up outputs
        YI     = YIt[0:n_i,:].copy()
        CovYI  = CovIt[0:n_i,:].copy()
        DYI    = np.zeros([n_i,n_x]) 
        CovDYI = np.zeros([n_i,n_x])
        for i_d in range(1,n_x+1):
            DYI[:,i_d-1]    = YIt[i_d*n_i:(i_d+1)*n_i,0].copy()
            CovDYI[:,i_d-1] = CovIt[i_d*n_i:(i_d+1)*n_i,0].copy()
        
        # pack data
        The_Data = IndexableBunch()
        The_Data['XI']     = XI
        The_Data['YI']     = YI
        The_Data['CovYI']  = CovYI
        The_Data['DYI']    = DYI
        The_Data['CovDYI'] = CovDYI
        
        return The_Data
            

    def setup_learning(self,problem):
        
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
        DX_min,DX_max,_ = vector_distance(X);
        DY_min,DY_max,_ = vector_distance(Y);
        
        if DX_min < 1e-10: DX_min = 1e-3;
        if DY_min < 1e-10: DY_min = 1e-3;
        
        sig_lo = np.log10(DY_min)-2.
        sig_hi = np.log10(DY_max)+2.
        len_lo = np.log10(DX_min)-2.
        len_hi = np.log10(DX_max)+1.
        
        # noise ranges
        #nze_lo = max([ probNze-bndNze , -10.  ])
        #nze_hi = max([ probNze+bndNze , -0.01 ])
        nze_lo = probNze-bndNze
        nze_hi = probNze+bndNze
        
        # some noise limits
        max_noise_ratio = -1.0
        min_noise_ratio = -8.0
        Kcond_limit     = -12.0         
        
        # set variables and bound constraints
        problem.variables = [
        #   ['tag'     , x0               , (lb,ub)        , scl ] ,
            ['sig_f'   , Hypers['sig_f']  , (sig_lo,sig_hi), 1.0 ] ,
            ['len_s'   , Hypers['len_s']  , (len_lo,len_hi), 1.0 ] ,
            ['sig_ny'  , Hypers['sig_ny'] , (nze_lo,nze_hi), 1.0 ] ,
            ['sig_ndy' , Hypers['sig_ndy'], (nze_lo,nze_hi), 1.0 ] ,
        ]
        
        problem.constraints = [
        #   [ function_handle     , ('output'    ,'><=',  val), scl] ,
            [ self.learning_cons , ('nze_rat_y' ,'<', max_noise_ratio), 1. ] ,
            [ self.learning_cons , ('nze_rat_y' ,'>', min_noise_ratio), 1. ] ,
            [ self.learning_cons , ('nze_rat_dy','<', max_noise_ratio), 1. ] ,
            [ self.learning_cons , ('nze_rat_dy','>', min_noise_ratio), 1. ] ,
            [ self.learning_cons , ('rel_nze'   ,'<', 0.0            ), 1. ] ,
            [ self.learning_cons , ('nze_dev'   ,'<', 0.0            ), 1. ] ,
            #[ self.likelihood_cons , ('k_cond'    ,'>', Kcond_limit    ), 1. ] ,
        ]
                
        return problem

    def learning_cons(self,hypers_dict):
        try:
            # unpack
            Hypers = self.Hypers 
            
            # unpack hypers
            Hypers.update(hypers_dict)
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
                'nze_rat_y'  : noise_ratio_y        , 
                'nze_rat_dy' : noise_ratio_dy       ,    
                'rel_nze'    : sig_ny  - sig_ndy    ,  
                'nze_dev'    : sig_ndy - sig_ny - 1 ,  
                #'k_cond'     : np.log10(Kcond)     , # expensive
            }
            
        # - failed ouputs -------
        except EvaluationFailure:
            raise
        
        return constraints
    

    