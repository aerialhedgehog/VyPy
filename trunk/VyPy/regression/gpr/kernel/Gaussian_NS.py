
class Gaussian_NS(Gaussian):
    
    def __init__(self,Train,Length):
        super(Gaussian_NS,self).__init__(Train)
        
        # non-stationary length-scale scaling function, range [0-1]
        self.Length = Length
        
    def evaluate_kernel(self,P,Q,n_dp,n_dq,block):
        ''' evaluate kernel
        '''
        
        # unpack hypers
        Hyp = self.Hypers
        sig_f   = 10**Hyp.sig_f
        len_s   = 10**Hyp.len_s
        sig_ny  = 10**Hyp.sig_ny
        sig_ndy = 10**Hyp.sig_ndy
        
        # unpack length function
        Len = self.Length
        
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
            diag_K = self.kernel_diag(P,Q,n_dp)
            return diag_K
        #: end
        
        # initialize covariance matrix
        K = np.zeros([ n_p*(1+n_dp) , n_q*(1+n_dq) ])        
        
        # Main subBlock - Point-Point
        K[0:n_p,0:n_q] = self.kernel_full(P,Q)
        
        # Derivative subBlock - Point-Deriv
        for i_w in range(1,n_dq+1):
          
            # index limits
            i_w1 = (i_w*n_q)
            i_w2 = (i_w+1)*n_q
                        
            # Covariance K (point-deriv)
            # Kb{1,iw}  = (dw)/len_s^2 .* Ka; 
            K[0:n_p,i_w1:i_w2] = self.kernel_derv(P,Q,i_w-1)
            
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
                K[i_v1:i_v2,0:n_q] = -1 * K[0:n_p,i_w1:i_w2]
              
            else:
                # Covariance K (point-deriv)
                # Kb{1,iw}  = (dw)/len_s^2 .* Ka;  
                K[i_v1:i_v2,0:n_q] = -1 * self.kernel_derv(P,Q,i_v-1)
                
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
                K[i_v1:i_v2,i_w1:i_w2] = self.kernel_hess(P,Q,i_v-1,i_w-1)
                
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
    
    #: def evaluate()
    
    def kernel_full(self,P,Q):

        # unpack hypers
        Hyp = self.Hypers
        sig_f   = 10**Hyp.sig_f
        len_s   = 10**Hyp.len_s
        sig_ny  = 10**Hyp.sig_ny
        sig_ndy = 10**Hyp.sig_ndy
        
        # unpack length function
        Len = self.Length
        
        # sizes
        P = atleast_2d(P); Q = atleast_2d(Q)
        n_p,n_d = P.shape
        n_q,_   = Q.shape
        
        # distances along each dimension
        d = P[:,None,:] - Q[None,:,:]
        
        # L2 distance matrix of training data
        D_2 = np.sum( d**2 , 2 )
        
        # non-stationary length scale factor function
        LP, LQt = Len(P), Len(Q).T
        LL = LP**2 + (LQt)**2
        
        # Main subBlock - Point-Point
        # Ka = ( 2/len*(tau(P)*tau(Q)') ./ LL ).^(nd/2) .* exp( -1/len^2 * 1./LL .* D_2 );
        K =  sig_f**2 * ( 2. * (LP*LQt) / LL )**(n_d/2.) \
                             * np.exp( -1./len_s**2 * 1./LL * D_2 )
        
        return K
        
    def kernel_diag(self,P,Q,n_dp):
        
        K_diag = [ np.array([ np.diag(self.kernel_full(P,Q)) ]) ] + \
                 [ np.array([ np.diag(self.kernel_hess(P,Q,i_p,i_p)) ]) for i_p in range(n_dp) ]
        
        K_diag = np.hstack(K_diag).T
        
        return K_diag
    
    def kernel_derv(self,P,Q,i_q=None):
        
        diff = differentiate.central_difference
        step = 1.e-4
        
        function = lambda(X): self.kernel_full(P,X)
        
        deriv = diff(function,Q,step,i_q)
        
        return deriv
    
    
    def kernel_hess(self,P,Q,i_p=None,i_q=None):
        
        diff = differentiate.central_difference
        step = 1.e-4
        
        function = lambda(X): self.kernel_derv(X,Q,i_q)
        
        deriv = diff(function,P,step,i_p)
        
        return deriv    
        