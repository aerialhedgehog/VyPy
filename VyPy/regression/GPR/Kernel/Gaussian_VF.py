
class Gaussian_VF(object):
    def __init__(self,Trains):
        """ List of Training Data
        """
        
        # initial hyperparameters
        rho = np.log10(1.0)
        probNze = -6.0
        
        Kernels = IndexableBunch()
        Hypers  = IndexableBunch()
        
        # pack data
        for key,this_Train in Trains.items():
            this_Kernel = Gaussian(this_Train)
            Kernels[key] = this_Kernel
            Hypers[key]  = this_Kernel.Hypers
            
        Hypers['rho']     = rho
        Hypers['probNze'] = probNze
        
        # store data to class
        self.Kernels = Kernels
        self.Hypers  = Hypers
        self.Train   = Trains
        self.FI_names = Trains.keys()
        
        return
    
    #def evaluate_kernel(self,P,Q,n_dp,n_dq,block):
        #return Gaussian.evaluate_kernel(self,P,Q,n_dp,n_dq,block)
    
    def K1(self):
        
        Kernels = self.Kernels
        rho = 10**self.Hypers['rho']
        
        Ker0 = Kernels[0]
        Ker1 = Kernels[1]
        
        X0 = Ker0.Train.X
        X1 = Ker1.Train.X

        K11 = Ker0.evaluate_kernel(X0,X0,0,0,'K1') + \
              Ker1.evaluate_kernel(X0,X0,0,0,'K1') * rho**2
        K13 = Ker1.evaluate_kernel(X1,X0,0,0,'K3') * rho
        K14 = Ker1.evaluate_kernel(X1,X1,0,0,'K1')
        
        K1s = [ [K11,K13.T] ,
                [K13,K14  ]  ]
        
        K1 = np.array( np.bmat(K1s) )
        
        return K1
    
    def K3(self,XI):
        
        Kernels = self.Kernels
        rho = 10**self.Hypers['rho']
        
        Ker0 = Kernels[0]
        Ker1 = Kernels[1]
        
        X0 = Ker0.Train.X
        X1 = Ker1.Train.X
        
        K31 = Ker0.evaluate_kernel(XI,X0,0,0,'K3') + \
              Ker1.evaluate_kernel(XI,X0,0,0,'K3') * rho**2
        K34 = Ker1.evaluate_kernel(XI,X1,0,0,'K3') * rho
        
        K3s = [ [K31,K34] ]
        
        K3 = np.array( np.bmat(K3s) )
        
        return K3
    
    def diag_K4(self,XI):
        
        Kernels = self.Kernels
        rho = 10**self.Hypers['rho']
        
        Ker0 = Kernels[0]
        Ker1 = Kernels[1]
        
        X0 = Ker0.Train.X
        X1 = Ker1.Train.X
        
        diag_K41 = Ker0.evaluate_kernel(XI,XI,0,0,'diag_K4') + \
                   Ker1.evaluate_kernel(XI,XI,0,0,'diag_K4') * rho**2 
        
        diag_K4 = diag_K41
        
        return diag_K4
    
    def Yt(self):
        
        Kernels = self.Kernels
        
        Yts = []
        
        for i,Kernel in enumerate(Kernels.values()):
            Yts.append( Kernel.Yt() )
        
        Yt = np.vstack(Yts)
        
        return Yt
        
    def pack_outputs(self,XI,YIt,CovIt):
        
        Kernels = self.Kernels
        
        # sizes
        nx,ndy = XI.shape
        ny = nx
        nyt = ny + ndy*ny
        
        # output data
        The_Data = IndexableBunch()
        
        # only one fidelity output to unpack
        Kernel = Kernels[0] # hack
        
        # no gradients yet
        fi = 0 # hack
        yi1 = nx*fi
        yi2 = nx*(fi+1)
        
        # extract
        this_YIt   = np.zeros([nyt,1])
        this_CovIt = np.zeros([nyt,1])
        this_YIt[0:nx,:]   = YIt[yi1:yi2,:]
        this_CovIt[0:nx,:] = CovIt[yi1:yi2,:]
        
        # unpack this fidelity
        this_data = Kernel.pack_outputs(XI,this_YIt,this_CovIt)
        The_Data = this_data # hack
        #The_Data[key] = this_data
        
        #: for each fidelity
        
        return The_Data

    
    # ------------------------------------------------------------------
    #   VF Gaussian Learning
    # ------------------------------------------------------------------
    
    def setup_learning(self,The_Problem):
        
        Hypers = self.Hypers
        Train  = self.Train
        
        # variables and bound constraints
        X  = Train[1].X  # use training with more data
        Y  = Train[1].Y
        DY = Train[1].DY
        
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
        
        # model ratio lims
        rho_lo = np.log10(0.5)
        rho_hi = np.log10(2.0)
        
        # noise ranges
        nze_lo = max([ probNze-bndNze , -10.  ])
        nze_hi = min([ probNze+bndNze , -0.01 ])
        
        # set variables and bound constraints
        The_Problem.addVar('sig_f'  , 'c', lower=sig_lo, upper=sig_hi, value=Hypers[0]['sig_f']  )
        The_Problem.addVar('len_s'  , 'c', lower=len_lo, upper=len_hi, value=Hypers[0]['len_s']  )
        The_Problem.addVar('sig_ny' , 'c', lower=nze_lo, upper=nze_hi, value=Hypers[0]['sig_ny'] )
        #The_Problem.addVar('sig_ndy', 'c', lower=nze_lo, upper=nze_hi, value=Hypers[0]['sig_ndy'])
        The_Problem.addVar('rho'    , 'c', lower=rho_lo, upper=rho_hi, value=Hypers['rho']       )
        
        #The_Problem.addVar('sig_f'  ,'c',lower=1.0/100.,upper=1.0*100.,value=Hypers[0]['sig_f']  )
        #The_Problem.addVar('len_s'  ,'c',lower=1.0/100.,upper=1.0*5.  ,value=Hypers[0]['len_s']  )
        #The_Problem.addVar('sig_ny' ,'c',lower=nze_lo     ,upper=nze_hi     ,value=Hypers[0]['sig_ny'] )
        ##The_Problem.addVar('sig_ndy','c',lower=nze_lo     ,upper=nze_hi     ,value=Hypers[0]['sig_ndy'])
        
        
        # nonlinear constraints
        The_Problem.addConGroup('Kernel Constraints',2,type='i')
        
        return The_Problem
    
    def likelihood(self,Hyp_vec):
        raise NotImplementedError
    
    def likelihood_cons(self,Hyp_vec):
        
        # some limits
        max_noise_ratio = -1.0
        min_noise_ratio = -8.0
        Kcond_limit     = -12.0       
        
        # unpack
        Hypers = self.Hypers       
        
        # unpack hypers
        self.unpack_vec(Hyp_vec)
        sig_f   = Hypers[0].sig_f
        len_s   = Hypers[0].len_s
        sig_ny  = Hypers[0].sig_ny
        sig_ndy = Hypers[0].sig_ndy   
        rho     = Hypers.rho
        
        # noise ratios
        noise_ratio_y   = sig_ny  - sig_f
        noise_ratio_dy  = sig_ndy - (sig_f-len_s)
        
        # kernel condition
        #Kcond = self.Kcond # expensive...
        
        # the constraints
        Con = np.array([ [ noise_ratio_y - max_noise_ratio  ] ,  # max ny
                         [ min_noise_ratio - noise_ratio_y  ] ,  # min ny
                         #[ noise_ratio_dy - max_noise_ratio ] ,  # max ndy
                         #[ min_noise_ratio - noise_ratio_dy ] ,  # min ndy
                         #[ sig_ny - sig_ndy                 ] ,  # relative noise
                         #[ sig_ndy- sig_ny - 1              ] ,  # max noise deviation
                         #[ Kcond_limit - np.log10(Kcond)    ] ,  # condition        
                      ])
        
        return Con
    
    #: def likelihood_cons()
    
    def unpack_vec(self,Hyp_vec):
        ''' unpack hyperparameter vector and store in kernel
        '''
        # check update
        bool_new = not np.all( Hyp_vec == self.pack_vec )
        
        # unpack vector
        sig_f   = Hyp_vec[0]
        len_s   = Hyp_vec[1]
        sig_ny  = Hyp_vec[2]
        sig_ndy = Hyp_vec[2] # hack
        rho     = Hyp_vec[3]
        
        # load across all Trains
        for key in self.FI_names:
            Hypers = self.Hypers[key]
            Hypers.sig_f   = sig_f
            Hypers.len_s   = len_s
            Hypers.sig_ny  = sig_ny
            Hypers.sig_ndy = sig_ndy
        self.Hypers.rho = rho
        
        return bool_new
        
    #: def unpack_vec()
    
    def pack_vec(self):
        ''' pack kernel's hyperparameters into a vector
        '''
        
        Hyp_vec = np.zeros([4])
        
        # pack vector
        Hyp_vec[0] = self.Hypers[0].sig_f  
        Hyp_vec[1] = self.Hypers[0].len_s  
        Hyp_vec[2] = self.Hypers[0].sig_ny 
        #Hyp_vec[2] = self.Hypers[0].sig_ndy # hack
        Hyp_vec[3] = self.Hypers.rho
                    
        return Hyp_vec
        
    #: def unpack_vec()