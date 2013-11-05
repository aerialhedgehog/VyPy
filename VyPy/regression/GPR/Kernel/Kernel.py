
from VyPy.tools import IndexableDict

    
class Kernel(object):
    ''' Gaussian Covariance Matrix with Derivatives 
        Hyp = sig_f,len_s,sig_ny,sig_ndy, all scalars
        probNze nominal noize for learning
    '''
    
    def __init__(self,Train,**hypers):
        
        self.Train = Train

        # initialize hyperparameters
        self.Hypers = IndexableDict()
        self.default_hypers()
        
        # set wild hypers
        self.Hypers.update(hypers)
        
        return 
    
    #: def __init__()
    
    def default_hypers(self):
        """ Kernel.default_hypers()
            
            initialize hyperparameters, order matters
            
            Example:
            
            def default_hypers(self)
                self.Hypers['sig_f'] = np.log10(1.0)
                self.Hypers['len_s'] = np.log10(0.5)
                
            Inputs:
                None
            
            Outputs:
                None
        
        """
        
        return
    
    #: def default_hypers()
    
    
    def K1(self):
        """ K1 = Kernel.K1()
            
            calculate first covariance block - training to training
            must be square symmetric
            
            K = [[ K1 , K2 ],
                 [ K3 , K4 ]]
                        
            Inputs:
                None
                
            Outputs
                K1 - first covariance matrix block
                     n-inputdata x n-inputdata numpy array
                
        """
        
        raise NotImplementedError
        
        return K1
    
    #: def K1()
    
    def K3(self,XI):
        """ K3 = Kernel.K3(XI)
            
            calculate third covariance block - prediction to training
            not necessarily square
            do not add noise here
            
            K = [[ K1 , K2 ],
                 [ K3 , K4 ]]
            
            Inputs:
                XI    - Design matrix of prediction locations, 
                        n-location x m-dimension
                
            Outputs
                K3 - third covariance matrix block
                     n-outputdata x m-inputdata numpy array
                
        """
        
        raise NotImplementedError
        
        return K3
    
    #: def K3()
    
    def diag_K4(self,XI):
        """ diag_K4 = Kernel.diag_K4(XI):
        
            diagonal of fourth covariance block - prediction to prediction
            
            K = [[ K1 , K2 ],
                 [ K3 , K4 ]]
            
            Inputs:
                XI    - Design matrix of prediction locations, 
                        n-location x m-dimension numpy array
                
            Outputs
                diag_K4 - diagonal of fourth covariance matrix block
                          n-outputdata x 1 numpy array
            
        """

        raise NotImplementedError
        
        return diag_K4
    
    def Yt(self):
        """ Yt = Kernel.Yt():
        
            total input data vector
            pack all inputs into a column vector
            
            Inputs:
                None
                
            Outputs
                Yt - vector of all training data
                     n-inputdata x 1 numpy array
            
        """
        
        raise NotImplementedError
        
        return Yt
    
    #: def Yt()
    
    
    def pack_outputs(self,XI,YIt,CovIt):
        """ data = pack_outputs(self,XI,YIt,CovIt)
            
            pack total prediction vector YIt and total covariance vector CovIt
            optional, unless Modeling.reinterpolate() is needed
            
            Inputs:
                Train - training data of class Training()
                XI    - prediction locations 2D design matrix 
                YIt   - total prediction data column vector
                CovIt - total covariance column vector
            
            Outputs:
                data - a dict() type with packed data
            
            To work with Modeling.reinterpolate(), this function must
            return a dictionary with at least the same keys as self.Train
            with the format <key>I (suffix of 'I', for 'interpolated')
            
            Example:
            
            if Train contains keys:
                Train.X
                Train.Y
            
            then Model.pack_outputs() should return at least:
                data.XI
                data.YI
            
        """
        
        raise NotImplementedError
            
        return data
            
    #: def pack_outputs()         
        
#: class Kernel()