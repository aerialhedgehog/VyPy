
import numpy as np

def estimate_error(G_X, X,F, Scaling=None):
    ''' root mean square error between G_X(X) and F
        provide Scaling object to get relative errors
    '''
    
    G = G_X(X)
    
    if Scaling:
        F = Scaling.Y.set_scaling(F)
        G = Scaling.Y.set_scaling(G)
        
    # rms errors
    E_rms = np.sqrt( np.mean( (F-G)**2 ) )
    
    return E_rms