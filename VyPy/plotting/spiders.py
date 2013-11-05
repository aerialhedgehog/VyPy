
import numpy as np

import matplotlib

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

def spider_axis(FIG,X0,XB):

    X0 = check_array(X0,'row')
    XB = check_array(XB,'row')
    
    n_d = X0.shape[1]
    
    # thetas
    TP = np.linspace(0,np.pi,n_d+1)
    
    # axis
    AX = FIG.gca(projection='3d')
    
    # each dimension
    for i_d in range(n_d):
        
        # this axis coordinates
        Xi = [XB[i_d,0],XB[i_d,1]]
        
        # rotate into plot axis
        XP = ( Xi-X0[0,i_d] )* ( np.cos(TP[i_d]) )
        YP = ( Xi-X0[0,i_d] )* ( np.sin(TP[i_d]) )
        
        # plot axis line
        AX.plot( [ XP[0] , XP[1] ] ,
                 [ YP[0] , YP[1] ] ,
                 [ 0.    , 0.     ] , 'k-' )
        # plot axis ends
        AX.plot([XP[0]],[YP[0]],[0],'ko',mfc='none',mew=1.5)
        AX.plot([XP[-1]],[YP[-1]],[0],'ko')
        # plot axis labels
        AX.text(XP[0] ,YP[0] ,0.0,'X%i_L'%(i_d+1),fontsize=10)
        AX.text(XP[-1],YP[-1],0.0,'X%i_U'%(i_d+1),fontsize=10)
                
    #: for each dimension
    
    # disable x and y axis tick labels
    AX.xaxis.set_ticklabels([])
    AX.yaxis.set_ticklabels([])
    
    return AX



def spider_trace(AX,FF,X0,XB,NP,*plt_args,**plt_kwarg):
    ''' plots a multidimensional spider with function 
        FF, centered at X0, in bounds XB, on plot axis AX
    '''
    
    X0 = check_array(X0,'row')
    XB = check_array(XB,'row')
    
    n_d = X0.shape[1]
    n_x = NP
    
    # thetas
    TP = np.linspace(0,np.pi,n_d+1)
        
    # each dimension
    for i_d in range(n_d):
        
        # this axis coordinates
        Xi = np.linspace(XB[i_d,0],XB[i_d,1],n_x)
        XX = np.ones([n_x,n_d]) * X0
        XX[:,i_d] = Xi
        
        # rotate into plot axis
        XP = ( Xi-X0[0,i_d] )* ( np.cos(TP[i_d]) )
        YP = ( Xi-X0[0,i_d] )* ( np.sin(TP[i_d]) )

        # evaluate function
        ZP = FF(XX)[:,0]
        
        # plot function
        AX.plot(XP,YP,ZP,*plt_args,**plt_kwarg)
        
        if i_d == 0 and plt_kwarg.has_key('label'):
            del plt_kwarg['label']
        
    #: for each dimension
    
    # disable x and y axis tick labels
    AX.xaxis.set_ticklabels([])
    AX.yaxis.set_ticklabels([])
    
    # for legend
    P = AX.plot([0.],[0.],[np.nan],*plt_args,**plt_kwarg)
    
    return P