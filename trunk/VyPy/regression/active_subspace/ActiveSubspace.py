
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import sys, copy, weakref,gc

import numpy as np
import cvxopt

from Modeling import Modeling
from VyPy.data import obunch
from VyPy.tools import vector_distance, atleast_2d, atleast_2d_row

from VyPy.regression.gpr.training import Training
from VyPy.regression.least_squares import Quadratic, Linear

import learn as as_learn
import project as as_project

# ----------------------------------------------------------------------
#   Active Subspace Model
# ----------------------------------------------------------------------

class ActiveSubspace(Modeling):
    
    def __init__(self,Train):
        
        self.Train = Train
        
        self.nAS  = 2
        self.nASp = 2
        
        self.regularization = 'semi_quadratic'
        
        #self.W    = None
        #self.d    = None
        #self.W1   = None
        #self.W2   = None
        #self.YB   = None
        #self.A    = None
        #self.b    = None
        #self.Mz   = None
    
    def learn(self):
        
        # unpack
        Train = self.Train
        XB = Train.XB
        X  = Train.X
        F  = Train.Y
        DF = Train.DY
        
        # number of actice subspace dimensions
        nAS  = self.nAS
        nASp = self.nASp
        
        # learn the active subpace by gradients
        print 'Learn Active Subspaces ...'
        W,d = as_learn.gradient(DF)
        
        print '  Number of dimensions = %i' % nAS
        
        # down select the number of dimensions
        U = W[:,0:nAS]      # active subspace
        V = W[:,nAS:]       # inactive subspace
        R = W[:,0:nAS+nASp] # semiquadratic subspace
        
        # forward map training data to active space
        Y  = as_project.simple(X,U)
        YB = np.vstack([ np.min(Y,axis=0) , 
                         np.max(Y,axis=0) ]).T * 1.3
        
        if self.regularization == 'quadratic':
            
            # quadratic model
            train = Training(XB,X,F,DF)
            model = Quadratic(train)
            model.learn()
            A = model.A
            b = model.b
        
        elif self.regularization == 'semi_quadratic':
            
            # semiquadratic space
            K   = as_project.simple(X,R)
            FK  = F
            DFK = as_project.simple(DF,R)
            KB  = np.vstack([ np.min(K,axis=0) , 
                              np.max(K,axis=0) ]).T * 1.3            

            # semiquadratic model
            train = Training(KB,K,FK,None)
            model = Quadratic(train)
            model.learn()    
            A = model.A
            b = model.b
            
            # rotate back to full space
            b = np.dot(R,b)
            A = np.dot( R , np.dot(A , R.T))        
            
        elif self.regularization == 'linear':
            train = Training(XB,X,F,None)
            model = Linear(train)
            model.learn()    
            A = None
            b = model.b        
            
        elif self.regularization == 'inactive_norm':
            A = None
            b = None
            model = None
            
        else:
            raise Exception , 'No regularization type %s' % self.regularization
        
        # store
        self.W   = W
        self.d   = d
        self.W1  = U
        self.W2  = V
        self.YB  = YB
        self.A   = A
        self.b   = b
        self.Mz  = model
        
        return
    
    def forward_map(self,X):
        
        U = self.W1
        
        Y  = as_project.simple(X,U)
        
        return Y
        
    
    def inverse_map(self,Y):
        
        # active space
        Y = atleast_2d_row(Y)
        
        # full space bounds
        XB = self.Train.XB
        
        # active and inactive space bases
        W1 = self.W1
        W2 = self.W2
        n1 = W1.shape[1]
        n2 = W2.shape[1]
        
        # submodel
        A  = self.A
        b  = self.b
        
        # real space bounds
        G = np.vstack([-W2,W2])
        h = np.vstack([-XB[:,0,None] + np.dot(W1,Y.T) ,  # lower bounds
                        XB[:,1,None] - np.dot(W1,Y.T) ]) # upper bounds
        G = cvxopt.matrix( G.T.tolist() )
        h = cvxopt.matrix( h.T.tolist() )        
        
        if self.regularization == 'inactive_norm':
            
            # norm of Z
            P = cvxopt.spdiag([1.0]*n2)
            q = cvxopt.matrix([[0.0]*n2])       
            
            # solve regularization
            solution = cvxopt.solvers.qp(P,q,G,h)
            Z = list(solution['x'])
            Z = np.array([Z])                       
        
        elif self.regularization in ['quadratic','semi_quadratic']:
            
            # global quadratic model
            P = np.dot( W2.T , np.dot(A , W2) )
            q = np.dot( b.T , W2 ).T + 2.*np.dot( np.dot(W1,Y.T).T , np.dot( A , W2 ) ).T
            P = cvxopt.matrix(P)
            q = cvxopt.matrix(q)
            
            # solve regularization
            try:
                solution = cvxopt.solvers.qp(P,q,G,h)
            except ValueError:
                x0 = q*0.
                s0 = h*0.+1e-6
                solution = cvxopt.solvers.qp(P,q,G,h,initvals={'x':x0,'s':s0})
                
            Z = list(solution['x'])
            Z = np.array([Z])            
        
        elif self.regularization == 'linear':    

            # global linear model
            s = cvxopt.matrix( np.dot(b.T , W2).T )

            # solve regularization
            solution = cvxopt.solvers.lp(s,G,h) 
            Z = list(solution['x'])
            Z = np.array([Z])
            
        else:
            raise Exception , 'No regularization type %s' % self.regularization
        
        ## check
        #if solution['status'] != 'optimal':
            #print 'warning could not solve qp'
        
        # exercise the map
        X = np.dot(W1,Y.T) + np.dot(W2,Z.T)
        X = X[:,0]
        
        # done!
        return X
