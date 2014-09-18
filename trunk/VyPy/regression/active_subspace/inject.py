
from warnings import warn
import numpy as np
import scipy as sp
from VyPy.tools import atleast_2d
from VyPy.exceptions import Infeasible
import VyPy.optimize as opt
from VyPy.regression.active_subspace import project

# linear programming package
import cvxopt.solvers
cvxopt.solvers.options['show_progress'] = False

def simple(basis_as,points_as,bounds_fs):
    
    # setup
    V  = atleast_2d(basis_as,'row')
    Y  = atleast_2d(points_as,'row')
    XB = atleast_2d(bounds_fs,'row')
    
    X = np.dot(Y,V)
    
    # check full space bounds
    for i,x in enumerate(X):
        if not ( np.all(x>=XB[:,0]) and 
                 np.all(x<=XB[:,1]) ):
            X[i] = X[i] * np.nan
            
    points_fs = X
    return points_fs


def bounded(points_as,basis_as,bounds_fs):
    
    # setup
    V  = atleast_2d(basis_as,'row')
    Y  = atleast_2d(points_as,'row')
    
    XB = atleast_2d(bounds_fs,'row')
    dim_X = len(V[0])
    X  = []
    
    # dummy objective weights
    f = cvxopt.matrix( [ 0.0 ] * dim_X )
    
    # equality constraint weights
    Aeq = cvxopt.matrix( V )
    
    # full space bounds
    A = cvxopt.matrix([ cvxopt.spdiag([-1.0]*dim_X) ,
                        cvxopt.spdiag([ 1.0]*dim_X) ])
    b = cvxopt.matrix([ list(-XB[:,0]) + 
                        list( XB[:,1]) ])
        
    for y in Y:
        
        # try simple injection
        x = simple(V,y,XB)[0]
        
        # nans if not in fullspace bounds
        if not np.all(np.isnan(x)):
            X += [x]
            continue
            
        # constraint edges
        beq = cvxopt.matrix(y)
        
        # solve
        solution = cvxopt.solvers.lp(f, A,b, Aeq,beq)
        
        # check
        if solution['status'] == 'optimal':
            x = list(solution['x'])
        else:
            x = [np.nan] * dim_X
            
        #print 'guess in bounds  -' , np.all(xg>=bounds_fs[:,0]) and np.all(xg<=bounds_fs[:,1])
        #print 'solve in bounds  -' , np.all(x>=bounds_fs[:,0]) and np.all(x<=bounds_fs[:,1])
        #print 'solve-guess      -' , np.linalg.norm(x - xg)
        #print 'projection error -' , np.linalg.norm( np.dot(x,V.T) - y )
        #print 'iterations       -' , solution['iterations']
        #print ''
        
        # store
        X += [x]
        
    points_fs = np.array(X)
    return points_fs

def constrained(points_as,basis_as,bounds_fs,basis_con):
    # setup
    V  = atleast_2d(basis_as,'row')
    Y  = atleast_2d(points_as,'row')
    
    XB = atleast_2d(bounds_fs,'row')
    dim_X = len(V[0])
    X  = []
    
    #y_con = 0.0016
    y_con = 0.00
    #y_con = -0.0016
    
    V_con = atleast_2d(basis_con,'row')
    
    # dummy objective weights
    f = cvxopt.matrix( [ 0.0 ] * dim_X )
    
    # equality constraint weights
    Aeq = cvxopt.matrix( V )
    
    # full space bounds
    A = cvxopt.matrix([ cvxopt.spdiag([-1.0]*dim_X) ,
                        cvxopt.spdiag([ 1.0]*dim_X) ,
                        cvxopt.matrix(V_con) ])
    b = cvxopt.matrix([ list(-XB[:,0]) + 
                        list( XB[:,1]) +
                        [y_con]        ])
        
    for y in Y:
        
        # try simple injection
        x = simple(V,y,XB)[0]
        
        # nans if not in fullspace bounds
        if not np.all(np.isnan(x)) and np.dot(V_con,x) < y_con:
            X += [x]
            continue
            
        # constraint edges
        beq = cvxopt.matrix(y)
        
        # solve
        solution = cvxopt.solvers.lp(f, A,b, Aeq,beq)
        
        # check
        if solution['status'] == 'optimal':
            x = list(solution['x'])
        else:
            x = [np.nan] * dim_X
            
        #print 'guess in bounds  -' , np.all(xg>=bounds_fs[:,0]) and np.all(xg<=bounds_fs[:,1])
        #print 'solve in bounds  -' , np.all(x>=bounds_fs[:,0]) and np.all(x<=bounds_fs[:,1])
        #print 'solve-guess      -' , np.linalg.norm(x - xg)
        #print 'projection error -' , np.linalg.norm( np.dot(x,V.T) - y )
        #print 'iterations       -' , solution['iterations']
        #print ''
        
        # store
        X += [x]
        
    points_fs = np.array(X)
    return points_fs

    
def dummy_objective(inputs):
    return {'dummy_result':0.0}

class AS_Constraint(opt.Evaluator):
    def __init__(self,basis_as,point_as=None,model_as=None):
        opt.Evaluator.__init__(self)
        
        self.basis_as = basis_as
        self.point_as = point_as
        self.model_as = model_as
        
    def function(self,inputs):
        X = inputs.values()
        V = self.basis_as
        YT = self.point_as
        
        Y = project.simple(X,V)[0]
        
        outputs = {
            'point_as':Y,
        }
        
        if not self.point_as is None:
            C = np.sum( (Y-YT)**2 )
            outputs['C'] = C
        
        if not self.model_as is None:
            FI = self.model_as.predict(Y).YI[0][0]
            outputs['value_as'] = FI
        
        return outputs
        
    
    

#def inject_mean(basis_as, points_as, points_fs):
    
    ## setup
    #basis_as = atleast_2d(basis_as,'row')
    #points_as = atleast_2d(points_as,'col')
    
    #dim_fs = len(basis_as[0])
    #dim_rs = len(points_as[0])
    
    #basis_rs = basis_as[0:dim_rs,:]
    
    #basis_hs = basis_as[dim_rs:,:]
    #mean_hs = np.zeros([1,dim_fs])

    #tol = 0.01
    
    #ikeep = np.logical_and(Xv1<(xp1+tol) , Xv1>(xp1-tol))
    #X = X[ikeep,:]
    #Xv1 = Xv1[ikeep]
    #Y = Y[ikeep,:]
    
    #for i in range(50):
        
        #xvi = np.dot(X,v_all[i,:])
        
        #plt.figure(2)
        #plt.clf()
        ##plt.plot(Xv1,xvi,'bx')
        ##plt.scatter(Xv1,xvi,c=Y,s=100)
        #plt.scatter(xvi,Y,c=Xv1,s=100)
        #plt.xlabel('Reduced Coordinate %i'%i)
        ##plt.ylabel('Reduced Coordinate %i' % i)
        #plt.ylabel(objective)
        #plt.xlim([-0.1,0.1])
        #plt.ylim([-0.1,0.1])
        #plt.title(plot_title)
        #plt.show(block=False)
        #wait = 0

    
    #for xl in points_as:
        
        #xh = inject_simple(basis_rs,xl)
        
        
        
    