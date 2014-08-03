


from warnings import warn
import numpy as np
from VyPy.tools import atleast_2d


def simple(points_fs,basis_as):
    
    X = atleast_2d(points_fs,'row')
    V = atleast_2d(basis_as,'row')
    
    Y = np.dot(V.T,X.T).T
    
    points_as = Y
    return points_as