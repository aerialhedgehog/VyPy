


from warnings import warn
import numpy as np
from VyPy.tools import check_array


def simple(points_fs,basis_as):
    
    X = check_array(points_fs,'row')
    V = check_array(basis_as,'row')
    
    Y = np.dot(V,X.T).T
    
    points_as = Y
    return points_as