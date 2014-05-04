
from VyPy.regression import gpr


def Gaussian(XB,X,Y,DY=None,learn=True,**hypers):
    """ class factory for a Gaussian Model
    """
    
    Train = gpr.Training(XB,X,Y,DY)
    
    Scaling = gpr.scaling.Linear(Train)
    Train = Scaling.set_scaling(Train)
    
    Kernel = gpr.kernel.Gaussian(Train,**hypers)
    Infer  = gpr.inference.Gaussian(Kernel)
    Learn  = gpr.learning.Likelihood(Infer)
    Model  = gpr.modeling.Regression(Learn,Scaling)
    
    if learn:
        Model.learn()
    
    return Model