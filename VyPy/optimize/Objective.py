
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from Evaluator import Evaluator

from VyPy.data import IndexableDict

# ----------------------------------------------------------------------
#   Objective Function
# ----------------------------------------------------------------------
class Objective(Evaluator):
    
    Container = None # linked at end of module
    
    # TODO: tag=None
    def __init__(self,variables,evaluator,tag,scale=1.0):
        
        Evaluator.__init__(self)
        
        if not isinstance(evaluator, Evaluator):
            evaluator = Evaluator(function=evaluator)
        
        self.evaluator = evaluator
        self.tag       = tag
        self.scale     = scale
        self.variables = variables
        
        if evaluator.gradient is None:
            self.gradient = None
        if evaluator.hessian is None:
            self.hessian = None
        
    def function(self,x):
        
        x = self.variables.scaled.pack(x)
        
        func = self.evaluator.function
        tag  = self.tag
        scl  = self.scale
        
        result = func(x)[tag]
        
        result = result * scl
        
        return result
    
    def gradient(self,x):
        
        x = self.variables.scaled.pack(x)
        
        func = self.evaluator.gradient
        tag  = self.tag
        scl  = self.scale
        
        result = func(x)[tag]
        
        result = result * scl
        
        return result    
    
    def hessian(self,x):
        raise NotImplementedError
    
    def __repr__(self):
        return "<Objective '%s'>" % self.tag


# ----------------------------------------------------------------------
#   Objectives Container
# ----------------------------------------------------------------------

class Objectives(IndexableDict):
    
    def __init__(self,variables):
        self.variables = variables
    
    def __set__(self,problem,arg_list):            
        self.clear()
        self.extend(arg_list)
    
    def append(self,evaluator,tag,scale=1.0):
        objective = Objective(self.variables,evaluator,tag,scale)
        tag = objective.tag
        self[tag] = objective
        
    def extend(self,arg_list):
        for args in arg_list:
            self.append(*args)
        
    def tags(self):
        return self.keys()
    def scales(self):
        return [ obj.scale for obj in self.values() ]
    def evaluators(self):
        return [ obj.evaluator for obj in self.values() ]
    
    def set(self,scales=None):
        if scales:
            for i,s in enumerate(scales):
                self[i].scale = s
    
    
# ----------------------------------------------------------------------
#   Container Linking
# ----------------------------------------------------------------------
Objective.Container = Objectives