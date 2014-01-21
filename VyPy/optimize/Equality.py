
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from Evaluator import Evaluator

from VyPy.data import IndexableDict
from VyPy.data.input_output import flatten_list


# ----------------------------------------------------------------------
#   Equality Function
# ----------------------------------------------------------------------

class Equality(Evaluator):
    
    Container = None
    
    def __init__( self, variables, evaluator, 
                  tag, sense='=', edge=0.0, 
                  scale=1.0 ):
        
        Evaluator.__init__(self)
        
        if not isinstance(evaluator, Evaluator):
            evaluator = Evaluator(function=evaluator)
        
        self.evaluator = evaluator
        self.tag       = tag
        self.edge      = edge
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
        val  = self.edge
        scl  = self.scale
        
        result = ( func(x)[tag] - val )
        
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
        return "<Equality '%s'>" % self.tag

# ----------------------------------------------------------------------
#   Equality Container
# ----------------------------------------------------------------------

class Equalities(IndexableDict):
    
    def __init__(self,variables):
        self.variables = variables
    
    def __set__(self,problem,arg_list):            
        self.clear()
        self.extend(arg_list)
    
    def append(self,*args):
        args = [self.variables] + flatten_list(args)
        equality = Equality(*args)
        tag = equality.tag
        self[tag] = equality
        
    def extend(self,arg_list):
        for args in arg_list:
            self.append(*args)
                    
    def tags(self):
        return self.keys()
    def senses(self):
        return ['='] * len(self)
    def edges(self):
        return [ con.edge for con in self.values() ]
    def scales(self):
        return [ con.scale for con in self.values() ]
    def evaluators(self):
        return [ con.evaluator for con in self.values() ]
    
    def set(edges=None,scales=None):
        if edges:
            for i,e in enumerate(edges):
                self[i].edge = e
        if scales:
            for i,s in enumerate(scales):
                self[i].scale = s            
    
# ----------------------------------------------------------------------
#   Container Linking
# ----------------------------------------------------------------------
Equality.Container = Equalities