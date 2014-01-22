
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from Evaluator import Evaluator

from VyPy.data import IndexableDict
from VyPy.data.input_output import flatten_list


# ----------------------------------------------------------------------
#   Inequality Function
# ----------------------------------------------------------------------

class Inequality(Evaluator):
    
    Container = None
    
    def __init__( self, variables, evaluator, 
                  tag, sense='<', edge=0.0, 
                  scale=1.0 ):
        
        Evaluator.__init__(self)
        
        if not isinstance(evaluator, Evaluator):
            evaluator = Evaluator(function=evaluator)
            
        if sense not in '><':
            raise TypeError , 'sense must be > or <'
                
        self.evaluator = evaluator
        self.tag       = tag
        self.sign      = sense
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
        sgn  = self.sign
        val  = self.edge
        scl  = self.scale
        
        if sgn == '>':
            result = ( val - func(x)[tag] )
        elif sgn == '<':
            result = ( func(x)[tag] - val )
        else:
            raise Exception, 'unrecognized sign %s' % sgn        
        
        result = result * scl
                
        return result
    
    def gradient(self,x):
        
        x = self.variables.scaled.pack(x)
        
        func = self.evaluator.gradient
        tag  = self.tag
        sgn  = self.sign
        scl  = self.scale
        
        if sgn == '>':
            result = -1* func(x)[tag]
        elif sgn == '<':
            result = +1* func(x)[tag]
        else:
            raise Exception, 'unrecognized sign %s' % sgn        
        
        result = result * scl
        
        return result 
    
    def hessian(self,x):
        raise NotImplementedError

    def __repr__(self):
        return "<Inequality '%s'>" % self.tag

# ----------------------------------------------------------------------
#   Inequality Container
# ----------------------------------------------------------------------

class Inequalities(IndexableDict):
    
    def __init__(self,variables):
        self.variables = variables
    
    def __set__(self,problem,arg_list):            
        self.clear()
        self.extend(arg_list)
    
    def append(self,*args):
        args = [self.variables] + flatten_list(args)
        inequality = Inquality(*args)
        tag = inequality.tag
        self[tag] = inequality
        
    def extend(self,arg_list):
        for args in arg_list:
            self.append(*args)
                        
    def tags(self):
        return self.keys()
    def senses(self):
        return [ con.sense for con in self.values() ]
    def edges(self):
        return [ con.edge for con in self.values() ]
    def scales(self):
        return [ con.scale for con in self.values() ]
    def evaluators(self):
        return [ con.evaluator for con in self.values() ]
    
    def set(senses=None,edges=None,scales=None):
        if senses:
            for i,s in enumerate(senses):
                self[i].sense = s            
        if edges:
            for i,e in enumerate(edges):
                self[i].edge = e
        if scales:
            for i,s in enumerate(scales):
                self[i].scale = s           
    
    
# ----------------------------------------------------------------------
#   Container Linking
# ----------------------------------------------------------------------
Inequality.Container = Inequalities