
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from Evaluator  import Evaluator
from Constraint import Constraint

from VyPy.data import IndexableDict
from VyPy.data.input_output import flatten_list
from VyPy.tools.arrays import atleast_2d


# ----------------------------------------------------------------------
#   Equality Function
# ----------------------------------------------------------------------

class Equality(Constraint):
    
    Container = None
    
    def __init__( self, evaluator=None, 
                  tag='ceq', sense='=', edge=0.0, 
                  scale=1.0,
                  variables=None ):
        
        Constraint.__init__( self,evaluator,
                             tag,sense,edge,
                             scale,variables )
         
    def function(self,x):
        
        x = self.variables.scaled.unpack_array(x)
        
        func = self.evaluator.function
        tag  = self.tag
        val  = self.edge
        scl  = self.scale
        
        result = func(x)[tag]/scl - val/scl
        
        result = atleast_2d(result,'col')
        
        return result
    
    def gradient(self,x):
        
        x = self.variables.scaled.unpack_array(x)
        
        func = self.evaluator.gradient
        tag  = self.tag
        scl  = self.scale
        
        result = func(x)[tag]
        
        result = result / scl ## !!! PROBLEM WHEN SCL is NOT CENTERED
        
        result = atleast_2d(result,'row')
        
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
    
    def append(self, evaluator, 
               tag=None, sense='=', edge=0.0, 
               scale=1.0 ):
        
        if tag is None and isinstance(evaluator,Equality):
            equality = evaluator
            equality.variables = self.variables
        else:
            args = flatten_list(args) + [self.variables]
            equality = Equality(evaluator,tag,sense,edge,scale,self.variables)
        
        equality.__check__()
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
    
    def edges_array(self):
        return np.vstack([ atleast_2d(x,'col') for x in self.edges() ])    
    
    def set(edges=None,scales=None):
        if edges:
            for i,e in enumerate(edges):
                self[i].edge = e
        if scales:
            for i,s in enumerate(scales):
                self[i].scale = s      
                
# ----------------------------------------------------------------------
#   Equality Container
# ----------------------------------------------------------------------
Equality.Container = Equalities