
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from Evaluator  import Evaluator
from Equality   import Equality
from Inequality import Inequality

from VyPy.data import IndexableDict, Object
from VyPy.data.input_output import flatten_list


class_map = {
    '=' : Equality ,
    '>' : Inequality,
    '<' : Inequality,
}
   

# ----------------------------------------------------------------------
#   Constraint Function
# ----------------------------------------------------------------------

class Constraint(Evaluator):
    
    Container = None
    
    def __init__( self ):
        raise NotImplementedError
        
    def function(self,x):
        raise NotImplementedError
    
    def gradient(self,x):
        raise NotImplementedError
    
    def hessian(self,x):
        raise NotImplementedError


# ----------------------------------------------------------------------
#   Constraint Container
# ----------------------------------------------------------------------

class Constraints(Object):
    
    def __init__(self,variables):
        self.variables    = variables
        self.equalities   = Equality.Container(self.variables)
        self.inequalities = Inequality.Container(self.variables)
        
        self._container_map = {
            '=' : self.equalities ,
            '>' : self.inequalities,
            '<' : self.inequalities,
        }            
    
    
    def __set__(self,problem,arg_list):            
        self.clear()
        self.extend(arg_list)
        
    def clear(self):
        self.equalities.clear()
        self.inequalities.clear()
    
    def append(self, evaluator, tag, sense, edge=0.0, scale=1.0 ):
        
        if sense not in class_map.keys():
            raise KeyError , 'invalid constraint sense "%s"' % sense
        
        constraint = class_map[sense](self.variables,evaluator,tag,sense,edge,scale)
        self._container_map[sense][tag] = constraint
        
    def extend(self,arg_list):
        for args in arg_list:
            args = flatten_list(args)
            self.append(*args)
            
    def tags(self):
        return self.equalities.tags() + self.inequalities.tags()
    def senses(self):
        return self.equalities.senses() + self.inequalities.senses()
    def edges(self):
        return self.equalities.edges() + self.inequalities.edges()
    def scales(self):
        return self.equalities.scales() + self.inequalities.scales()
    def evaluators(self):
        return self.equalities.evaluators() + self.inequalities.evaluators()
    
    # no set(), modify constraints by type
    
    def __len__(self):
        return len(self.equalities) + len(self.inequalities)
    def items(self):
        return self.equalities.items() + self.inequalities.items()
    def __repr__(self):
        return repr(self.equalities) + '\n' + repr(self.inequalities)
    def __str__(self):
            return str(self.equalities) + '\n' + str(self.inequalities)
    
# ----------------------------------------------------------------------
#   Container Linking
# ----------------------------------------------------------------------
Constraint.Container = Constraints