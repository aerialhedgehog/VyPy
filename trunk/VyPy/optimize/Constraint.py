
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from Evaluator  import Evaluator
from Equality   import Equality
from Inequality import Inequality

from VyPy.data import IndexableDict, Descriptor
from VyPy.data.input_output import flatten_list


class_map = {
    '=' : Equality ,
    '>' : Inequality,
    '<' : Inequality,
}
   
        
# ----------------------------------------------------------------------
#   Constraint Container
# ----------------------------------------------------------------------

class Constraints(object):
    
    def __init__(self,variables):
        self._variables    = variables
        self._equalities   = Equality.Container(self.variables)
        self._inequalities = Inequality.Container(self.variables)
        
        self._container_map = {
            '=' : self.equalities ,
            '>' : self.inequalities,
            '<' : self.inequalities,
        }            
    
    # setup descriptors
    variables    = Descriptor('_variables')
    equalities   = Descriptor('_equalities')
    inequalities = Descriptor('_inequalities')    
    
    def __set__(self,problem,arg_list):            
        self.clear()
        self.extend(arg_list)
        
    def clear(self):
        self.equalities.clear()
        self.inequalities.clear()
    
    def append(self, evaluator, tag='c', sense='=', edge=0.0, scale=1.0 ):
        
        if isinstance(evaluator,(Equality,Inequality)):
            constraint = evaluator
            constraint.variables = self.variables
        else:
            constraint = class_map[sense](evaluator,tag,sense,edge,scale,self.variables)
        
        constraint.__check__()
        
        sense = constraint.sense
        if sense not in class_map.keys():
            raise KeyError , 'invalid constraint sense "%s"' % sense        
        
        tag = constraint.tag
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
#   Constraint Function
# ----------------------------------------------------------------------

class Constraint(Evaluator):
    
    Container = Constraints
    
    def __init__( self ):
        pass
        
    def function(self,x):
        raise NotImplementedError
    
    def gradient(self,x):
        raise NotImplementedError
    
    def hessian(self,x):
        raise NotImplementedError