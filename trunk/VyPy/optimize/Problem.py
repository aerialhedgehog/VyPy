
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from Variable   import Variable
from Objective  import Objective
from Constraint import Constraint
from Equality   import Equality
from Inequality import Inequality

from VyPy.data import IndexableDict, Object, Descriptor


# ----------------------------------------------------------------------
#   Problem
# ----------------------------------------------------------------------

class Problem(object):
    
    def __init__(self):
        
        self._variables    = Variable.Container()
        self._objectives   = Objective.Container(self.variables)
        self._constraints  = Constraint.Container(self.variables)
        self._equalities   = self.constraints.equalities
        self._inequalities = self.constraints.inequalities
      
    def has_gradients(self):
        
        # objectives
        grads = [ not evalr.gradient is None for evalr in self.objectives ]
        obj_grads = any(grads) and all(grads)
        
        # inequalities
        grads = [ not evalr.gradient is None for evalr in self.inequalities ]
        ineq_grads = any(grads) and all(grads)
            
        # equalities
        grads = [ not evalr.gradient is None for evalr in self.equalities ]
        eq_grads = any(grads) and all(grads)            
           
        return obj_grads, ineq_grads, eq_grads
  
    # setup descriptors
    variables    = Descriptor('_variables')
    objectives   = Descriptor('_objectives')
    constraints  = Descriptor('_constraints')
    equalities   = Descriptor('_equalities')
    inequalities = Descriptor('_inequalities')  
  
