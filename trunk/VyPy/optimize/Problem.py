
from Variable   import Variable
from Objective  import Objective
from Constraint import Constraint
from Equality   import Equality
from Inequality import Inequality

from VyPy.data import IndexableDict, Object

# ----------------------------------------------------------------------
#   Problem
# ----------------------------------------------------------------------
class Problem(Object):
    
    def __init__(self):
        
        self.variables    = Variable.Container()
        self.objectives   = Objective.Container(self.variables)
        self.constraints  = Constraint.Container(self.variables)
        self.equalities   = self.constraints.equalities
        self.inequalities = self.constraints.inequalities
      
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
        


