
from Variables  import Variables
from Objective  import Objective
from Equality   import Equality
from Inequality import Inequality

# ----------------------------------------------------------------------
#   Problem
# ----------------------------------------------------------------------
class Problem(object):
    
    def __init__(self):
        
        self.variables    = []
        self.objectives   = []
        self.equalities   = []
        self.inequalities = []
                
    def compile(self):
        
        # sort objectives        
        self.variables = Variables(self.variables)
        var = self.variables
        
        # sort objectives
        self.objectives = [ Objective(obj,var) for obj in self.objectives ]
        
        # sort constraints
        for con in self.constraints:
            _,(_,sgn,_),_ = con
            # equality constraint
            if sgn == '=':
                con = Equality(con,var)
                self.equalities.append(con)
            # inequality constraint
            elif sgn in '><':
                con = Inequality(con,var)
                self.inequalities.append(con)
            # uhoh
            else:
                raise Exception, 'unrecognized sign %s' % sgn
        #: for each constraint
        
        return
    
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
        
