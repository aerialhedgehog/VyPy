
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------        

from VyPy.data import ibunch, obunch

# ----------------------------------------------------------------------
#   Driver
# ----------------------------------------------------------------------        

class Driver(object):
    
    def __init__(self):
        
        self.verbose = True
        self.other_options = obunch()
    
    def run(self,problem):
        raise NotImplementedError
    
    def pack_outputs(self,vars_min):
        
        # unpack
        objectives = self.problem.objectives
        equalities = self.problem.constraints.equalities
        inequalities = self.problem.constraints.inequalities
        
        # start the data structure
        outputs = ibunch()
        outputs.variables    = None
        outputs.objectives   = ibunch()
        outputs.equalities   = ibunch()
        outputs.inequalities = ibunch()
        outputs.success      = False
        outputs.messages     = ibunch()
        
        # varaiables
        outputs.variables = vars_min
        
        if vars_min.keys() == ['vector']:
            vars_min = vars_min['vector']           
        
        # objectives
        for tag in objectives.tags():
            result = objectives[tag].evaluator.function(vars_min)
            if isinstance(result,dict): result = result[tag]
            outputs.objectives[tag] = result
        
        # equalities
        for tag in equalities.tags():
            result = equalities[tag].evaluator.function(vars_min)
            if isinstance(result,dict): result = result[tag]
            outputs.equalities[tag] = result
            
        # inequalities
        for tag in inequalities.tags():
            result = inequalities[tag].evaluator.function(vars_min)
            if isinstance(result,dict): result = result[tag]
            outputs.inequalities[tag] = result
            
        return outputs