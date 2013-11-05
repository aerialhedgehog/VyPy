
from Evaluator import Evaluator

# ----------------------------------------------------------------------
#   Equality Function
# ----------------------------------------------------------------------
class Equality(Evaluator):
    def __init__(self,constraint,variables):
        
        Evaluator.__init__(self)
        
        evalr,(key,sgn,val),scl = constraint

        if not isinstance(evalr, Evaluator):
            evalr = Evaluator(function=evalr)
        
        self.evaluator = evalr
        self.output    = key
        self.sign      = sgn
        self.value     = val
        self.scale     = scl
        self.variables = variables
        
        if evalr.gradient is None:
            self.gradient = None
        if evalr.hessian is None:
            self.hessian = None
         
        
    def function(self,x):
        
        x = self.variables.scaled.unpack(x)
        
        func = self.evaluator.function
        key  = self.output
        sgn  = self.sign
        val  = self.value
        scl  = self.scale
        
        result = ( func(x)[key] - val )
        
        result = result * scl
        
        return result
    
    def gradient(self,x):
        
        x = self.variables.scaled.unpack(x)
        
        func = self.evaluator.gradient
        key  = self.output
        sgn  = self.sign
        scl  = self.scale
        
        result = func(x)[key]
        
        result = result * scl
        
        return result    

    def hessian(self,x):
        raise NotImplementedError

