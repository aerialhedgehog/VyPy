
from Evaluator import Evaluator

# ----------------------------------------------------------------------
#   Inequality Function
# ----------------------------------------------------------------------
class Inequality(object):
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
        
        if sgn == '>':
            result = ( val - func(x)[key] )
        elif sgn == '<':
            result = ( func(x)[key] - val )
        else:
            raise Exception, 'unrecognized sign %s' % sgn        
        
        result = result * scl
                
        return result
    
    def gradient(self,x):
        
        x = self.variables.scaled.unpack(x)
        
        func = self.evaluator.gradient
        key  = self.output
        sgn  = self.sign
        scl  = self.scale
        
        if sgn == '>':
            result = -1* func(x)[key]
        elif sgn == '<':
            result = +1* func(x)[key]
        else:
            raise Exception, 'unrecognized sign %s' % sgn        
        
        result = result * scl
        
        return result 
    
    def hessian(self,x):
        raise NotImplementedError

