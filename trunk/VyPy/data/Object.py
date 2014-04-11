
class Object(object):
    
    # implement descriptor protocol for instances
    def __getattribute__(self,k):
        try:
            return super(Object,self).__getattribute__(k).__get__(self,type(self))
        except AttributeError:
            return super(Object,self).__getattribute__(k)
        except AttributeError:
            raise AttributeError(k)        
    
    def __setattr__(self,k,v):
        try:
            super(Object,self).__getattribute__(k).__set__(self,v)
        except AttributeError:
            super(Object,self).__setattr__(k,v)
    
    def __delattr__(self,k):
        try:
            super(Object,self).__getattribute__(k).__del__(self)
        except AttributeError:
            super(Object,self).__delattr__(k)
        except AttributeError:
            raise AttributeError(k)                    
            
if __name__ == '__main__':
    
    class TestDescriptor(object):
        def __init__(self,x):
            self.x = x
        
        def __get__(self,obj,kls=None):
            print '__get__'
            print type(obj), type(self)
            print self.x
            return self.x
        
        def __set__(self,obj,val):
            print '__set__'
            print type(obj), type(self)
            print val
            self.x = val
        
    class TestObject(Object):
        def __init__(self,x):
            self.x = TestDescriptor(x)
        
    
    o = TestObject([1,2,3])
    
    print o.x
    
    o.x = [3,4,5]
            