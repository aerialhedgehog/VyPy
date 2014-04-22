
#from Object import Object

class Dict(dict):

    ## implement descriptor protocol for items
    #def __getitem__(self,k):
        #try:
            #return super(Dict,self).__getitem__(k).__get__(self,type(self))
        #except AttributeError:
            #return super(Dict,self).__getitem__(k)
        #except KeyError:
            #raise KeyError(k)

    #def __setitem__(self,k,v):
        #try:
            #super(Dict,self).__getitem__(k).__set__(self,v)
        #except AttributeError:
            #super(Dict,self).__setitem__(k,v)
        #except KeyError:
            #raise KeyError(k)

    #def __delitem__(self,k):
        #try:
            #super(Dict,self).__getitem__(k).__del__(self)
        #except AttributeError:
            #super(Dict,self).__delitem__(k)
        #except KeyError:
            #raise KeyError(k)

    ## recuresive updates
    def update(self,other):
        if not isinstance(other,dict):
            raise TypeError , 'input is not a dictionary type'
        for k,v in other.iteritems():
            # recurse only if self's value is a Dict()
            if k.startswith('_'):
                continue
            try:
                self[k].update(v)
            except:
                self[k] = v
        return
    
    def append(self,key_wild,val):
        key = self.next_key(key_wild)
        self[key] = val
    
    # new keys by wild card integer
    def next_key(self,key_wild):
        
        if '%i' not in key_wild:
            return key_wild
        
        ksplit = key_wild.split('%i')
        
        keys = [ int( k.lstrip(ksplit[0]).rstrip(ksplit[1]) ) for k in self.keys()]
        
        if keys:
            key_index = max(keys)+1
        else:
            key_index = 0
        
        key = key_wild % (key_index)
        
        return key

    # prettier printing
    def __repr__(self):
        """ Invertible* string-form of a Bunch.
        """
        keys = self.keys()
        args = ', '.join(['%s=%r' % (key, self[key]) for key in keys if not key.startswith('_')])
        return '%s(%s)' % (self.__class__.__name__, args)

    def __str__(self,indent=''):
        """ String-form of a OrderedBunch.
        """

        new_indent = '  '
        args = ''

        # trunk data name
        if indent: args += '\n'

        # print values   
        for key,value in self.iteritems():
            if key.startswith('_'):
                continue
            
            if isinstance(value,Dict):
                if not value:
                    val = '\n'
                else:
                    try:
                        val = value.__str__(indent+new_indent)
                    except RuntimeError: # recursion limit
                        val = ''
            else:
                val = str(value) + '\n'

            # this key-value
            args+= indent + str(key) + ' : ' + val

        return args


if __name__ == '__main__':

    o = Dict()
    o['x'] = 'hello'
    o['y'] = 1
    o['z'] = [3,4,5]
    o['t'] = Dict()
    o['t']['h'] = 20
    o['t']['i'] = (1,2,3)
    
    print o
    
    import pickle
    
    d = pickle.dumps(o)
    p = pickle.loads(d)
    
    print ''
    print p
    
    o['t']['h'] = 'changed'
    p.update(o)
    p['t'].update(o)

    print ''
    print p


    #class TestDescriptor(object):
        #def __init__(self,x):
            #self.x = x

        #def __get__(self,obj,kls=None):
            #print '__get__'
            #print type(obj), type(self)
            #print self.x
            #return self.x

        #def __set__(self,obj,val):
            #print '__set__'
            #print type(obj), type(self)
            #print val
            #self.x = val

    #class TestObject(Dict):
        #def __init__(self,c):
            #self.c = c

    #o = TestObject(555)
    #o['x'] = TestDescriptor([1,2,3])
    #o['y'] = 1
    #o.desc = TestDescriptor([5,7,8])

    #print ''
    #print o['x']
    #print o['y']
    ##print o.desc
    ##print o.c
    #print o

    #print ''
    #o['x'] = [3,4,5]


    #import pickle

    #d = pickle.dumps(o)
    #p = pickle.loads(d)

    #print ''
    #print p['x']
    #print p['y']
    #print o.c
    #print o.desc