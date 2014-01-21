
""" Backport of OrderedDict() class that runs on Python 2.4, 2.5, 2.6, 2.7 and pypy.
    Passes Python2.7's test suite and incorporates all the latest updates.
    {{{ http://code.activestate.com/recipes/576693/ (r9)
"""

import re
from Dict import Dict

try:
    from thread import get_ident as _get_ident
except ImportError:
    from dummy_thread import get_ident as _get_ident

try:
    from _abcoll import KeysView, ValuesView, ItemsView
except ImportError:
    pass


class OrderedDict(Dict):
    """Dictionary that remembers insertion order"""
    # An inherited dict maps keys to values.
    # The inherited dict provides __getitem__, __len__, __contains__, and get.
    # The remaining methods are order-aware.
    # Big-O running times for all methods are the same as for regular dictionaries.

    # The internal self.__map dictionary maps keys to links in a doubly linked list.
    # The circular doubly linked list starts and ends with a sentinel element.
    # The sentinel element never gets deleted (this simplifies the algorithm).
    # Each link is stored as a list of length three:  [PREV, NEXT, KEY].
    
    __root = None
    __map  = None
    
    def __new__(klass,*args,**kwarg):

        self = super(OrderedDict,klass).__new__(klass)
        
        if len(args) > 1:
            raise TypeError('expected at most 1 arguments, got %d' % len(args))
        if self.__root is None:
            self.__root = root = [] # sentinel node
            root[:] = [root, root, None]
            self.__map = {}
        
        return self

    def __init__(self, *args, **kwds):
        '''Initialize an ordered dictionary.  Signature is the same as for
        regular dictionaries, but keyword arguments are not recommended
        because their insertion order is arbitrary.
        '''
        self.__update(*args, **kwds)


    def __setitem__(self, key, value):
        'od.__setitem__(i, y) <==> od[i]=y'
        # Setting a new item creates a new link which goes at the end of the linked
        # list, and the inherited dictionary is updated with the new key/value pair.
        if key not in self:
            root = self.__root
            last = root[0]
            last[1] = root[0] = self.__map[key] = [last, root, key]
        super(OrderedDict,self).__setitem__(key, value)

    def __delitem__(self, key):
        'od.__delitem__(y) <==> del od[y]'
        # Deleting an existing item uses self.__map to find the link which is
        # then removed by updating the links in the predecessor and successor nodes.
        super(OrderedDict,self).__delitem__(key)
        link_prev, link_next, key = self.__map.pop(key)
        link_prev[1] = link_next
        link_next[0] = link_prev

    def __iter__(self):
        'od.__iter__() <==> iter(od)'
        root = self.__root
        curr = root[1]
        while curr is not root:
            yield curr[2]
            curr = curr[1]

    def __reversed__(self):
        'od.__reversed__() <==> reversed(od)'
        root = self.__root
        curr = root[0]
        while curr is not root:
            yield curr[2]
            curr = curr[0]

    def clear(self):
        'od.clear() -> None.  Remove all items from od.'
        try:
            for node in self.__map.itervalues():
                del node[:]
            root = self.__root
            root[:] = [root, root, None]
            self.__map.clear()
        except AttributeError:
            pass
        dict.clear(self)

    def popitem(self, last=True):
        '''od.popitem() -> (k, v), return and remove a (key, value) pair.
        Pairs are returned in LIFO order if last is true or FIFO order if false.

        '''
        if not self:
            raise KeyError('dictionary is empty')
        root = self.__root
        if last:
            link = root[0]
            link_prev = link[0]
            link_prev[1] = root
            root[0] = link_prev
        else:
            link = root[1]
            link_next = link[1]
            root[1] = link_next
            link_next[0] = root
        key = link[2]
        del self.__map[key]
        value = dict.pop(self, key)
        return key, value

    # -- the following methods do not depend on the internal structure --
    
    # allow override of __iter__
    __iter = __iter__

    def keys(self):
        'od.keys() -> list of keys in od'
        return list(self.__iter())
    
    def values(self):
        'od.values() -> list of values in od'
        return [self[key] for key in self.__iter()]

    def items(self):
        'od.items() -> list of (key, value) pairs in od'
        return [(key, self[key]) for key in self.__iter()]

    def iterkeys(self):
        'od.iterkeys() -> an iterator over the keys in od'
        return self.__iter()

    def itervalues(self):
        'od.itervalues -> an iterator over the values in od'
        for k in self.__iter():
            yield self[k]

    def iteritems(self):
        'od.iteritems -> an iterator over the (key, value) items in od'
        for k in self.__iter():
            yield (k, self[k])
            
    def append(self,key_wild,val):
        key = self.next_key(key_wild)
        self[key] = val
    
    def update(*args, **kwds):
        '''od.update(E, **F) -> None.  Update od from dict/iterable E and F.

        If E is a dict instance, does:           for k in E: od[k] = E[k]
        If E has a .keys() method, does:         for k in E.keys(): od[k] = E[k]
        Or if E is an iterable of items, does:   for k, v in E: od[k] = v
        In either case, this is followed by:     for k, v in F.items(): od[k] = v

        '''
        if len(args) > 2:
            raise TypeError('update() takes at most 2 positional '
                            'arguments (%d given)' % (len(args),))
        elif not args:
            raise TypeError('update() takes at least 1 argument (0 given)')
        self = args[0]
        # Make progressively weaker assumptions about "other"
        other = ()
        if len(args) == 2:
            other = args[1]
        if hasattr(other, 'iterkeys'):
            for key in other.keys():
                self[key] = other[key]
        elif hasattr(other, 'keys'):
            for key in other.keys():
                self[key] = other[key]
        else:
            for key, value in other:
                self[key] = value
        for key, value in kwds.items():
            self[key] = value

    __update = update  # let subclasses override update without breaking __init__

    __marker = object()

    def pop(self, key, default=__marker):
        '''od.pop(k[,d]) -> v, remove specified key and return the corresponding value.
        If key is not found, d is returned if given, otherwise KeyError is raised.

        '''
        if key in self:
            result = self[key]
            del self[key]
            return result
        if default is self.__marker:
            raise KeyError(key)
        return default

    def setdefault(self, key, default=None):
        'od.setdefault(k[,d]) -> od.get(k,d), also set od[k]=d if k not in od'
        if key in self:
            return self[key]
        self[key] = default
        return default
    
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

    def __repr__(self, _repr_running={}):
        'od.__repr__() <==> repr(od)'
        call_key = id(self), _get_ident()
        if call_key in _repr_running:
            return '...'
        _repr_running[call_key] = 1
        try:
            if not self:
                return '%s()' % (self.__class__.__name__,)
            return '%s(%r)' % (self.__class__.__name__, self.items())
        finally:
            del _repr_running[call_key]

    def __reduce__(self):
        'Return state information for pickling'
        items = [[k, super(Dict,self).__getitem__(k) ] for k in self.__iter()]
        inst_dict = vars(self).copy()
        for k in vars(OrderedDict()):
            inst_dict.pop(k, None)
        return (_reconstructor, (self.__class__,items,), inst_dict)

    def copy(self):
        'od.copy() -> a shallow copy of od'
        return self.__class__(self)

    @classmethod
    def fromkeys(cls, iterable, value=None):
        '''OD.fromkeys(S[, v]) -> New ordered dictionary with keys from S
        and values equal to v (which defaults to None).

        '''
        d = cls()
        for key in iterable:
            d[key] = value
        return d

    def __eq__(self, other):
        '''od.__eq__(y) <==> od==y.  Comparison to another OD is order-sensitive
        while comparison to a regular mapping is order-insensitive.

        '''
        if isinstance(other, OrderedDict):
            return len(self)==len(other) and self.items() == other.items()
        return dict.__eq__(self, other)

    def __ne__(self, other):
        return not self == other

    # -- the following methods are only used in Python 2.7 --

    def viewkeys(self):
        "od.viewkeys() -> a set-like object providing a view on od's keys"
        return KeysView(self)

    def viewvalues(self):
        "od.viewvalues() -> an object providing a view on od's values"
        return ValuesView(self)

    def viewitems(self):
        "od.viewitems() -> a set-like object providing a view on od's items"
        return ItemsView(self)
    
## end of http://code.activestate.com/recipes/576693/ }}}

# for rebuilding disctionaries with attributes
def _reconstructor(klass,items):
    self = OrderedDict.__new__(klass)
    OrderedDict.__init__(self,items)
    return self


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
        
    class TestObject(OrderedDict):
        pass
        #def __init__(self,c):
            #self.c = c
    
    o = TestObject()
    o['x'] = TestDescriptor([1,2,3])
    o['y'] = TestDescriptor([4,3,5])
    for i in range(10):
        o['x%i'%i] = 'yo'    
    
    print ''
    print o
    #print o.c
    
    print ''
    o['x'] = [3,4,5]
            
    print ''
    print 'pickle'
    import pickle
        
    d = pickle.dumps(o)
    p = pickle.loads(d)
    
    print ''
    print p
    #print p.c