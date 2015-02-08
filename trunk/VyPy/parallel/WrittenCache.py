

from VyPy.data import filelock, load, save, HashedDict
import os

class WrittenCache(object):
    
    def __init__( self, filename, initial=None, timeout=None, delay=0.5):
        self.filename = os.path.abspath(filename)
        self.timeout = timeout
        self.delay = delay
        if not (os.path.exists(filename) and os.path.isfile(filename)) and not initial:
            initial = HashedDict()
        if not initial is None:
            save(initial,self.filename,file_format='pickle') 
        
    def get(self,key):
        cache = load(self.filename,file_format='pickle')
        return cache[key]
    
    def set(self,key,value):
        with filelock(self.filename,self.timeout,self.delay) as lock:
            cache = load(self.filename,lock=lock,file_format='pickle')
            cache[key] = value
            save(cache,self.filename,lock=lock,file_format='pickle')        
           
    def __getitem__(self,key):
        return self.get(key)
    def __setitem__(self,key,value):
        self.set(key,value)