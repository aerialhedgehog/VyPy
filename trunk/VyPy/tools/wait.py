

import time
from random import random
import traceback

WARN_TIME = 3600.

def wait(check,timeout=None,delay=0.5, *args,**kwarg):
    
    start_time = time.time()
    warned = False
    
    while True:
        try:
            result = check(*args,**kwarg)
            break
        
        except Exception as exc:
            if timeout and (time.time()-start_time) > timeout:
                raise exc
            
            #if (time.time()-start_time) > WARN_TIME and not warned:
                #print "wait(): warning, waiting for a long time - \n%s" % traceback.format_exc() 
                #warned = True
            
        time.sleep( delay*(1.+0.2*random()) )           
            
    return result


#class wait(object):

    #def __init__(self,block=True, timeout=None, interval=1.0):

        #self.block = block
        #self.timeout = timeout
        #self.interval = interval

        #self._start_time = time.time()

    #def __iter__(self):
        #return self

    #def next(self):       
        #if self.timeout and (time.time()-self._start_time) > self.timeout:
            #raise StopIteration
        #time.sleep(self.interval)
        #try:
            #return True
        #except:
            #print 'caught'





#for now in wait(timeout=None):

    #print 'hello!'


