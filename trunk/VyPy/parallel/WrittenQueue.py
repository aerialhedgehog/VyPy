
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import os

from VyPy.data import filelock, FileLockException
from VyPy.data import load, save
from VyPy.tools import wait

from Full import Full
from Empty import Empty

MASTER_TIMEOUT = 10.0

# ----------------------------------------------------------------------
#   WrittenQueue
# ----------------------------------------------------------------------

class WrittenQueue(object):
    
    def __init__( self, filename, max_size=None, delay=0.5 ):
        self.filename = os.path.abspath(filename)
        self.delay = delay
        # start the queue data file if needed
        if not os.path.exists(self.filename):
            queue = QueueData()
            queue.max_size = max_size
            save(queue,self.filename)
        # add a listener under a filelock
        with filelock(self.filename,timeout=MASTER_TIMEOUT):
            queue = load(self.filename)
            queue.listeners += 1
            save(queue,self.filename)
            
    
    def put( self, task, block=True, timeout=None ):
        ''' put a task on the queue
        '''
        if not block: timeout = 0.0
        def check():
            with filelock(self.filename,timeout=0.0):
                queue = load(self.filename)
                if queue.unfinished_tasks >= queue.max_size:
                    raise Full
                queue.unfinished_tasks += 1
                queue.task_list.append(task)
                save(queue,self.filename)
        wait(check,timeout,self.delay )
        
       
    def get( self, block=True, timeout=None ):
        ''' get a task from the queue
        '''
        if not block: timeout = 0.0
        def check():
            with filelock(self.filename,timeout=0.0):
                queue = load(self.filename)
                try: 
                    task = queue.task_list.pop()
                    save(queue,self.filename)
                except IndexError:
                    raise Empty
            return task
        return wait(check,timeout,self.delay)


    def task_done( self, block=True, timeout=None ):
        if not block: timeout = 0.0
        def check():
            with filelock(self.filename,timeout=0.0):
                queue = load(self.filename)
                
                if queue.unfinished_tasks <= 0:
                    print 'warning: task_done() called too many times'
                else: 
                    queue.unfinished_tasks -= 1
                save(queue,self.filename)
        wait(check,timeout,self.delay)
        
        
    def join(self):
        def check():
            with filelock(self.filename,timeout=0.0):
                queue = load(self.filename)
                if unfinished_tasks > 0:
                    raise Full
        wait(check,timeout=None,delay=self.delay)
        
    def __del__(self):
        with filelock(self.filename,timeout=MASTER_TIMEOUT):
            queue = load(self.filename)
            queue.listeners -= 1
            if queue.listeners <= 0:
                os.remove(self.filename)
            else:
                save(queue,self.filename)
        
        
# ----------------------------------------------------------------------
#   QueueData
# ----------------------------------------------------------------------

class QueueData(object):
    
    def __init__(self):
        self.listeners = 0
        self.unfinished_tasks = 0
        self.task_list = []
        self.max_size = None
        