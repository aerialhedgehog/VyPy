
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
    
    def __init__( self, filename, max_size=None, delay=0.5, new=False):
        self.filename = os.path.abspath(filename)
        self.delay = delay
        # start the queue data file if needed
        if not os.path.exists(self.filename) or new:
            queue = QueueData()
            queue.max_size = max_size
            if os.path.exists(self.filename):
                lock = filelock(self.filename,MASTER_TIMEOUT,self.delay)
            else:
                lock = None
            save(queue,self.filename,file_format='pickle',lock=lock)
    
    def put( self, task, block=True, timeout=None ):
        ''' put a task on the queue
        '''
        if not block: timeout = 0.0
        def check():
            with filelock(self.filename,timeout=0.0) as lock:
                queue = load(self.filename,lock=lock,file_format='pickle')
                if not queue.max_size is None and queue.unfinished_tasks >= queue.max_size:
                    raise Full
                queue.unfinished_tasks += 1
                queue.task_list.append(task)
                save(queue,self.filename,lock=lock,file_format='pickle')
        wait(check,timeout,self.delay )
        
       
    def get( self, block=True, timeout=None ):
        ''' get a task from the queue
        '''
        if not block: timeout = 0.0
        def check():
            with filelock(self.filename,timeout=0.0) as lock:
                queue = load(self.filename,lock=lock,file_format='pickle')
                try: 
                    task = queue.task_list.pop(0)
                    save(queue,self.filename,lock=lock,file_format='pickle')
                except IndexError:
                    raise Empty
            return task
        return wait(check,timeout,self.delay)


    def task_done( self, block=True, timeout=None ):
        if not block: timeout = 0.0
        def check():
            with filelock(self.filename,timeout=0.0) as lock:
                queue = load(self.filename,lock=lock,file_format='pickle')
                queue.unfinished_tasks -= 1
                if queue.unfinished_tasks < 0:
                    print 'warning: task_done() called too many times'
                    queue.unfinished_tasks = 0                    
                save(queue,self.filename,lock=lock,file_format='pickle')
        wait(check,timeout,self.delay)
        
        
    def join(self):
        def check():
            with filelock(self.filename,timeout=0.0) as lock:
                queue = load(self.filename,lock=lock,file_format='pickle')
                if queue.unfinished_tasks > 0:
                    raise Full
        wait(check,timeout=None,delay=self.delay)
        
    def empty(self):
        with filelock(self.filename,timeout=0.0) as lock:
            queue = load(self.filename,lock=lock,file_format='pickle')
            if len(queue.task_list) > 0:
                return False
            else:
                return True
                    
    def close(self):
        with filelock(self.filename,timeout=MASTER_TIMEOUT) as lock:
            os.remove(self.filename)
            
            
# ----------------------------------------------------------------------
#   QueueData
# ----------------------------------------------------------------------

class QueueData(object):
    
    def __init__(self):
        self.unfinished_tasks = 0
        self.task_list = []
        self.max_size = None
        
    def __str__(self):
        args = ''
        
        args += object.__repr__(self).split(' ')[0].lstrip('<') + '\n'
        args += '  unfinished_tasks : %i\n' % self.unfinished_tasks
        args += '  task_length : %i\n' % len(self.task_list)
        
        if not self.max_size:
            args += '  max_size : %i' % 0
        else:
            args += '  max_size : %i' % self.max_size
        
        return args
    
    def __repr__(self):
        return self.__str__()
        