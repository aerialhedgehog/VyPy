
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import os, sys, shutil, traceback

import multiprocessing as mp
import multiprocessing.queues

from Task           import Task
from Operation      import Operation
from KillTask       import KillTask
from ShareableQueue import ShareableQueue
from Remote         import Remote

from VyPy.tools import (
    redirect, 
    check_pid, 
)
 
# ----------------------------------------------------------------------
#   Process
# ----------------------------------------------------------------------

class Service(mp.Process):
    """ preloaded func before task specified func
    """
    def __init__( self, function=None, inbox=None, outbox=None, 
                  name=None, verbose=False ):

        # super init
        mp.Process.__init__(self)      
        self.name = name or self.name
        
        # initialize function carrier
        if function and not isinstance(function,Operation):
            function = Operation(function)
        
        # store class vars
        self.inbox     = inbox or ShareableQueue()
        self.outbox    = outbox
        self.function  = function or self.function
        self.daemon    = True
        self.parentpid = os.getpid()
        self.verbose   = verbose
        
        # Remote.__init__(self,self.inbox,self.outbox)       ???
        
        #self.start()
        
    def function(self,inputs):
        raise NotImplementedError
        
    def __func__(self,inputs,function):
        if self.verbose: print '%s: Starting task' % self.name; sys.stdout.flush()
        outputs = function(inputs)
        return outputs

    def run(self):
        
        # setup
        name = self.name
        if self.verbose: print 'Starting %s' % name; sys.stdout.flush()    
        
        # keep on keepin on
        while True:
            
            # check parent process status
            if not check_pid(self.parentpid):
                break
            
            # check task queue
            try:
                this_task = self.inbox.get(block=True,timeout=1.0)
            except (mp.queues.Empty, EOFError, IOError):
                continue
            except Exception,err:
                if self.verbose: 
                    sys.stderr.write( '%s: Get Failed \n' % name )
                    sys.stderr.write( traceback.format_exc() )
                    sys.stderr.write( '\n' )
                    sys.stderr.flush()
                continue
            
            # report
            if self.verbose: print '%s: Got task' % name; sys.stdout.flush()
            
            # robust process
            try:
                # check task
                if not isinstance(this_task,Task):
                    #raise Exception, 'Task must be of type VyPy.Task'
                    this_task = Task(this_task,self.function)
                
                # unpack task
                this_input    = this_task.inputs
                this_function = this_task.function or self.function
                this_owner    = this_task.owner
                this_folder   = this_task.folder
                
                # check process kill signal
                if isinstance(this_input,KillTask.__class__):
                    break
                
                # report
                if self.verbose: print '%s: Inputs = %s, Operation = %s' % (name,this_input,self.function); sys.stdout.flush()
                
                # execute task, in the folder it was created
                with redirect.folder(this_folder):
                    if self.verbose: print os.getcwd(); sys.stdout.flush()
                    this_output = self.__func__(this_input,this_function)
                    this_task.outputs = this_output
                
                # report
                if self.verbose: print '%s: Task complete' % name; sys.stdout.flush()                
                
                # make sure we get std's
                sys.stdout.flush()
                sys.stderr.flush()
                                
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception,err:
                if self.verbose: 
                    sys.stderr.write( '%s: Task Failed \n' % name )
                    sys.stderr.write( traceback.format_exc() )
                    sys.stderr.write( '\n' )
                    sys.stderr.flush()
                this_task.outputs = err
            
            #: try task
            
            # pick outbox
            this_outbox = this_task.outbox or self.outbox
            # avoid serialization error with managed outboxes
            this_task.outbox = None 
            
            # end task
            if this_outbox: this_outbox.put(this_task)
            self.inbox.task_done()
            
        #: while alive
        
        # process end
        self.inbox.task_done()
        if self.verbose: print '%s: Ending' % name; sys.stdout.flush()
        
        return   
        
    def remote(self):
        return Remote(self.inbox,self.outbox)
    
    
# ----------------------------------------------------------------------
#   Tests
# ----------------------------------------------------------------------    

def test_func(x):
        y = x*2.
        print x, y
        return y    
    
if __name__ == '__main__':
    
    inbox = ShareableQueue()
    outbox = ShareableQueue()
    
    function = test_func
    
    service = Service(function, inbox, outbox,
                      name='TestService',verbose=True)
    
    service.start()
    
    inbox.put(10.)
    inbox.put(20.)
    
    print outbox.get().outputs
    print outbox.get().outputs
    
    remote = service.remote()
    
    print remote(30.)
    
    inbox.put(KillTask)
    
    inbox.join()