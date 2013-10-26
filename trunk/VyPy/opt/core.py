
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

# standard 
import os, sys, time, traceback, inspect, pickle
from warnings import warn
from collections import OrderedDict as odict

# multiprocessing
import multiprocessing as mp
import multiprocessing.queues

# helpers
from check_pid import check_pid
from hashing import hashed_dict, make_hashable
import redirect

# numpy
try:
    import numpy as np
    array_type  = np.ndarray
    matrix_type = np.matrixlib.defmatrix.matrix
    numpy_isloaded = True
except ImportError:
    numpy_isloaded = False

# ----------------------------------------------------------------------
#   Problem
# ----------------------------------------------------------------------
class Problem(object):
    
    def __init__(self):
        
        self.variables    = []
        self.objectives   = []
        self.equalities   = []
        self.inequalities = []
                
    def compile(self):
        
        # sort objectives        
        self.variables = Variables(self.variables)
        var = self.variables
        
        # sort objectives
        self.objectives = [ Objective(obj,var) for obj in self.objectives ]
        
        # sort constraints
        for con in self.constraints:
            _,(_,sgn,_),_ = con
            # equality constraint
            if sgn == '=':
                con = Equality(con,var)
                self.equalities.append(con)
            # inequality constraint
            elif sgn in '><':
                con = Inequality(con,var)
                self.inequalities.append(con)
            # uhoh
            else:
                raise Exception, 'unrecognized sign %s' % sgn
        #: for each constraint
        
        return
    
    def has_gradients(self):
        
        # objectives
        grads = [ not evalr.gradient is None for evalr in self.objectives ]
        obj_grads = any(grads) and all(grads)
        
        # inequalities
        grads = [ not evalr.gradient is None for evalr in self.inequalities ]
        ineq_grads = any(grads) and all(grads)
            
        # equalities
        grads = [ not evalr.gradient is None for evalr in self.equalities ]
        eq_grads = any(grads) and all(grads)            
           
        return obj_grads, ineq_grads, eq_grads
        


# ----------------------------------------------------------------------
#   Objective Function
# ----------------------------------------------------------------------
class Objective(object):
    def __init__(self,objective,variables):
        evalr,key,scl = objective
        
        if not isinstance(evalr, Evaluator):
            func = evalr
            evalr = Evaluator()
            evalr.function = func
        
        self.evaluator = evalr
        self.output    = key
        self.scale     = scl
        self.variables = variables
        
        if not hasattr(objective,'gradient'):
            self.gradient = None
        
    def function(self,x):
        
        x = self.variables.scaled.unpack(x)
        
        func = self.evaluator.function
        key  = self.output
        scl  = self.scale
        
        result = func(x)[key]
        
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
    

# ----------------------------------------------------------------------
#   Equality Function
# ----------------------------------------------------------------------
class Equality(object):
    def __init__(self,constraint,variables):
        evalr,(key,sgn,val),scl = constraint
        
        if not isinstance(evalr, Evaluator):
            func = evalr
            evalr = Evaluator()
            evalr.function = func
        
        self.evaluator = evalr
        self.output    = key
        self.sign      = sgn
        self.value     = val
        self.scale     = scl
        self.variables = variables
        
        if not hasattr(constraint,'gradient'):
            self.gradient = None        
        
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
    
# ----------------------------------------------------------------------
#   Inequality Function
# ----------------------------------------------------------------------
class Inequality(object):
    def __init__(self,constraint,variables):
        evalr,(key,sgn,val),scl = constraint

        if not isinstance(evalr, Evaluator):
            func = evalr
            evalr = Evaluator()
            evalr.function = func        
        
        self.evaluator = evalr
        self.output    = key
        self.sign      = sgn
        self.value     = val
        self.scale     = scl
        self.variables = variables
        
        if not hasattr(constraint,'gradient'):
            self.gradient = None        
        
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


# ----------------------------------------------------------------------
#   Variables
# ----------------------------------------------------------------------
class Variables(object):
    def __init__(self,inputs,scale=1.0):
        
        names,initial,bounds,_ = self.process_inputs(inputs)
        
        self.names   = names
        self.initial = initial
        self.bounds  = bounds
        
        self.scaled  = ScaledVariables(inputs)

        return
    
    def process_inputs(self,inputs):
        
        names   = []
        initial = []
        bounds  = []
        scales  = []
        
        default_bnd = [-1e100,1e100]
        
        for i,vals in enumerate( inputs ):
            
            if not isinstance(vals,(list,tuple)):
                # init only
                init = vals
                initial.append(init)
                
            else:
                # ['name' , vals , scale]
                if len(vals) == 3:
                    name,vals,scale = vals
                    names.append(name)
                    scales.append(scale)
                    
                # ['name' , vals ]
                elif len(vals) == 2:
                    name,vals = vals
                    scale = 1.0
                    names.append(name)
                    scales.append(scale)
                    
                # no name or scale
                else:
                    name = 'Var_%i' % i
                    scale = 1.0
                    names.append(name)
                    scales.append(scale)
                
                # vals = init only
                if not isinstance(vals,(list,tuple)):
                    init = vals
                    initial.append(init)
                    bounds.append(default_bnd)
                    
                # vals = (init)
                elif len(vals) == 1:
                    init = vals[0]
                    initial.append(init)
                    bounds.append(default_bnd)
                
                # vals = (lb,x0,ub)
                else:
                    lb,init,ub = vals
                    bound = [lb,ub]
                    initial.append(init)
                    bounds.append(bound)
            
            #: if named
            
        #: for each var
            
        return names,initial,bounds,scales
    
    def unpack(self,values):
        
        if self.names:
            # Unpack Variables
            names = self.names
            variables = odict()
            
            for name,val in zip(names,values):
                variables[name] = val
                
        else:
            variables = values
                
        return variables        
    
    def pack(self,variables):
        
        assert self.names , 'variable names not defined'
        names = self.names
        
        # Pack Variables
        values = [ variables[n] for n in names ]
                
        return values
    
    def update_initial(self,variables):
        x0 = self.pack(variables)
        x0_scl = self.scaled.pack(variables)
        
        for i,(v,vs) in enumerate(zip(x0,x0_scl)):
            self.initial[i] = v
            self.scaled.initial[i] = vs
        
        return
        
class ScaledVariables(Variables):
    def __init__(self,inputs):
        
        names,initial,bounds,scales = self.process_inputs(inputs)
        
        for i in range(len(names)):
            scale = scales[i]
            initial[i] = initial[i]*scale
            bounds[i] = [ b*scale for b in bounds[i] ]
            
        self.names   = names
        self.initial = initial
        self.bounds  = bounds
        self.scales  = scales
        
        return
        
    def unpack(self,values):
        scales = self.scales
        values = [ v/s for v,s in zip(values,scales) ]
        variables = Variables.unpack(self,values)
        return variables
    
    def pack(self,variables):
        scales = self.scales
        values = Variables.pack(self,variables)
        values = [ v*s for v,s in zip(values,scales) ]
        return values
        
# ----------------------------------------------------------------------
#   Driver
# ----------------------------------------------------------------------        
class Driver(object):
    pass
        
# ----------------------------------------------------------------------
#   Evaluator
# ----------------------------------------------------------------------
class Evaluator(object):
    pass
    #def __init__(self,constants=None,server=None):
        
        #self.setup(constants,server)
        
        #return

    #@staticmethod
    #def function(variables,constants):
        #raise NotImplementedError
        #try:
            ## code
            #outputs = {}
        #except Exception:
            #outputs = {}
        #return outputs 
    
    #@staticmethod
    #def gradient(variables,constants):
        #raise NotImplementedError
        #try:
            ## code
            #outputs = {}
        #except Exception:
            #outputs = {}
        #return outputs
    
    #@staticmethod
    #def hessian(variables,constants):
        #raise NotImplementedError
        #try:
            ## code
            #outputs = {}
        #except Exception:
            #outputs = {}
        #return outputs

        

# ----------------------------------------------------------------------
#   Worker
# ----------------------------------------------------------------------
class Worker(object):
    
    def __init__( self, function, constants=None,
                  nodes=0, server=None, procs=None,
                  outbox=None, inbox=None,
                  name='Worker', verbose=False,
                  filename='', write_freq=1 ):
        # todo: priority, save cache, setup
        
        # initialize function carrier
        if function and not isinstance(function,Function):
            function = Function(function)        
        
        # initialize queues
        inbox  = inbox  or _Queue()
        outbox = outbox or _Queue()

        # initialize cache
        if filename and os.path.exists(filename) and os.path.isfile(filename):
            # load from file
            self.__cache__ = load_data(filename)        
        else:
            # new cache
            self.__cache__ = hashed_dict()
               
        # check processes
        nprocs = nodes
        if nprocs < 0: nprocs = max(mp.cpu_count()+(nprocs+1),0)
        
        # initialize processes
        if verbose: print 'Starting %i Nodes' % nprocs; sys.stdout.flush()
        nodes = [ 
            Process( inbox, None, function, constants, 
                     server, procs, ('%s - Proc %i'%(name,i)), verbose ) 
            for i in xrange(nprocs) 
        ]
        
        # store
        self.function   = function
        self.constants  = constants
        self.name       = name
        self.outbox     = outbox
        self.inbox      = inbox
        self.nodes      = nodes 
        self.verbose    = verbose
        self.filename   = filename
        self.write_freq = write_freq
        
        
    def __func__(self,inputs):
        if not self.constants is None:
            outputs = self.function(inputs,self.constants)
        else:
            outputs = self.function(inputs)
        return outputs
        
    def __call__(self,x):
        
        # check for multiple inputs
        if isinstance(x,(list,tuple)): 
            xx = x   # multiple inputs
            ff = [None] * len(xx)
            f = [None]
        elif numpy_isloaded and isinstance(x,(array_type,matrix_type)):
            xx = x
            ff = np.zeros_like(x)
            f = np.array([None])
        else:
            xx = [x] # single input
            ff = [None]
            f = None
            
        # initialize list of hashables for cache
        _xx = [None]*len(xx)
            
        # submit input
        for i,x in enumerate(xx):
            
            # hashable type for cache
            _x = make_hashable(x)
            _xx[i] = _x
                
            # check cache
            if self.__cache__.has_key(_x): 
                print 'PULLED FROM CACHE'
                continue
                
            # submit to inbox
            else: self.put(x)
                
        # pull output 
        for i,_x in enumerate(_xx):
            # wait for outbox (using hashed x)
            ff[i] = self.get(_x)
            
        # format for output
        if f is None: f = ff[0] # single input
        else:         f = ff    # multiple input
        
        # save cache
        if self.filename and len(self.__cache__)%self.write_freq == 0:
            save_data(self.__cache__,self.filename)
        
        # done
        return f
                
    def put(self,x):
        
        # serial
        if not self.nodes:
            self.__cache__[x] = self.__func__(x)
            
        # submit to queue
        else:
            this_task = Task( inputs   = x             ,
                              function = self.function ,
                              outbox   = self.outbox    )
            self.inbox.put( this_task )
            
        return
        
    def get(self,x):
                
        # the result for returning
        f = None
        
        # wait for results
        while True:
            
            # check cache
            if self.__cache__.has_key(x):
                f = self.__cache__[x]
                break
            
            # serial
            if not self.nodes:
                self.__cache__[x] = self.__func__(x)            
            
            # check outbox queue
            else:
                try:
                    task = self.outbox.get(block=False)
                    x = task.inputs
                    f = task.outputs
                    if isinstance(f,BaseException):
                        warn(('\nFailed Task: inputs = %s' % x),FailedTask)
                    self.__cache__[x] = f
                except mp.queues.Empty:
                    pass
        
        #: while wait for results
        
        return f
    
    def __del__(self):
        for p in self.nodes:
            self.inbox.put(KillTask())
        self.inbox.join()


# ----------------------------------------------------------------------
#   Process
# ----------------------------------------------------------------------
class Process(mp.Process):
    """ preloaded func before task specified func
    """
    def __init__( self, inbox, outbox=None, 
                  function=None, constants=None,
                  server=None, procs=None, name=None,
                  verbose=False ):
        
        # super init
        mp.Process.__init__(self)      
        self.name = name or self.name
        
        # initialize function carrier
        if function and not isinstance(function,Function):
            function = Function(function)
            
        if server: client = server.new_client(self.name)
        else: client = None
        
        # store class vars
        self.inbox     = inbox
        self.outbox    = outbox
        self.function  = function
        self.constants = constants
        self.client    = client
        self.procs     = procs
        self.daemon    = True
        self.parentpid = os.getpid()
        self.verbose   = verbose
        
        self.start()
        
    def __func__(self,inputs,function):
 
        # check out server resources
        with ServerRequest(self.client,self.procs):
            
            if self.verbose: print '%s: Starting task' % self.name; sys.stdout.flush()
            
            if not self.constants is None:
                outputs = function(inputs,self.constants)
            else:
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
            if not self.inbox.empty():
                this_task = self.inbox.get(block=True,timeout=1.0)
            else:
                continue
            
            # report
            if self.verbose: print '%s: Got task' % name; sys.stdout.flush()
            
            # robust process
            try:
                # check task
                if not isinstance(this_task,Task):
                    #raise Exception, 'Task must be of type ViPy.Task'
                    this_task = Task(this_task,self.function)
                
                # unpack task
                this_input    = this_task.inputs
                this_function = this_task.function or self.function
                this_owner    = this_task.owner
                this_folder   = this_task.folder
                
                # check process kill signal
                if isinstance(this_input,KillTask):
                    break                    
                
                # report
                if self.verbose: print '%s: Inputs = %s, Function = %s' % (name,this_input,self.function); sys.stdout.flush()
                
                # execute task, in the folder it was created
                with redirect.folder(this_folder):
                    if self.verbose: print os.getcwd(); sys.stdout.flush()
                    this_output = self.__func__(this_input,this_function)
                    this_task.outputs = this_output
                
                # report
                if self.verbose: print '%s: Task complete' % name; sys.stdout.flush()                
                
                # make sure std prints
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
            
            # end task
            this_outbox = this_task.outbox or self.outbox
            this_task.outbox = None # avoid serialization error
            if this_outbox: this_outbox.put(this_task)
            self.inbox.task_done()
            
        #: while alive
        
        # process end
        self.inbox.task_done()
        if self.verbose: print '%s: Ending' % name; sys.stdout.flush()
        
        return   
    
# ----------------------------------------------------------------------
#   Task
# ----------------------------------------------------------------------
    
class Task(object):
    def __init__(self,inputs,function=None,outbox=None):
        
        if function and not isinstance(function,Function):
            function = Function(function)
            
        self.inputs   = inputs
        self.function = function
        self.outputs  = None 
        self.owner    = os.getpid()
        self.folder   = os.getcwd()
        self.outbox   = outbox
    
class FailedTask(Warning):
    pass

class KillTask(object):
    def __repr__(self):
        return 'KillTask Signal'
        
# ----------------------------------------------------------------------
#   Function
# ----------------------------------------------------------------------
class Function(object):
    def __init__(self, function):
        self.function  = function
    def __call__(self, *arg, **kwarg):  
        # makes object callable
        outputs = self.function(*arg, **kwarg)
        return outputs
    def __repr__(self):
        return '<function %s>' % self.function.__name__

    # pickling
    #def __getstate__(self):
        #dict = self.__dict__.copy()
        #data_dict = cloudpickle.dumps(dict)
        #return data_dict

    #def __setstate__(self,data_dict):
        #self.__dict__ = pickle.loads(data_dict)
        #return    
        

# ----------------------------------------------------------------------
#   Queue
# ----------------------------------------------------------------------

def Queue(maxsize=0):
    manager = mp.Manager()
    queue = manager.JoinableQueue(maxsize)
    return queue

_Queue = mp.queues.JoinableQueue

# ----------------------------------------------------------------------
#   Server
# ----------------------------------------------------------------------
class Server(object):
    
    def __init__(self,max_procs=0):
        
        checkout = _Queue()
        checkin  = _Queue()
        
        manager   = mp.Manager()
        semaphore = manager.Semaphore(max_procs)
        record    = manager.dict()
                
        p_checkout = Process( inbox     = checkout           ,
                              function  = proc_checkout      ,
                              constants = [semaphore,record] ,
                              name      = 'server_checkout'  ,
                              verbose   = False              )
        p_checkin  = Process( inbox     = checkin            ,
                              function  = proc_checkin       ,
                              constants = [semaphore,record] ,
                              name      = 'server_checkin'   ,
                              verbose   = False              )
        
        self.checkout  = checkout
        self.checkin   = checkin
        self.semaphore = semaphore
        self.record    = record
        self.manager   = manager
        
        self.max_procs = max_procs
        
        self.proc_checkout = p_checkout
        self.proc_checkin  = p_checkin
        
        return
        
    def new_client(self,name):
        
        client = Client(name, self.max_procs, self.checkout, self.checkin, self.manager)
        
        return client
    
    def __del__(self):
        self.checkout.put(KillTask())
        self.checkin.put(KillTask())
        self.checkout.join()
        self.checkin.join()
        
        
        
def proc_checkout(inputs,constants):
    
    # setup
    n_proc,name,green_light = inputs
    semaphore,record = constants
    
    # checkout
    for p in range(n_proc):
        semaphore.acquire(blocking=True)
    record[name] = n_proc
    
    # confirm checkout
    green_light.set()
    
    return

def proc_checkin(inputs,constants):
    
    # setup
    n_proc,name = inputs
    semaphore,record = constants
    
    # check request
    r_proc = record[name]
    r_proc -= n_proc
    if n_proc > record[name]:
        n_proc = record[name]
        warn(ServerWarning,'%s attempted to checkin more procs then checkedout'%name)
    
    # checkin
    record[name] -= n_proc
    for p in range(n_proc):
        semaphore.release()
    
    return


# ----------------------------------------------------------------------
#   Client
# ----------------------------------------------------------------------
class Client(object):
    def __init__(self,name,max_procs,checkout,checkin,manager):
        
        self.max_procs   = max_procs
        self._checkout   = checkout
        self._checkin    = checkin
        
        self.n_procs     = 0
        self.name        = name
        self.green_light = manager.Event()
        
        
    def checkout(self,n_procs):
        
        n_total = self.n_procs+n_procs
        if n_total > self.max_procs:
            raise ServerException , 'Server size = %i, not enough procs for %i' % (self.max_procs,n_total)
        
        if n_procs == 0:
            return
        
        checkout    = self._checkout
        name        = self.name
        green_light = self.green_light
        
        green_light.clear()
        checkout.put([n_procs,name,green_light])
        
        green_light.wait()
        self.n_procs += n_procs
        
        return
        
    def checkin(self,n_procs=None):
        
        if n_procs is None:
            n_procs = self.n_procs
        else:
            if n_procs > self.n_procs:
                n_procs = self.n_procs
                warn(ServerWarning,'%s attempted to checkin more procs then checkedout'%name)
                
        if n_procs == 0:
            return
        
        checkin     = self._checkin
        name        = self.name
        green_light = self.green_light
        
        checkin.put([n_procs,name])
        green_light.clear()
        
        self.n_procs -= n_procs
        
        return
    
    def acquire(self,n_procs):
        return ServerRequest(self,n_procs)
        
class ServerRequest(object):
    def __init__(self,client,n_procs):
        self.client = client
        self.n_procs = n_procs
        return
        
    def __enter__(self):
        if self.client:
            self.client.checkout(self.n_procs)
        return 
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.client:
            self.client.checkin(self.n_procs)
        
class ServerException(Exception):
    pass

class ServerWarning(Warning):
    pass


# ----------------------------------------------------------------------
#   File IO
# ----------------------------------------------------------------------
def save_data(data,filename):
    fileout = open(filename,'wb')
    pickle.dump(data,fileout)
    fileout.close()    

def load_data(filename):
    filein = open(filename,'rb')
    data = pickle.load(filein)
    filein.close()
    return data

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():
    
    try: os.remove('worker_cache.pkl')
    except: pass
    
    server = Server(max_procs=2)
    
    worker = Worker( function=test_func, 
                     constants=None, 
                     nodes=2,
                     outbox=Queue(),
                     server= server,
                     procs=2,
                     verbose=True, 
                     filename='worker_cache.pkl' )
    
    worker([1,2])
    worker(np.array([1,2]))
    
    os.chdir('test')
    worker([[1,2],[3,4]])       

def test_func(inputs):
    time.sleep(4.0)
    print inputs
    sys.stdout.flush()
    return inputs

if __name__ == '__main__':
    main()