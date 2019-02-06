    
    
import os, platform

SYSTEM = platform.system()

if SYSTEM == 'Windows':
    import ctypes.wintypes
    kernel32 = ctypes.windll.kernel32
    SYNCHRONIZE = 0x100000
        
def check_pid(pid):
    """ checks if pid is alive (True) or dead (False)
    """
    import time  

    # windows
    if SYSTEM == 'Windows':
        # source: http://www.madebuild.org/blog/?p=30
        
        # try to open process
        process = kernel32.OpenProcess(SYNCHRONIZE, 0, pid)
        if process != 0:
            kernel32.CloseHandle(process)
            return True
        else:
            return False
     
        ## If the process exited recently, a pid may still exist for the handle.
        ## So, check if we can get the exit code.
        #exit_code = ctypes.wintypes.DWORD()
        #exit_code_process = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        #is_running = exit_code_process == 0
        #kernel32.CloseHandle(handle)
     
        ## See if we couldn't get the exit code or the exit code indicates that the
        ## process is still running.
        #return is_running or exit_code.value == 259 # special exit code        
    
    # unix
    else:
        try: 
            # test signal 0 to pid
            os.kill(pid,0)
        except OSError:
            # pid is dead
            return False
        else:
            # pid is alive
            return True
    

#try:
    #import psutil

#except ImportError:
    #from warnings import warn
    #warn('check_pid.py - could not import psutil, VyPy.parallel package will not work',ImportWarning)

#def check_pid(pid):
    #return psutil.pid_exists(pid)