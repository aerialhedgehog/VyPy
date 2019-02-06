
import os
from glob import glob

def next_file(pattern,format='%03d'):
    ''' pattern is a wildcard expression using *
        format is a %% escape sequence
    '''
    
    pattern = os.path.join(*os.path.split(pattern))
    
    files = glob(pattern)
    
    names = pattern.split('*')
    
    indeces = [ int( f.lstrip(names[0]).rstrip(names[1]) )
                for f in files ]
    
    if not indeces:
        next = 0
    else:
        next = max(indeces) + 1
    
    return pattern.replace('*',format) % next

        
    