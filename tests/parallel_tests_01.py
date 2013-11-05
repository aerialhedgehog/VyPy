
import numpy as np
import os, sys, shutil, time

import VyPy
from VyPy import parallel as para


def main():
    
    #operation(test_func)
    #remember(test_func)
    #multitask(test_func)
    #multiremember(test_func)
    test_all()
    
def operation(function):
    
    print function
    
    resource = para.Resource.ComputeCores(max_cores=4)
    print resource
    
    gate = resource.gate(default=2)
    print gate
    
    function = para.Operation(function,gate)
    print function
    
    #print function(10.)
    
    return function
    
    
def remember(function):
    
    function = operation(function)
    print function
    
    function = para.Remember(function)
    print function  
    
    #print function(10.)
    #print function(10.)
    
    return function
    
def multitask(function):
    
    function = operation(function)
    print function
    
    function = para.MultiTask(function,copies=4)
    print function  
    
    #print function(10.)
    #print function(10.)
    
    return function
    
def multiremember(function):
    
    function = remember(function)
    print function
    
    function = para.MultiTask(function,copies=4)
    print function
    
    print function(10.)
    print function(10.)
    
    
    
    
def test_all():
    
    function = test_func
    print function
    
    resource = para.Resource.ComputeCores(max_cores=4)
    print resource
    
    gate = resource.gate(default=2)
    print gate
    
    function = para.Operation(function,gate)
    print function
    
    function = para.Remember(function)
    print function    
    
    function = para.MultiTask(function,copies=4)
    print function
    
    #function = para.Service(function)
    #print function
    
    #function = function.remote()
    #print function
    
    print function(1.)
    


def test_func(x):
    y = x*2.
    print 'wait...'
    sys.stdout.flush()
    time.sleep(1.0)
    print x, y
    return y


if __name__ == '__main__':
    main()