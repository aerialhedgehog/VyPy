# setup.py
# 
# Created:  Trent L., Jan 2013
# Modified:         

""" VyPy setup script
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import sys, os

# ----------------------------------------------------------------------
#   Main - Run Setup
# ----------------------------------------------------------------------

def main():
    
    the_package = 'VyPy'
    version = '1.0.0'
    date = 'January 21, 2013'
    
    if len(sys.argv) >= 2:
        command = sys.argv[1]
    else:
        command = ''
    
    if command == 'uninstall':
        uninstall(the_package,version,date)
    else:
        install(the_package,version,date)
 
 
# ----------------------------------------------------------------------
#   Install Pacakge
# ----------------------------------------------------------------------

def install(the_package,version,date):
    
    # imports
    try:
        from setuptools import setup, find_packages
    except ImportError:
        print 'setuptools required to install'
        sys.exit(1)
    
    # list all sub packages
    exclude = [ d+'.*' for d in os.listdir('.') if d not in [the_package] ]
    packages = find_packages( exclude = exclude )
    
    # run the setup!!!
    setup(
        name = the_package,
        version = '1.0.0', 
        description = 'VyPy: An Optimization Toolbox',
        author = 'Stanford University Aerospace Design Lab (ADL)',
        author_email = 'twl26@stanford.edu',
        maintainer = 'The Developers',
        url = 'adl.stanford.edu',
        packages = packages,
        license = 'BSD',
        platforms = ['Win, Linux, Unix, Mac OS-X'],
        zip_safe  = False,
        long_description = read('../README.md')
    )  
    
    return


# ----------------------------------------------------------------------
#   Un-Install Package
# ----------------------------------------------------------------------

def uninstall(the_package,version,date):
    """ emulates command "pip uninstall"
        just for syntactic sugar at the command line
    """
    
    import sys, shutil
    
    # clean up local egg-info
    try:
        shutil.rmtree(the_package + '.egg-info')
    except:
        pass        
        
    # import pip
    try:
        import pip
    except ImportError:
        print 'pip is required to uninstall this package'
        sys.exit(1)
    
    # setup up uninstall arguments
    args = sys.argv
    del args[0:1+1]
    args = ['uninstall', the_package] + args
    
    # uninstall
    try:
        pip.main(args)
    except:
        pass
    
    return
    
    
# ----------------------------------------------------------------------
#   Helper Functions
# ----------------------------------------------------------------------

def read(path):
    """Build a file path from *paths and return the contents."""
    with open(path, 'r') as f:
        return f.read()
    
# ----------------------------------------------------------------------
#   Call Main
# ----------------------------------------------------------------------
if __name__ == '__main__':
    main()
