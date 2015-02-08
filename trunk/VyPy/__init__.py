
from warnings import simplefilter, warn
import os
if not os.environ.has_key('VYPY_WARNINGS'):
    simplefilter('ignore',ImportWarning)

import exceptions

import plugins

import tools
import data

import sampling

# will fail if matplotlib not installed
try:
    import plotting
except ImportError as exc:
    warn('could not import VyPy.plotting: %s' % exc,ImportWarning)

import parallel
import optimize

import regression

# reset import warning
simplefilter('default',ImportWarning)