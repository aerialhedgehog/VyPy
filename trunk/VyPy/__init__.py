
import exceptions

import plugins

import tools
import data

import sampling

# will fail if matplotlib not installed
try:
    import plotting
except ImportError as exc:
    from warnings import warn
    warn('could not import VyPy.plotting: %s' % exc)

import parallel
import optimize

import regression