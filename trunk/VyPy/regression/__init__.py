
import gpr
import gpr.library
import least_squares

# will fail if cvxopt is not installed
try:
    import active_subspace
except ImportError as exc:
    from warnings import warn
    warn('could not import VyPy.regression.active_subspace: %s' % exc, ImportWarning)
