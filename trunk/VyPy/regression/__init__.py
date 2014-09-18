
import gpr
import gpr.library
import least_squares

# will fail if cvxopt is not installed
try:
    import active_subspace
except ImportError:
    pass
