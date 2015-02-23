
try:
    import pylab
except RuntimeError:
    import matplotlib
    matplotlib.use("Agg",warn=False,force=True)
    import pylab

from spiders import (
    spider_axis,
    spider_trace,
)

try:
    from dock import PlotDock
except:
    pass
