
try:
    import wxversion
    wxversion.ensureMinimal('2.8')
except:
    raise ImportError, 'Plot_Dock requires wxPython version 2.8+'

else:
    import wx
    import wx.aui
    import matplotlib as mpl
    from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as Canvas
    from matplotlib.backends.backend_wxagg import NavigationToolbar2Wx as Toolbar

# ------------------------------------------------------------
#   Plot Dock
# ------------------------------------------------------------

class PlotDock(wx.Panel):
    def __init__(self, name = 'Plot Dock', size=None, parent=None, id=-1):
        """ SUAVE.Methods.Utilities.Plot_Dock(parent=None,id=-1)
            starts a window for tabbed plots
            
            Inputs - 
                name - name of the dock window
                size (default 800px x 600px) - tuple size of the window (w,h)
                parent (optional) - wx frame object
                id (optional) - name of the plot window
                
            Example - 
                # start single plot notebook
                dock = Plot_Notebook('my notebook')
                axes1 = dock.add_axes('Figure 1')
                axes1.plot([1,2,3],[2,1,4])
                axes2 = dock.add_axes('Figure 1')
                axes2.plot([1,2,3],[2,1,4])
                dock.show()
        """
        
        if parent is None:
            if size is None: size = (800,600)
            
            app = wx.PySimpleApp()
            parent = wx.Frame(None,-1,'Plotter',size=size)
            
            self._app = app
            self._parent = parent            
        
        wx.Panel.__init__(self, parent, id=id)
        self.nb = wx.aui.AuiNotebook(self)
        sizer = wx.BoxSizer()
        sizer.Add(self.nb, 1, wx.EXPAND)
        self.SetSizer(sizer)

    def new_axes(self,name="plot"):
        return self.add(name).gca()

    def add(self,figure="Plot Name"):
        if isinstance(figure,str):
            name = figure
            figure = mpl.figure.Figure()
            figure.gca()
        else:
            name = figure.canvas.get_window_title()
        page = Plot(self.nb,figure=figure)
        self.nb.AddPage(page,name)
        return page.figure
    
    def show(self):
        self._parent.Show()
        self._app.MainLoop()        


# ------------------------------------------------------------
#   Helper Objects
# ------------------------------------------------------------
    
class Plot(wx.Panel):
    def __init__(self, parent, figure, id=-1, **kwargs):
        wx.Panel.__init__(self, parent, id=id, **kwargs)
        
        self.figure = figure
        
        self.canvas = Canvas(self, -1, self.figure)
        self.toolbar = Toolbar(self.canvas)
        self.toolbar.Realize()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas,1,wx.EXPAND)
        sizer.Add(self.toolbar, 0 , wx.LEFT | wx.EXPAND)
        self.SetSizer(sizer)

    
# ------------------------------------------------------------
#   Module Test
# ------------------------------------------------------------

if __name__ == "__main__":
    
    import pylab as plt

    dock = PlotDock('my dock')
    
    fig = plt.figure(1)
    plt.plot([1,2,3],[2,1,4],label='123')
    plt.plot([1,2,3],[2,1,4],label='123')
    axes1 = dock.add(fig)
    
    axes2 = dock.new_axes('Figure 2')
    axes2.plot([1,2,3],[2,10,4])
        
    dock.show()    

    