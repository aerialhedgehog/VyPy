
import os
import pylab as plt


def savefig(filename):
    filename = os.path.splitext(filename)[0]
    folder,filename = os.path.split(filename)
    
    filename_png = os.path.join(folder,filename+'.png')
    
    filename_pdf = os.path.join(folder,'pdf')
    if not os.path.exists(filename_pdf): os.makedirs(filename_pdf)
    filename_pdf = os.path.join(filename_pdf,filename+'.pdf')
    
    plt.savefig(filename_png,dpi=400)
    plt.savefig(filename_pdf)
    