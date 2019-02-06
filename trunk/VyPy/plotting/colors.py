

from VyPy.data import obunch, odict
import matplotlib
from matplotlib.colors import hex2color

colors = obunch()

colors.sky_blue = odict()
colors.sky_blue[-2] = hex2color('#cadee5')
colors.sky_blue[-1] = hex2color('#84cfeb')
colors.sky_blue[0] = hex2color('#18aae0')
colors.sky_blue[1] = hex2color('#197192')
colors.sky_blue[2] = hex2color('#112128')

colors.hot_purple = odict()
colors.hot_purple[-2] = hex2color('#e5cadd')
colors.hot_purple[-1] = hex2color('#e77ac7')
colors.hot_purple[0] = hex2color('#e018a5')
colors.hot_purple[1] = hex2color('#841665')
colors.hot_purple[2] = hex2color('#281121')

colors.yellow_green = odict()
colors.yellow_green[-2] = hex2color('#dfe5ca')
colors.yellow_green[-1] = hex2color('#c7e77a')
colors.yellow_green[0] = hex2color('#a9e018')
colors.yellow_green[1] = hex2color('#678416')
colors.yellow_green[2] = hex2color('#222811')

