############################################################## 
# Date: 01/01/17
# Name: plot_anomalies6.py
# Author: Alek Petty

import matplotlib
matplotlib.use("AGG")
from mpl_toolkits.basemap import Basemap, shiftgrid
import numpy as np
from pylab import *
import numpy.ma as ma
from matplotlib import rc
from netCDF4 import Dataset


rcParams['xtick.major.size'] = 2
rcParams['ytick.major.size'] = 2
rcParams['axes.linewidth'] = .5
rcParams['lines.linewidth'] = .5
rcParams['patch.linewidth'] = .5

rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize']=9
rcParams['ytick.labelsize']=9
rcParams['legend.fontsize']=9
rcParams['font.size']=9
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

m = Basemap(projection='npstere',boundinglat=65,lon_0=0, resolution='l'  )


rawdatapath='../Data/'
weightsdataoutpath='../DataOutput/Weights/'
dataoutpath='../Data/MeltOnset/'
figpath='../Figures/'

start_year=1979
end_year1=2012
end_year2=2013

fmonth=5
xpts=load(dataoutpath+'xpts100km')
ypts=load(dataoutpath+'ypts100km')


r_valsDT1=load(weightsdataoutpath+'rvalsDTconc'+str(fmonth)+str(start_year)+str(end_year1)+'.txt')
r_valsDT2=load(weightsdataoutpath+'rvalsDTmelt'+str(fmonth)+str(start_year)+str(end_year1)+'.txt')
r_valsDT4=load(weightsdataoutpath+'rvalsDTconc'+str(fmonth)+str(start_year)+str(end_year2)+'.txt')
r_valsDT5=load(weightsdataoutpath+'rvalsDTmelt'+str(fmonth)+str(start_year)+str(end_year2)+'.txt')

predvarDT1=load(weightsdataoutpath+'predvarYrDTconc'+str(fmonth)+str(start_year)+str(end_year1)+'.txt')
predvarDT2=load(weightsdataoutpath+'predvarYrDTmelt'+str(fmonth)+str(start_year)+str(end_year1)+'.txt')
predvarDT4=load(weightsdataoutpath+'predvarYrDTconc'+str(fmonth)+str(start_year)+str(end_year2)+'.txt')
predvarDT5=load(weightsdataoutpath+'predvarYrDTmelt'+str(fmonth)+str(start_year)+str(end_year2)+'.txt')



lonsf = Dataset(rawdatapath+'pondsum/lon.nc', 'r')
Plons = lonsf.variables['TLON'][:]
latsf = Dataset(rawdatapath+'pondsum/lat.nc', 'r')
Plats = latsf.variables['TLAT'][:]
xptsP, yptsP=m(Plons, Plats)

#weights31f = Dataset(rawdatapath+'pondsum/weights31.nc', 'r')
#r_valsDT = -weights31f.variables['var1'][0, 0]
weights56f = Dataset(rawdatapath+'pondsum/weights56.nc', 'r')
r_valsDT6 = -weights56f.variables['var1'][0, 0]
r_valsDT3 = -weights56f.variables['var1'][0, 0]

pdtvar1 = Dataset(rawdatapath+'pondsum/apeff_detrend/orgapeff_56_1979_'+str(end_year1)+'_detrend.nc', 'r')
predvarDT3 = pdtvar1.variables['var77'][-1]
pdtvar2 = Dataset(rawdatapath+'pondsum/apeff_detrend/orgapeff_56_1979_'+str(end_year2)+'_detrend.nc', 'r')
predvarDT6 = pdtvar2.variables['var77'][-1]

weights=[]
xptsALL=[]
yptsALL=[]
weights.append(r_valsDT1)
weights.append(r_valsDT2)
weights.append(r_valsDT3)
weights.append(r_valsDT4)
weights.append(r_valsDT5)
weights.append(r_valsDT6)

predvars=[]

predvars.append(predvarDT1)
predvars.append(predvarDT2)
predvars.append(predvarDT3)
predvars.append(predvarDT4)
predvars.append(predvarDT5)
predvars.append(predvarDT6)

xptsALL.append(xpts)
xptsALL.append(xpts)
xptsALL.append(xptsP)
xptsALL.append(xpts)
xptsALL.append(xpts)
xptsALL.append(xptsP)
yptsALL.append(ypts)
yptsALL.append(ypts)
yptsALL.append(yptsP)
yptsALL.append(ypts)
yptsALL.append(ypts)
yptsALL.append(yptsP)



textwidth=5.3
minval=-0.6
maxval=0
levs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
var_str=['SIC', 'MO', 'MP','SIC', 'MO', 'MP']
fig = figure(figsize=(textwidth,textwidth*0.75))

vmins=[-0.5, -50, -1, -0.5, -50, -1]
vmaxs=[0.5, 50, 1, 0.5, 50, 1]
for x in [0, 3]:
	vars()['ax'+str(x+1)]=subplot(2, 3, 1+x)
	
	im1 = m.contourf(xptsALL[x] , yptsALL[x], weights[x],levels=[0.3, 0.8], colors='none', hatches=['//'], zorder=3)
	
	#im1 = m.contour(xptsALL[x] , yptsALL[x], weights[x],levels=[0.3], colors=['m'], zorder=3)
	im2 = m.pcolormesh(xptsALL[x] , yptsALL[x], predvars[x],vmin=-0.5, vmax=0.5, cmap=cm.RdBu_r, shading='flat', zorder=2)
	#im1 = m.pcolormesh(xpts , ypts, rvals, cmap=cm.cubehelix, vmin=minval, vmax=maxval,shading='flat', zorder=2)
	
	m.drawcoastlines(linewidth=0.5, zorder=5)
	m.drawparallels(np.arange(90,-90,-10), linewidth = 0.25, zorder=3)
	m.drawmeridians(np.arange(-180.,180.,30.), linewidth = 0.25, zorder=3)

label_str='Conc anom'
cax = fig.add_axes([0.08, 0.1, 0.2, 0.03])
cbar = colorbar(im2,cax=cax, orientation='horizontal', extend='both', use_gridspec=True)
cbar.set_label(label_str, labelpad=4, rotation=0)
cbar.set_ticks([-0.5, 0, 0.5])
cbar.solids.set_rasterized(True)

for x in [1, 4]:
	vars()['ax'+str(x+1)]=subplot(2, 3, 1+x)

	im3 = m.contourf(xptsALL[x] , yptsALL[x], weights[x],levels=[0.3, 0.8], colors='none', hatches=['//'], zorder=3)
	
	#im3 = m.contour(xptsALL[x] , yptsALL[x], weights[x],levels=[0.3], colors=['m'], zorder=3)
	im4 = m.pcolormesh(xptsALL[x] , yptsALL[x], predvars[x],vmin=-50, vmax=50, cmap=cm.RdBu_r, shading='flat', zorder=2)
	#im1 = m.pcolormesh(xpts , ypts, rvals, cmap=cm.cubehelix, vmin=minval, vmax=maxval,shading='flat', zorder=2)
	
	m.drawcoastlines(linewidth=0.5, zorder=5)
	m.drawparallels(np.arange(90,-90,-10), linewidth = 0.25, zorder=3)
	m.drawmeridians(np.arange(-180.,180.,30.), linewidth = 0.25, zorder=3)

label_str2='Melt onset anom (day)'
cax2 = fig.add_axes([0.37, 0.1, 0.2, 0.03])
cbar2 = colorbar(im4,cax=cax2, orientation='horizontal', extend='both', use_gridspec=True)
cbar2.set_label(label_str2, labelpad=4, rotation=0)
cbar2.set_ticks([-50, 0, 50])
cbar2.solids.set_rasterized(True)

for x in [2, 5]:
	vars()['ax'+str(x+1)]=subplot(2, 3, 1+x)

	im3 = m.contourf(xptsALL[x] , yptsALL[x], weights[x],levels=[0.3, 0.8], colors='none', hatches=['//'], zorder=3)
	
	im4 = m.pcolormesh(xptsALL[x] , yptsALL[x], predvars[x],vmin=-0.1, vmax=0.1, cmap=cm.RdBu_r, shading='flat', zorder=2)

	m.drawcoastlines(linewidth=0.5, zorder=5)
	m.drawparallels(np.arange(90,-90,-10), linewidth = 0.25, zorder=3)
	m.drawmeridians(np.arange(-180.,180.,30.), linewidth = 0.25, zorder=3)

label_str3='Melt pond frac anom'
cax3 = fig.add_axes([0.66, 0.1, 0.2, 0.03])
cbar3 = colorbar(im4,cax=cax3, orientation='horizontal', extend='both', use_gridspec=True)
cbar3.set_label(label_str3, labelpad=4, rotation=0)
cbar3.set_ticks([-0.1, 0, 0.1])
cbar3.solids.set_rasterized(True)

ax1.annotate(var_str[0], xy=(0.5, 1.01),xycoords='axes fraction', horizontalalignment='center', verticalalignment='bottom', zorder=10)
ax2.annotate(var_str[1], xy=(0.5, 1.01),xycoords='axes fraction', horizontalalignment='center', verticalalignment='bottom', zorder=10)
ax3.annotate(var_str[2], xy=(0.5, 1.01),xycoords='axes fraction', horizontalalignment='center', verticalalignment='bottom', zorder=10)


ax1.annotate(str(end_year1), xy=(0.0, 0.5), xycoords='axes fraction', horizontalalignment='right', verticalalignment='center', rotation=90, zorder=10)
ax4.annotate(str(end_year2), xy=(0.0, 0.5),xycoords='axes fraction', horizontalalignment='right', verticalalignment='center', rotation=90, zorder=10)


subplots_adjust(wspace=0.02, hspace=0.03, bottom=0.14, top=0.95, left=0.04, right=0.9)
savefig(figpath+'anomalies'+str(end_year1)+str(end_year2)+str(fmonth)+'.png', dpi=300)
close(fig)




