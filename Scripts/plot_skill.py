############################################################## 
# Date: 01/10/17
# Name: plot_skill.py
# Author: Alek Petty
# Description: Script to plot skill values for different forecasts.

import matplotlib
matplotlib.use("AGG")
from mpl_toolkits.basemap import Basemap, shiftgrid
import numpy as np
from pylab import *
import numpy.ma as ma
from matplotlib import rc


rcParams['xtick.major.size'] = 2
rcParams['ytick.major.size'] = 2
rcParams['axes.linewidth'] = .5
rcParams['lines.linewidth'] = .5
rcParams['patch.linewidth'] = .5
rcParams['axes.labelsize'] = 8
rcParams['xtick.labelsize']=8
rcParams['ytick.labelsize']=8
rcParams['legend.fontsize']=8
rcParams['font.size']=8
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
skilldatapath='../DataOutput/SkillVals/'
#dataoutpathC='./Data_output/CONC_OUT/'
figpath='../Figures/'

endYear=2016
startYearPred=1985
skills=[]
skillsP=[]

# PUT SKILL ON ONE SIDE AND FERR ON OTHER AXIS.
varstrs=["conc", "melt", "melt_nan", "pond"]

skill_num=0
skills=np.zeros((6, 6, 2))

for m in xrange(2, 8):
	skillC = loadtxt(skilldatapath+'Skill_'+varstrs[0]+str(m)+str(startYearPred)+str(endYear)+'W1.txt', skiprows=1)
	skillCU = loadtxt(skilldatapath+'Skill_'+varstrs[0]+str(m)+str(startYearPred)+str(endYear)+'W0.txt', skiprows=1)
	skillM = loadtxt(skilldatapath+'Skill_'+varstrs[1]+str(m)+str(startYearPred)+str(endYear)+'W1.txt', skiprows=1)
	skillMU = loadtxt(skilldatapath+'Skill_'+varstrs[1]+str(m)+str(startYearPred)+str(endYear)+'W0.txt', skiprows=1)
	skillMN = loadtxt(skilldatapath+'Skill_'+varstrs[2]+str(m)+str(startYearPred)+str(endYear)+'W1.txt', skiprows=1)
	
	if ((m==4)|(m==5)):
		skillP = loadtxt(skilldatapath+'Skill_'+varstrs[3]+str(m)+str(startYearPred)+str(endYear)+'W1.txt', skiprows=1)
	else:
		skillP=[np.nan, np.nan, np.nan, np.nan]
		
	skills[:,m-2, 0]=(skillC[0], skillCU[0], skillM[0], skillMU[0], skillMN[0], skillP[0])
	skills[:,m-2, 1]=(skillC[3], skillCU[3], skillM[3], skillMU[3], skillMN[3], skillP[3])


Concdays=np.arange(90, 260, 30)
Meltdays=np.arange(90, 220, 30)
Combdays=np.arange(150, 200, 30)

fig = figure(figsize=(3.5,3.8))
ax1=subplot(2, 1, 1)
im1 = plot(Concdays, skills[0, :, 0], 'o',color='b', linestyle='-', markersize=5, alpha=0.8)
im2 = plot(Concdays, skills[1, :, 0], 'v',color='b', linestyle='--', markersize=3, alpha=0.8)
im3 = plot(Concdays, skills[2, :, 0], 'o',color='r', linestyle='-', markersize=5, alpha=0.8)
im4 = plot(Concdays, skills[3, :, 0], 'v',color='r', linestyle='--', markersize=3, alpha=0.8)
im5 = plot(Concdays, skills[4, :, 0], 'o',color='g', linestyle='-', markersize=5, alpha=0.8)

im6 = plot(Combdays, skills[5, 2:4, 0], 's',color='k', linestyle='-', markersize=5, alpha=0.8)

ax1.axhline(0, linestyle='--', color='k')

ylim(-0.5, 1.)
xlim(20, 250)

ax2=subplot(2, 1, 2)
im1 = plot(Concdays, skills[0, :, 1], 'o',color='b', linestyle='-', markersize=5, alpha=0.8)
im2 = plot(Concdays, skills[1, :, 1], 'v',color='b', linestyle='--', markersize=3, alpha=0.8)
im3 = plot(Concdays, skills[2, :, 1], 'o',color='r', linestyle='-', markersize=5, alpha=0.8)
im4 = plot(Concdays, skills[3, :, 1], 'v',color='r', linestyle='--', markersize=3, alpha=0.8)
im5 = plot(Concdays, skills[4, :, 1], 'o',color='g', linestyle='-', markersize=5, alpha=0.8)

im6 = plot(Combdays, skills[5, 2:4, 1], 's',color='k', linestyle='-', markersize=5, alpha=0.8)

ax2.axhline(0, linestyle='--', color='k')

ylim(-0.5, 1.)
xlim(20, 250)

ax1.set_ylabel('Skill', labelpad=3)
ax2.set_ylabel('Skill', labelpad=3)

ax2.set_xlabel('Forecast month')
ax1.set_xticks(np.arange(30, 250, 30))
ax1.set_xticklabels([])
ax2.set_xticks(np.arange(30, 250, 30))
ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug'])
ax1.annotate('(a) 1985-2016', xy=(0.03, 1.01), xycoords='axes fraction', verticalalignment='bottom')
ax2.annotate('(b) 2008-2016', xy=(0.03, 1.01), xycoords='axes fraction', verticalalignment='bottom')
#ax1.xaxis.grid(True)
#ax1.yaxis.grid(True)

plts_net = im1+im2+im3+im4+im5+im6

methods = ['SIC','SICuw','MO', 'MOuw', 'MOmask', 'MP']

leg = ax1.legend(plts_net, methods, loc=2, ncol=2,columnspacing=1., frameon=False,handletextpad=1, borderaxespad=0.1)


subplots_adjust(left=0.13, right=0.98, top=0.96, bottom=0.1, hspace=0.1)
savefig(figpath+'skill_monthsALL'+str(skill_num)+'N1.pdf', dpi=300)
close(fig)







