""" 
 calc_Arcticforecast.py
 Date: 30/10/2016
 Author: Alek Petty
 Description: Script to generate Arctic sea ice extent forecast using a given variable (or multiple variables)

 Run with e.g. python calc_Arcticforecast.py --fmonth 5 --fvars conc --weight 1
 NB if you want plot_forecast to run, copy this in from forecast_funcs.py

"""

import matplotlib
matplotlib.use("AGG")
import argparse
import forecast_funcs as ff
from pylab import *

dataPath='../Data/'
figpath='../Figures/'
skilldatapath='../DataOutput/SkillVals/'
linedatapath='../DataOutput/TimeSeries/'
weightdatapath='../DataOutput/Weights/'

# Options for this Script
plotSkill=0
outSkill=1
outLine=1
outWeights=1

month=9 #=SEP
startYear=1979 # Start of forecast training
endYear=2016
startYearPred=1985 # Start of forecasts

yearsP=arange(startYearPred, endYear+1, 1)
numYearsReq=5 #number of years required in a grid cell for it to count towards the training data

# Defaults if no arguments added after the call.
fmonth=5 #5=June
fvars=['conc']
weight=1 # Spatially weighting the data

parser = argparse.ArgumentParser()
parser.add_argument('--fmonth', type=int)
parser.add_argument('--fvars', nargs='+')
parser.add_argument('--weight', type=int)
#parser.add_argument('--numVars', type=int)
args=parser.parse_args()
fvars=args.fvars
weight=args.weight
varStrsOut=''.join(fvars)
fmonth=args.fmonth
print 'fmonth:', fmonth
print fvars
print 'Weighted:', weight


# Load empty lists (not very Pythony)
extentObsDt=[] # Detrended (observed) ice extent from linear trend persistence (LTP)
extentPredDt=[] # Detrended (forecast) ice extent from LTP
extentPredAbs=[] # Aboslute forecast of sea ice extent (add LTP)
anoms=[] # Forecast anomaly from observed
perr=[] # Estimated forecast error (1 SD)

# Generate a new forecast each year
for year in xrange(startYearPred, endYear+1, 1):
	if (year<2017):
		vals=ff.CalcForecastMultiVar(year, startYear, fvars, fmonth, 
			region=0, anomObs=1, numYearsReq=numYearsReq, weight=weight, outWeights=outWeights)
		extentObsDt.append(vals[0])
		extentPredDt.append(vals[1])
		extentPredAbs.append(vals[2])
		anoms.append(vals[3])
		perr.append(vals[4])
		print 'Forecast year:', year, fvars, 'SIE forecast:', vals[2]
		
	

rmsF= ff.rms(anoms)
rmsP= ff.rms(extentObsDt)
errorFore = '%.2f' %sqrt(rmsF)
errorExt = '%.2f' %sqrt(rmsP)
skill = '%.2f' %(1 - (rmsF/rmsP))
skill2 = '%.2f' %(1 - (ff.rms(anoms[-9:]))/(ff.rms(extentObsDt[-9:])))
#skill3 = '%.2f' %(1 - (rms(anoms[-16:-8]))/(rms(extentObsDt[-16:-8])))

print '1985-2016 skill:',skill, '2008-2016 skill:', skill2

if (outSkill==1):
	savetxt(skilldatapath+'Skill_'+varStrsOut+str(fmonth)
		+str(startYearPred)+str(endYear)+'W'+str(weight)+'.txt', 
		array([skill, errorFore, errorExt, skill2])[None], 
		header='skill error_forr error_obs skill2', fmt='%s')

if (outLine==1):
	array(extentPredDt).dump(linedatapath+'extentPredDt'
		   +varStrsOut+str(fmonth)+str(startYearPred)+str(endYear)
		   +'W'+str(weight)+'.txt')
	array(extentObsDt).dump(linedatapath+'extentObsDt'
		   +varStrsOut+str(fmonth)+str(startYearPred)+str(endYear)
		   +'W'+str(weight)+'.txt')
	array(extentPredAbs).dump(linedatapath+'extentPredAbs'
		   +varStrsOut+str(fmonth)+str(startYearPred)+str(endYear)
		   +'W'+str(weight)+'.txt')

if (plotSkill==1):
	plot_forecast(varStrsOut)



