import matplotlib
matplotlib.use("AGG")
from mpl_toolkits.basemap import Basemap, shiftgrid
import numpy as np
from pylab import *
import numpy.ma as ma
from glob import glob
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
#from netCDF4 import Dataset


rawdatapath='../Data/'
datapath='../DataOutput/'

skilldatapath='../DataOutput/SkillVals/'
linedatapath='../DataOutput/TimeSeries/'
weightdatapath='../DataOutput/Weights/'
extdatapath = rawdatapath+'/IceExtent/'

def rms(var):
	"""calculate th root mean square of a given list """
	return sum([x**2 for x in var])/size(var)
                                                              
def get_detrended_yr(YearsT, var_yearsT, var_yrT, num_years_req):
	"""Detrend a 2D array using linear regression

       Mask based on valid number of years in each grid cell.
    """
	var_yearsDT=ma.masked_all((var_yearsT.shape))
	var_yrDT=ma.masked_all((var_yrT.shape))

	# Loop over each dimension
	for i in xrange(var_yearsT.shape[1]):
		for j in xrange(var_yearsT.shape[2]):
			mask=~var_yearsT[:, i, j].mask
			var_yearsT_ma = var_yearsT[:, i, j][mask]	
				
			if (len(var_yearsT_ma)>num_years_req):
				trendT, interceptT, r_valsT, probT, stderrT = stats.linregress(YearsT[mask],var_yearsT_ma)
				lineT = (trendT*YearsT) + interceptT
				var_yearsDT[:, i, j]=var_yearsT[:, i, j]-lineT
				
				# Calculate the detrended var (linear trend persistence) fo the given forecast year
				lineT_yr=interceptT + (trendT*(YearsT[-1]+1))
				var_yrDT[i, j]=var_yrT[i, j]-lineT_yr

	return var_yearsDT, var_yrDT


def plot_forecast(outVarStr):
	"""Plot forecast data """
    
	fig = figure(figsize=(3.5,2.2))
	ax1=subplot(2, 1, 1)
	im1 = plot(years, extent, 'k')
	#im2 = plot(Years[start_year_pred-start_year:], lineT[start_year_pred-start_year:]+ExtentG, 'r')
	im3 = plot(yearsP, extentPredAbs, 'r')
	#errorbar(YearsP, array(lineTP)+array(ExtentG) , yerr=prederror, color='r',fmt='',linestyle='',lw=0.4,capsize=0.5, zorder = 2)
	ax1.errorbar(yearsP, extentPredAbs , yerr=perr, color='r',fmt='',linestyle='',lw=0.6,capsize=0.5, zorder = 2)
	ax1.errorbar(yearsP, extentPredAbs , yerr=[1.96*x for x in perr], color='r',fmt='',linestyle='',lw=0.3,capsize=0.5, zorder = 2)

	ax1.set_ylabel(r'Extent (M km$^2$)')
	ax1.set_xlim(1978, 2017)
	ax1.set_xticks(np.arange(1980, 2018, 5))
	ax1.set_xticklabels([])
	#ylim(3, 9)

	ax2=subplot(2, 1, 2)
	ax2.yaxis.tick_right()
	ax2.yaxis.set_label_position("right")
	im21 = plot(yearsP[0:-1], extentObsDt, 'k')
	im3 = plot(yearsP, extentPredDt, 'r')
	ax2.errorbar(yearsP, extentPredDt , yerr=perr, color='r',fmt='',linestyle='',lw=0.6,capsize=0.5, zorder = 2)
	ax2.errorbar(yearsP, extentPredDt , yerr=[1.96*x for x in perr], color='r',fmt='',linestyle='',lw=0.3,capsize=0.5, zorder = 2)


	ax2.set_ylabel(r'Extent anomaly (M km$^2$)', rotation=270, labelpad=10)
	ax2.set_xlabel('Years')
	ax2.set_yticks([-2, -1, 0, 1, 2])
	ax2.set_xlim(1978, 2017)
	ax2.set_xticks(np.arange(1980, 2018, 5))
	ax2.axhline(0, linewidth=0.5,linestyle='--', color='k')
	ax2.annotate(r'$\sigma_{ferr}$='+errorFore+r' M km$^2$'+', S:'+skill, 
		xy=(0.03, 0.04), xycoords='axes fraction', horizontalalignment='left', verticalalignment='bottom')

	subplots_adjust(left=0.1, right=0.90, bottom=0.17, top=0.96, hspace=0)

	savefig(figpath+'/forecast'+str(startYear)+str(endYear)+'M'+str(fmonth)+outVarStr+'.pdf', dpi=300)
	close(fig)



def get_correlation_coeffs(var_yearsT, ExtentDT, num_years_req):
	""" Calculate the correlation coeficients between the detrended forecast
		variable and detrended ice extent
    """
	r_valsDT=np.zeros((var_yearsT.shape[1], var_yearsT.shape[2]))
	for i in xrange(var_yearsT.shape[1]):
		for j in xrange(var_yearsT.shape[2]):
			mask=~var_yearsT[:, i, j].mask
			var_yearsT_ma = var_yearsT[:, i, j][mask]
			if (len(var_yearsT_ma)>num_years_req):
				trendDT, interceptDT, r_valsDT[i, j], probDT, stderrDT = stats.linregress(ExtentDT[mask],var_yearsT_ma)
	return r_valsDT

def GetWeightedPredVar(yearsT, extentDTT, predvarYrsT, predvar_yrT, varT, fmonth, startYear, numYearsReq, normalize=0, rneg=0, rpos=1, absr=1, weight=1, outWeights=0):
	""" Get forecast data and weight using historical correlation if selected
    """
	# Get detrended 2D forecast data
	predvarYrsDT, predvarYrDT = get_detrended_yr(yearsT, predvarYrsT, predvar_yrT, numYearsReq)

	# Correlate detrended time series
	rvalsDT = get_correlation_coeffs(predvarYrsDT, extentDTT, numYearsReq)
	
	if (rneg==1):
		# Set positive R-vals to zero (assumed to be unphysical)
		rvalsDT[where(rvalsDT>0)]=0
	if (rpos==1):
		# Set negative R-vals to zero (assumed to be unphysical)
		rvalsDT[where(rvalsDT<0)]=0
	if (absr==1):
		# Use absolute values of correlation coefficeint
		rvalsDT=abs(rvalsDT)
	if (weight==0):
		print 'No weighting applied!'
		rvalsDT=np.ones((rvalsDT.shape))

	if (outWeights==1):
		rvalsDT.dump(weightdatapath+'rvalsDT'+varT+str(fmonth)+str(startYear)+str(yearsT[-1]+1)+'.txt')
		predvarYrDT.dump(weightdatapath+'predvarYrDT'+varT+str(fmonth)+str(startYear)+str(yearsT[-1]+1)+'.txt')
	
	# Calculated weighted forcast data
	weightedPredvar=[]
	for x in xrange(predvarYrsDT.shape[0]):
		weightedPredvar.append(ma.mean(rvalsDT*predvarYrsDT[x]))
	
	weightedPredvarYr = ma.mean(rvalsDT*predvarYrDT)
	
	if (normalize==1):
		# Normalize data (doesn't chagne single var forecasting, may be important for multivar)
		weightedPredvarN=(weightedPredvar-min(weightedPredvar))/(max(weightedPredvar)-min(weightedPredvar))
		weightedPredvarYrN=(weightedPredvarYr-min(weightedPredvar))/(max(weightedPredvar)-min(weightedPredvar))
		return weightedPredvarN, weightedPredvarYrN
	else:
		return weightedPredvar, weightedPredvarYr


def get_varDT(Years, Extent):
	""" Detrend linear time series  """
	trendT, interceptT, r_valsT, probT, stderrT = stats.linregress(Years,Extent)
	lineT = (trendT*Years) + interceptT
	ExtentDT=Extent-lineT
	return ExtentDT, lineT

def get_ice_extentN(rawdatapath, Month, start_year, end_year, icetype='extent', version=''):
	""" Get Arctic sea ice extent

	Data downlaoded from the NSIDC Arctic Sea Ice Index.

	Can also get ice area if icetype set to 'area', 
	   but beware of variable pole hole contaminating Arctic data

	"""
	Month_str = '%02d' %Month
	extent_data_path=rawdatapath+'IceExtent/N_'+Month_str+'_area'+version+'.txt'
	ice_extent_data=pd.read_csv(extent_data_path, delim_whitespace=True,header=(0),index_col=False)
	#ice_extent_data = np.loadtxt(extent_data_path, delimiter=',',skiprows=1)
	Extent = ice_extent_data[icetype]
	Year = ice_extent_data['year']
	
	# Choose a given subset of extent data
	Years=array(Year[start_year-1979:end_year-1979+1])
	Extent=array(Extent[start_year-1979:end_year-1979+1])

	return Years, Extent

def CalcForecastMultiVar(yearT, startYear, predvarYrs, fmonth, region=0, anomObs=1, month=9, outWeights=0, icetype='extent', numYearsReq=5, weight=1):
	""" The primary sea ice forecast function. 

	NB: This should probably be a class but hey. 

	"""
	# This should probably be a class but hey

	# Get ice extent then detrend.
	if (region==0): 
	# If Arctic
		yearsT, extentT = get_ice_extentN(rawdatapath, month, startYear, yearT-1, icetype=icetype, version='v2')
		extentDT, lineT=get_varDT(yearsT, extentT)
		
		if (anomObs==1):
			# If we have observed sea ice extent data for the given forecast year to check forecast skill.
			years2, extent2 = get_ice_extentN(rawdatapath, month, startYear, yearT, icetype=icetype, version='v2')
			extentyr=extent2[-1]
	else: 
	# If generating a regional forecast
		extentALL=loadtxt(extdatapath+'ice_extent_M9R'+str(region)+'_19792016')
		
		#get years and extent for years preceeding the given forecast year
		yearsT=np.arange(startYear, yearT, 1)
		extentT=extentALL[0:yearT-startYear]
		extentDT, lineT=get_varDT(yearsT, extentT)
		if (anomObs==1):
			extentyr=extentALL[yearT-startYear]
	
	# Fill with ones to act as the intercept
	predVarsTYr=[1]
	predVars=np.ones((size(yearsT)))

	# For melt pond forecast
	if (fmonth>=5):
		# June
		pdate='56'
	else:
		# May
		pdate='31'
	
	for varT in predvarYrs:
		#print 'Var:', varT
		if (varT in ['sst','conc','melt','melt_nan', 'pmas']):
			VarYearsT = get_gridvar(varT, fmonth, startYear, yearT)
			predVarT, predVarTYr = GetWeightedPredVar(yearsT, extentDT, VarYearsT[0:-1], VarYearsT[-1],varT, fmonth, startYear,numYearsReq, normalize=0, outWeights=outWeights, weight=weight)
		
		predVarsTYr.append(predVarTYr)
		predVars=np.column_stack((predVars, array(predVarT)))
	
	# Use SM to generate the regression model. Could have just used linregress (tested, gave same results, this was just a bit neater)
	model=sm.OLS(extentDT, predVars)
	fit=model.fit()

	# Forecast detrended sea ice extent!
	extentForrDT = fit.predict(predVarsTYr)[0]
	# Prediction uncertainty estimate
	prstd, iv_l, iv_u = wls_prediction_std(fit, exog=predVarsTYr)

	# Ice extent assuming inear trend persistnce
	extTrendP=(lineT[-1]+(lineT[-1]-lineT[-2]))

	extentForrAbs = extentForrDT+extTrendP
	
	if (anomObs==1):
		extentObsDT=extentyr-extTrendP
		anom=extentyr-extentForrAbs
		return  extentObsDT, extentForrDT, extentForrAbs, anom, prstd[0]
	else:
		
		return  extentForrDT, extentForrAbs, prstd[0]

def get_conc_gridded(dataoutpath, start_year, end_year, month):
	""" Get gridded ice concentration data

	Data gridded using linear interpolation of NASA Team concentration data onto a 100 km grid.
	Used monthly data, then monthly means of the daily NRT data for 2015 onwards.


	"""
	xpts=load(dataoutpath+'xpts100km')
	ypts=load(dataoutpath+'ypts100km')
	conc_years=ma.masked_all((end_year-start_year+1,xpts.shape[0], xpts.shape[1]))
	for year in xrange(start_year, end_year+1, 1):
		conc_years[year-start_year] = load(dataoutpath+'ice_conc100km'+str(month)+str(year))

	return xpts, ypts, conc_years

def get_meltonset_gridded(dataoutpath, start_year, end_year, freezemelt_str):
	""" Get gridded melt onset data

	Data gridded using linear interpolation of NASA's GSFC melt onset data onto a 100 km grid.

	"""
	xpts=load(dataoutpath+'xpts100km')
	ypts=load(dataoutpath+'ypts100km')
	Melt_onset_years=ma.masked_all((end_year-start_year+1,xpts.shape[0], xpts.shape[1]))
	for year in xrange(start_year, end_year+1, 1):
		Melt_onset_years[year-start_year] = load(dataoutpath+freezemelt_str+'100km'+str(year))

	return xpts, ypts, Melt_onset_years


def get_gridvar(fvar, fmonth, startYearT, endYearT):
	""" Select which gridded forecast dataset to use in forecast

	NB pond data left out for now.

	"""
	if (fvar=='conc'):
		dataoutpath=rawdatapath+'/IceConc/'
		xpts, ypts, VarYears=get_conc_gridded(dataoutpath, startYearT, endYearT, fmonth)
		#rneg=0
		#rpos=1
	if ((fvar=='melt')|(fvar=='melt_nan')):
		meltdays=[31, 59, 90, 120, 151, 181, 212, 243]
		print
		meltday=meltdays[fmonth]
		dataoutpath=rawdatapath+'/MeltOnset/'
		xpts, ypts, VarYears=get_onset_gridded(dataoutpath, startYearT, endYearT, fvar)
		# Express melt onset relative to the given forecast date (end of the forecast month)
		VarYears=meltday-VarYears
		VarYears[where(VarYears<0)]=0
		#reverse to make consistent with concentration - i.e. low vals lead to low ice extent
		VarYears=-VarYears
	return VarYears

