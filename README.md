# Arctic sea ice forecasting
by Alek Petty

Python scripts used to produce seasonal forecasts of Arctic (and Alaskan) sea ice extent.

Petty, A. A., D. Schroder, J. C. Stroeve, T. Markus, J. Miller, N. T. Kurtz, D. L. Feltham, D. Flocco (2017), Skillful spring forecasts of September Arctic sea-ice extent using passive microwave sea ice observations, Earth’s Future, 4 , doi:10.1002/2016EF000495.

## Scripts

Individual descriptions should be included at the top of each script. Not all processing/plotting scripts have been included yet.

Python 2.7 was used for all processing. I have not tested these scripts in Python 3.

I use Conda to intall/manage the various Python packages. Check out the file 'packages.txt' for a list of the Python package versions I used to run these Scripts. I should probably do this in a conda environment and output that information at some point.

Information about installing Conda/Python, and a brief introduction to using Python can be found on my NASA Cryospheric Sciences meetup repo: https://github.com/akpetty/cryoscripts.

## Data

The gridded forecast datasets were generated from the following, pubclically available datasets:

Sea ice concentration data (final): http://nsidc.org/data/nsidc-0051 
Sea ice concentration data (near real-time): https://nsidc.org/data/nsidc-0081
Melt onset data: http://neptune.gsfc.nasa.gov/csb/index.php?section=54
(note that the melt onset data are not made avilable each year near real-time, so contact me if required).

Simulated melt pond data were provided by CPOM-Reading, with the detrended forecast data included in this repo.

Contact me if you any any questions!

Alek

