#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 15:40:59 2019

@author: sps
"""

##################################################################################
import pandas as pd
from itertools import izip as zip, count
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
input_path='/home/sps/Desktop/radar_data/may_input/'
input_directory = sorted (os.listdir(input_path))
for i1 in range(len(input_directory)):
 input_directory[i1]= input_path+input_directory[i1]
	
	
output_path='/home/sps/Desktop/radar_data/may_output/'
output_directory = sorted (os.listdir(output_path))	
for i1 in range(len(output_directory)):
 output_directory[i1]= output_path+output_directory[i1]

for i2 in range(len(input_directory)):
 a=(glob.glob(input_directory[i2]+"/*uvw"))
 for i in range(len(a)):
	a0=pd.read_csv(a[i]) #for reading entire data
	if a0['Height (km)'][0]>3 and a0['Height (km)'][0]<4:
		a0.to_csv(output_directory[i2]+'/'+a[i][58:]+'.csv',index=False,)#+str(i)+'_')




#####################################################################################################
import pandas as pd
from itertools import izip as zip, count
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
a=sorted(glob.glob("/home/sps/Desktop/radar_data/may_input/17_May/*uvw"))
for i in range(len(a)):
	a0=pd.read_csv(a[i]) #for reading entire data
	if a0['Height (km)'][0]>3 and a0['Height (km)'][0]<4:
		a0.to_csv('/home/sps/Desktop/radar_data/may_output/17_May/'+a[i][58:]+'.csv',index=False,)#+str(i)+'_')
        
################################################################################################################        
import pandas as pd
from itertools import izip as zip, count
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
a=sorted(glob.glob("/home/sps/Desktop/radar_data/may_input/18_May/*uvw"))
for i in range(len(a)):
	a0=pd.read_csv(a[i]) #for reading entire data
	if a0['Height (km)'][0]>3 and a0['Height (km)'][0]<4:
		a0.to_csv('/home/sps/Desktop/radar_data/may_output/18_May/'+a[i][58:]+'.csv',index=False,)        
        
        
        
        
########################################################################################################        
import pandas as pd
from itertools import izip as zip, count
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
a=sorted(glob.glob("/home/sps/Desktop/radar_data/may_input/19_May/*uvw"))
for i in range(len(a)):
	a0=pd.read_csv(a[i]) #for reading entire data
	if a0['Height (km)'][0]>3 and a0['Height (km)'][0]<4:
		a0.to_csv('/home/sps/Desktop/radar_data/may_output/19_May/'+a[i][58:]+'.csv',index=False,)        
        
        
        
 ###################################################################################################       
        
import pandas as pd
from itertools import izip as zip, count
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import csv
from itertools import izip as zip, count
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
a=sorted(glob.glob('/home/sps/Desktop/radar_data/wind/*.csv'))
for i in range(len(a)):
    a0=pd.read_csv(a[i]) #for reading entire data
    b=a0.loc[:,'Zonal(U) (m/s)']
    c=a0.loc[:,'Meridional(V) (m/s)']
    d=a0.loc[:,'Vertical (W) (m/s)']
    b1=pd.DataFrame(b)
    c1=pd.DataFrame(c)
    d1=pd.DataFrame(d)
    b1.head()
    c1.head()
    d1.head()
    b1.columns
    c1.columns
    d1.columns
    k1=b1.rename(columns={'Zonal(U) (m/s)':a[i][34:52]})
    k2=c1.rename(columns={'Meridional(V) (m/s)':a[i][34:52]})
    k3=d1.rename(columns={'Vertical (W) (m/s)':a[i][34:52]})
    k1.to_csv('/home/sps/Desktop/radar_data/wind_product/uwind'+'/'+a[i][34:52]+'.csv',index=False,)
    k2.to_csv('/home/sps/Desktop/radar_data/wind_product/vwind'+'/'+a[i][34:52]+'.csv',index=False,)
    k3.to_csv('/home/sps/Desktop/radar_data/wind_product/w_wind'+'/'+a[i][34:52]+'.csv',index=False,)
    
    
#####################################################################################################################
    


import string
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import csv as csv
#from itertools import izip as zip, count
import numpy as np
import matplotlib.pyplot as plt
a0=pd.read_csv('/home/sps/Desktop/radar_data/wind_f/uwind.csv',header='infer')
reader = csv.DictReader(open('/home/sps/Desktop/radar_data/wind_f/uwind.csv'))
a=reader.fieldnames
ak=a[1:525]
time = [ak.replace('_', ':') for ak in ak]

time = [time.replace('17May2017:', 'May 17 2017 ') for time in time]
time = [time.replace('18May2017:', 'May 18 2017 ') for time in time]
time = [time.replace('19May2017:', 'May 19 2017 ') for time in time]


df=pd.DataFrame(a0)
height=df['Height(km)'][:]
#time1=df.iloc[0:0]
#time1=pd.DataFrame(time

#time = pd.read_csv(fn, parse_dates=a
#[0],
#                          date_parser=lambda x: pd.to_datetime(x, format='%m/%d/%Y %I:%M:%S %p'))

time2=pd.to_datetime((time))

#time3=time2.iloc[:,1:]
#time2=range(1,525,1)
data=df.iloc[:,1:]
#time = np.arange(0.0,746.0, dt)
fig, ax= plt.subplots()
fig = plt.figure(figsize=(14, 4))#plt.pcolormesh(time,height,data,shading='flat',cmap=plt.get_cmap('cool'),vmin=-1.5, vmax =1.5)
cs=plt.pcolormesh(time2,height,data,shading='flat',cmap=plt.cm.jet,vmin =-25, vmax =20)
#cbar = plt.colorbar(cs, shrink=0.5)
#cbar.set_clim(-30, 45)
#plt.xlabel('Date/Time (IST)')
plt.ylabel('Height (km)')

formatter = DateFormatter('%H:%M')
plt.gcf().axes[0].xaxis.set_major_formatter(formatter) 
#plt.xticks(time2)
#plt.xticks(np.arange(1,525,130))
#fig.autofmt_xdate()
plt.ylim(10,20)
plt.colorbar()
plt.title('Zonal(U) (m/s)')
#plt.show()
plt.savefig('/home/sps/Desktop/ursi_images/may_18_uwind2.jpg')    
#########################################################################################################



import string
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import csv as csv
#from itertools import izip as zip, count
import numpy as np
import matplotlib.pyplot as plt
a0=pd.read_csv('/home/sps/Desktop/radar_data/wind_f/vwind.csv',header='infer')
reader = csv.DictReader(open('/home/sps/Desktop/radar_data/wind_f/vwind.csv'))
a=reader.fieldnames
ak=a[1:525]
time = [ak.replace('_', ':') for ak in ak]

time = [time.replace('17May2017:', 'May 17 2017 ') for time in time]
time = [time.replace('18May2017:', 'May 18 2017 ') for time in time]
time = [time.replace('19May2017:', 'May 19 2017 ') for time in time]


df=pd.DataFrame(a0)
height=df['Height(km)'][:]
#time1=df.iloc[0:0]
#time1=pd.DataFrame(time

#time = pd.read_csv(fn, parse_dates=a
#[0],
#                          date_parser=lambda x: pd.to_datetime(x, format='%m/%d/%Y %I:%M:%S %p'))

time2=pd.to_datetime((time))

#time3=time2.iloc[:,1:]
#time2=range(1,525,1)
data=df.iloc[:,1:]
#time = np.arange(0.0,746.0, dt)
fig, ax= plt.subplots()
fig = plt.figure(figsize=(14, 4))#plt.pcolormesh(time,height,data,shading='flat',cmap=plt.get_cmap('cool'),vmin=-1.5, vmax =1.5)
cs=plt.pcolormesh(time2,height,data,shading='flat',cmap=plt.cm.jet,vmin =-25, vmax =20)
#cbar = plt.colorbar(cs, shrink=0.5)
#cbar.set_clim(-30, 45)
#plt.xlabel('Date/Time (IST)')
plt.ylabel('Height (km)')

formatter = DateFormatter('%H:%M')
plt.gcf().axes[0].xaxis.set_major_formatter(formatter) 
#plt.xticks(time2)
#plt.xticks(np.arange(1,525,130))
#fig.autofmt_xdate()
plt.ylim(10,20)
plt.colorbar()
plt.title('Meridional(V) (m/s)')
#plt.show()
plt.savefig('/home/sps/Desktop/ursi_images/vwind2.jpg')  



##################################################################################################
import string
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import csv as csv
#from itertools import izip as zip, count
import numpy as np
import matplotlib.pyplot as plt
a0=pd.read_csv('/home/sps/Desktop/radar_data/wind_f/wwind.csv',header='infer')
reader = csv.DictReader(open('/home/sps/Desktop/radar_data/wind_f/wwind.csv'))
a=reader.fieldnames
ak=a[1:525]
time = [ak.replace('_', ':') for ak in ak]

time = [time.replace('17May2017:', 'May 17 2017 ') for time in time]
time = [time.replace('18May2017:', 'May 18 2017 ') for time in time]
time = [time.replace('19May2017:', 'May 19 2017 ') for time in time]


df=pd.DataFrame(a0)
height=df['Height(km)'][:]
#time1=df.iloc[0:0]
#time1=pd.DataFrame(time

#time = pd.read_csv(fn, parse_dates=a
#[0],
#                          date_parser=lambda x: pd.to_datetime(x, format='%m/%d/%Y %I:%M:%S %p'))

time2=pd.to_datetime((time))

#time3=time2.iloc[:,1:]
#time2=range(1,525,1)
data=df.iloc[:,1:]
#time = np.arange(0.0,746.0, dt)
fig, ax= plt.subplots()
fig = plt.figure(figsize=(14, 4))#plt.pcolormesh(time,height,data,shading='flat',cmap=plt.get_cmap('cool'),vmin=-1.5, vmax =1.5)
cs=plt.pcolormesh(time2,height,data,shading='flat',cmap=plt.cm.jet,vmin =-2, vmax =2)
#cbar = plt.colorbar(cs, shrink=0.5)
#cbar.set_clim(-30, 45)
plt.xlabel('Date/Time (IST)')
plt.ylabel('Height (km)')

formatter = DateFormatter('%H:%M')
plt.gcf().axes[0].xaxis.set_major_formatter(formatter) 
#plt.xticks(time2)
#plt.xticks(np.arange(1,525,130))
#fig.autofmt_xdate()
plt.ylim(10,20)
plt.colorbar()
plt.title('Vertical velocity (W) (m/s)')
#plt.show()
plt.savefig('/home/sps/Desktop/ursi_images/wwind2.jpg')  








##################################################################################################
import string
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import csv as csv
#from itertools import izip as zip, count
import numpy as np
import matplotlib.pyplot as plt

#############################################################################
a1=pd.read_csv('/home/sps/Desktop/radar_data/wind_f/uwind.csv',header='infer')
reader1 = csv.DictReader(open('/home/sps/Desktop/radar_data/wind_f/uwind.csv'))
a=reader1.fieldnames
a2=pd.read_csv('/home/sps/Desktop/radar_data/wind_f/vwind.csv',header='infer')
reader2 = csv.DictReader(open('/home/sps/Desktop/radar_data/wind_f/vwind.csv'))
b=reader2.fieldnames
a3=pd.read_csv('/home/sps/Desktop/radar_data/wind_f/wwind.csv',header='infer')
reader3 = csv.DictReader(open('/home/sps/Desktop/radar_data/wind_f/wwind.csv'))
c=reader3.fieldnames

ak=a[1:525]
time = [ak.replace('_', ':') for ak in ak]
time = [time.replace('17May2017:', 'May 17 2017 ') for time in time]
time = [time.replace('18May2017:', 'May 18 2017 ') for time in time]
time = [time.replace('19May2017:', 'May 19 2017 ') for time in time]


df1=pd.DataFrame(a1)
df2=pd.DataFrame(a2)
df3=pd.DataFrame(a3)
height=df1['Height(km)'][:]
x,y = np.meshgrid(time,height)
#time1=df.iloc[0:0]
#time1=pd.DataFrame(time

#time = pd.read_csv(fn, parse_dates=a
#[0],
#                          date_parser=lambda x: pd.to_datetime(x, format='%m/%d/%Y %I:%M:%S %p'))

time2=pd.to_datetime((time))
x,y = np.meshgrid(np.arange(0, 524, 1),(height))
#x,y = np.meshgrid(time2,height)

#time3=time2.iloc[:,1:]
#time2=range(1,525,1)
data1=df1.iloc[:,1:]
data2=df2.iloc[:,1:]
data3=df3.iloc[:,1:]
uwind=data1
vwind=data2
windspeed = (uwind ** 2 + vwind ** 2) ** 0.5


plt.figure()
fig = plt.figure(figsize=(12, 8))

#plt.subplot(3,1,1)
#plt.pcolormesh(time2,height,data1,shading='flat',cmap=plt.cm.jet,vmin =-25, vmax =20)
#plt.ylim(10,20)
#plt.colorbar()
#plt.title('Zonal(U) (m/s)', fontdict=None, loc='right')
#formatter = DateFormatter('%H:%M')
#plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
#plt.ylabel('Height (km)')


#time = np.arange(0.0,746.0, dt)
#plt.subplot(3,1,2)# create the canvas for plotting
#plt.pcolormesh(time2,height,data2,shading='flat',cmap=plt.cm.jet,vmin =-25, vmax =20)
#plt.ylim(10,20)
#plt.colorbar()
#plt.title('Meridional(V) (m/s)', fontdict=None, loc='right')
#formatter = DateFormatter('%H:%M')
#plt.gcf().axes[2].xaxis.set_major_formatter(formatter)
#plt.ylabel('Height (km)')

 
#ax3 = plt.subplot(3,1,3) 
plt.pcolormesh(x,y,windspeed,shading='flat',cmap=plt.cm.jet,vmin =0, vmax =30)
plt.quiver(x,y,uwind,vwind)
plt.ylim(10,20)
plt.colorbar()
plt.title('windspeed (m/s)', fontdict=None, loc='right')
formatter = DateFormatter('%H:%M')
plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
plt.ylabel('Height (km)')
plt.xlabel('Date/Time (IST)')



#plt.xticks(time2)
#plt.xticks(np.arange(1,525,130))
#fig.autofmt_xdate()
#plt.show()
plt.savefig('/home/sps/Desktop/ursi_images/windmagnitude.png') 


#############################################################################################################################

#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset as ncfile
from cfplot import *
import math

nc = ncfile('/home/sps/Desktop/python/gdata.nc')
lats=nc.variables['lat'][:]
p=nc.variables['p'][:]
temp=nc.variables['temp'][:,:,0]
p0=1013.25
H=8
p1=(p/p0)
p2=np.log(p1)
z=(-8)*p2
print z




fig = plt.figure(figsize=(11, 8))
ax1 = fig.add_subplot(111)

#ax1.axis([-90, 90, 0, 20])
#ax1.tick_params(direction='out', which='both')
ax1.set_xlabel('Latitude (degrees)')
ax1.set_ylabel('height(km')
#ax1.set_yscale('log10')
ax1.set_xticks(np.arange(-90, 120, 30))
ax1.set_yticks([1,4,5,10,15])

ax1.contourf(lats, z, temp, np.arange(-85, 30, 5), extend='both')
cs=ax1.contour(lats, z, temp, np.arange(-85, 30, 5), colors='k')
ax1.clabel(cs, fmt = '%d', colors = 'k')
plt.savefig('ex13.png')

#########################################################################################################################


import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset as ncfile
from cfplot import *

nc = ncfile('gdata.nc')
lats=nc.variables['lat'][:]
p=nc.variables['p'][:]
temp=nc.variables['temp'][:,:,0]

fig = plt.figure(figsize=(11, 8))
ax1 = fig.add_subplot(111)

ax1.axis([-90, 90, 1000, 0.316])
ax1.tick_params(direction='out', which='both')
ax1.set_xlabel('Latitude (degrees)')
ax1.set_ylabel('Pressure (mb)')
ax1.set_yscale('log')
ax1.set_xticks(np.arange(-90, 120, 30))
ax1.set_yticks([1000,100,10,1])

ax1.contourf(lats, p, temp, np.arange(-85, 30, 5), extend='both')
cs=ax1.contour(lats, p, temp, np.arange(-85, 30, 5), colors='k')
ax1.clabel(cs, fmt = '%d', colors = 'k')

ax2 = ax1.twinx()
ax2.tick_params(direction='out', which='both')
ax2.axis([-90, 90, 0, 56.78])
ax2.set_ylabel('Height (km)')
ax2.set_yticks(np.arange(0, 60, 10))


plt.savefig('ex14.png')

##########################################################################################################################

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset as ncfile
from cfplot import *
import math
from datetime import date
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
nc = ncfile('/home/sps/Desktop/merra/merra.nc')
#a0=pd.read_csv('/home/sps/Desktop/time')
a0=pd.read_csv('/home/sps/Desktop/time.csv',header='infer')
reader = csv.DictReader(open('/home/sps/Desktop/time.csv'))
a=reader.fieldnames
time=a[1:25]
lats=nc.variables['lat'][:]
lons=nc.variables['lon'][:]
p=nc.variables['lev'][:]
#time=nc.variables['time'][:]
data=nc.variables['RH'][:,:,27,50]
data1=data.T
#time1=pd.DataFrame(time)
#time2=pd.to_datetime((time))
p0=1013.25
p1=(p/p0)
p2=np.log(p1)
z=(-7.5)*p2
print z
fig = plt.figure(figsize=(11, 4))
#ax1 = fig.add_subplot(111)
#time1 = date.fromtimestamp([time])

#ax1.tick_params(direction='out', which='both')
ax1.set_xlabel('Latitude (degrees)')
ax1.set_ylabel('height(km')
#ax1.set_yscale('log10')
#ax1.set_xticks([])
#ax1.set_yticks([2,4,6,8,10,12,14,16,18])

cs=plt.contourf(time2,z, data1,shading='flat',cmap=plt.get_cmap('cool'))
#formatter = DateFormatter('%H:%M')
#plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
plt.ylim(0,20)
#plt.ylim(10,20)
plt.colorbar()
#cs=ax1.contour(lats, z, temp, np.arange(-85, 30, 5), colors='k')
#ax1.clabel(cs, fmt = '%d', colors = 'k')
plt.savefig('ex13.png')


#########################################################################################
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset as ncfile
from cfplot import *
import math
from datetime import date
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
nc = ncfile('/media/sps/682c98d2-7de2-4dd0-9e71-0ab80b05b2b0/home/Ph.D/merra data/may15_21 (2)/merraf.nc')
a0=pd.read_csv('/home/sps/Desktop/time.csv')
a0=pd.read_csv('/home/sps/Desktop/time.csv',header='infer')
reader = csv.DictReader(open('/home/sps/Desktop/time.csv'))
a=reader.fieldnames
time1=a[1:49]
lats=nc.variables['lat'][:]
lons=nc.variables['lon'][:]
p=nc.variables['lev'][:]
time=nc.variables['time'][:]
datar=nc.variables['RH'][:,:,27,50]
data1=datar.T
datac=nc.variables['QL'][:,:,27,50]
data2=datac.T
datai=nc.variables['QI'][:,:,27,50]
data3=datai.T

#time1=pd.DataFrame(time)
time2=pd.to_datetime((time1))
p0=1013.25
p1=(p/p0)
p2=np.log(p1)
z=(-7.5)*p2
print z
##########################################################################################
fig = plt.figure(figsize=(14, 4))

ax1 = fig.add_subplot(111)
#time1 = date.fromtimestamp([time])
#fig, ax= plt.subplots()
#ax1.tick_params(direction='out', which='both')
#ax1.set_xlabel('Latitude (degrees)')
#ax1.set_ylabel('height(km')
#ax1.set_yscale('log10')
#ax1.set_xticks([04,16,04,16,04,16])
ax1.set_yticks([2,4,6,8,10,12,14,16,18])
############################################################################################
cs=plt.contourf(time2,z, data1,shading='flat',cmap=plt.get_cmap('cool'))
#formatter = DateFormatter('%H:%M')
formatter = DateFormatter('%d')
plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
#formatter = DateFormatter('%H:%M')
#plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
plt.ylim(0,20)
#plt.xlim(17,20)
plt.colorbar()
plt.ylabel('Height (km)')
plt.xlabel('Date/Time (UTC)')
plt.title('relative_humidity_after_moist', fontdict=None, loc='right')
#cs=ax1.contour(lats, z, temp, np.arange(-85, 30, 5), colors='k')
#ax1.clabel(cs, fmt = '%d', colors = 'k')
plt.savefig('/home/sps/Desktop/ursi_images/rh.jpg')


#############################################################################################################################
######panel plot######################

###############################################################################################################
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset as ncfile
from cfplot import *
import math
from datetime import date
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
nc = ncfile('/media/sps/682c98d2-7de2-4dd0-9e71-0ab80b05b2b0/home/Ph.D/merra data/may15_21 (2)/merraf.nc')
#a0=pd.read_csv('/home/sps/Desktop/time.csv')
a0=pd.read_csv('/home/sps/Desktop/time.csv',header='infer')
reader = csv.DictReader(open('/home/sps/Desktop/time.csv'))
a=reader.fieldnames
time1=a[1:49]
lats=nc.variables['lat'][:]
lons=nc.variables['lon'][:]
p=nc.variables['lev'][:]
time=nc.variables['time'][:]
datar=nc.variables['RH'][:,:,27,50]
data1=datar.T
datac=nc.variables['QL'][:,:,27,50]
data2=datac.T
datai=nc.variables['QI'][:,:,27,50]
data3=datai.T

#time1=pd.DataFrame(time)
time2=pd.to_datetime((time1))
p0=1013.25
p1=(p/p0)
p2=np.log(p1)
z=(-7.5)*p2
print z
plt.figure()
fig = plt.figure(figsize=(12, 10))

plt.subplot(3,1,1)
plt.contourf(time2,z,data1,shading='gouraud',cmap=plt.get_cmap('cool'))#,vmin =-25, vmax =20)
plt.ylim(0,20)
plt.colorbar()
plt.title('relative_humidity_after_moist', fontdict=None, loc='right')
formatter = DateFormatter('%d')
plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
plt.ylabel('Height (km)')


#time = np.arange(0.0,746.0, dt)
plt.subplot(3,1,2)# create the canvas for plotting
plt.contourf(time2,z,data2,shading='gouraud',cmap=plt.get_cmap('cool'))#,vmin =-25, vmax =20)
plt.ylim(0,20)
plt.colorbar()
plt.title('mass_fraction_of_cloud_liquid_water (kg kg-1)', fontdict=None, loc='right')
formatter = DateFormatter('%d')
plt.gcf().axes[2].xaxis.set_major_formatter(formatter)
plt.ylabel('Height (km)')

 
ax3 = plt.subplot(3,1,3) 
plt.contourf(time2,z,data3,shading='gouraud',cmap=plt.get_cmap('cool'))#,vmin =-2, vmax =2)
plt.ylim(0,20)
plt.colorbar()
plt.title('mass_fraction_of_cloud_ice_water (kg kg-1)', fontdict=None, loc='right')
formatter = DateFormatter('%d')
plt.gcf().axes[4].xaxis.set_major_formatter(formatter)
plt.ylabel('Height (km)')
plt.xlabel('Date/Time (IST)')

plt.savefig('/home/sps/Desktop/ursi_images/rha.jpg')





import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset as ncfile
import math
import datetime as dt
from datetime import date
from datetime import datetime
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
ds = xr.open_dataset('/home/sps/Desktop/data]/merra.nc')
 
#a0=pd.read_csv('/home/sps/Desktop/time.csv')
#a0=pd.read_csv('/home/sps/Desktop/time.csv',header='infer')
#reader = csv.DictReader(open('/home/sps/Desktop/time.csv'))
#a=reader.fieldnames
#time1=a[1:49]
lats=nc.variables['lat'][:]
lons=nc.variables['lon'][:]
p=nc.variables['lev'][:]
time=nc.variables['time'][:]
datar=nc.variables['RH'][:,:,27,50]
data1=datar.T
datac=nc.variables['QL'][:,:,27,50]
data2=datac.T
datai=nc.variables['QI'][:,:,27,50]
data3=datai.T






#time_idx = 2  # some random day in 2012
# Python and the renalaysis are slightly off in time so this fixes that problem
#offset = dt.timedelta(hours=3)
# List of all times in the file as datetime objects
#dt_time = [dt.date(1, 1, 1) + dt.timedelta(hours=t) - offset\
#           for t in time]
#cur_time = dt_time[time_idx]



#time1=pd.DataFrame(time)
#time2=pd.to_datetime((time1))
p0=1013.25
p1=(p/p0)
p2=np.log(p1)
z=(-7.5)*p2
print z
plt.figure()
fig = plt.figure(figsize=(12, 10))

plt.subplot(3,1,1)
plt.contourf(time2,z,data1,shading='gouraud',cmap=plt.get_cmap('cool'))#,vmin =-25, vmax =20)
plt.ylim(0,20)
plt.colorbar()
plt.title('relative_humidity_after_moist', fontdict=None, loc='right')
formatter = DateFormatter('%d')
plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
plt.ylabel('Height (km)')


#time = np.arange(0.0,746.0, dt)
plt.subplot(3,1,2)# create the canvas for plotting
plt.contourf(time2,z,data2,shading='gouraud',cmap=plt.get_cmap('cool'))#,vmin =-25, vmax =20)
plt.ylim(0,20)
plt.colorbar()
plt.title('mass_fraction_of_cloud_liquid_water (kg kg-1)', fontdict=None, loc='right')
formatter = DateFormatter('%d')
plt.gcf().axes[2].xaxis.set_major_formatter(formatter)
plt.ylabel('Height (km)')

 
ax3 = plt.subplot(3,1,3) 
plt.contourf(time2,z,data3,shading='gouraud',cmap=plt.get_cmap('cool'))#,vmin =-2, vmax =2)
plt.ylim(0,20)
plt.colorbar()
plt.title('mass_fraction_of_cloud_ice_water (kg kg-1)', fontdict=None, loc='right')
formatter = DateFormatter('%d')
plt.gcf().axes[4].xaxis.set_major_formatter(formatter)
plt.ylabel('Height (km)')
plt.xlabel('Date/Time (IST)')

plt.savefig('/home/sps/Desktop/ursi_images/rha.jpg')
































