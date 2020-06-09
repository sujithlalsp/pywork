#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 15:40:59 2019

@author: sps
"""

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
a=sorted(glob.glob('/home/sps/Desktop/radar_data/moments/19_may/*.csv'))
for i in range(len(a)):
	a0=pd.read_csv(a[i]) #for reading entire data
	b=a0.loc[:,'M1 Mean Doppler (Hz)']
	print b
	c=pd.DataFrame(b)
	c.head()
	c.columns
	k=c.rename(columns={'M1 Mean Doppler (Hz)':a[i][44:62]})
	print k
	k.to_csv('/home/sps/Desktop/radar_data/moments/doppler'+'/'+a[i][44:62]+'.csv',index=False,)
    
    
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
a0=pd.read_csv('/home/sps/Desktop/work/snr.csv',header='infer')
reader = csv.DictReader(open('/home/sps/Desktop/work/snr.csv'))
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
cs=plt.contourf(time2,height,data,shading='flat',cmap=plt.cm.jet,vmin =-30, vmax =50)
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
plt.title('Signal-to-noise ratio(dB)')
#plt.show()
plt.savefig('/home/sps/Desktop/ursi_images/may_18_SNR_new7.jpg')    






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
a=sorted(glob.glob('/home/sps/Desktop/radar_data/moments/doppler/*.csv'))
combined_csv = pd.concat([pd.read_csv(f) for f in a ],axis=1)
combined_csv.to_csv( '/home/sps/Desktop/radar_data/moments/combined_doppler.csv', index=False)
