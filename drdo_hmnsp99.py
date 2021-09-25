import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import numpy as np
from numpy import gradient
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from netCDF4 import Dataset
from wrf import to_np, getvar, CoordPair, vertcross ,latlon_coords , xy, interp2dxy, ALL_TIMES,get_basemap,vinterp
ncfile =Dataset('/home/sps/Documents/DRDO_PROJECT/data/wrfout_d03_2020-12-02_00_00_00')

tym = getvar(ncfile,"times",timeidx=ALL_TIMES,)
p= getvar(ncfile,"p",units="hPa",timeidx=ALL_TIMES,)
t= getvar(ncfile,"tk",timeidx=ALL_TIMES,)
u= getvar(ncfile,"ua",units="m s-1",timeidx=ALL_TIMES,)
v= getvar(ncfile,"va",units="m s-1",timeidx=ALL_TIMES,)

#interpolate data (0-10km with 100m resolution)

h=np.arange(0,10,0.1)

pi=vinterp(ncfile,p, vert_coord='ght_msl', interp_levels=h, extrapolate=False, field_type=None, log_p=False, timeidx=ALL_TIMES,  squeeze=True, cache=None, meta=True)

T=vinterp(ncfile,t, vert_coord='ght_msl', interp_levels=h, extrapolate=False, field_type=None, log_p=False, timeidx=ALL_TIMES,  squeeze=True, cache=None, meta=True)

ui=vinterp(ncfile,u, vert_coord='ght_msl', interp_levels=h, extrapolate=False, field_type=None, log_p=False, timeidx=ALL_TIMES,  squeeze=True, cache=None, meta=True)

vi=vinterp(ncfile,v, vert_coord='ght_msl', interp_levels=h, extrapolate=False, field_type=None, log_p=False, timeidx=ALL_TIMES,  squeeze=True, cache=None, meta=True)

#subset data (a particular lat,lon and time)
p=pi[0,:,0,0]

T=T[0,:,0,0]

u=ui[0,:,0,0]
v=vi[0,:,0,0]

#compute refractive index structure parameter based on HMNSP99 MODEL
h=h*1000
TT=T**2
k=(p/TT)
k=k*((7.9)* 10**(-5))
k=k**2
theta=((1000/p)**0.286)*T
dthetadh= gradient(theta)/gradient(h)
dudh= gradient(u)/gradient(h)
dvdh=gradient(v)/gradient(h)
dTdh = gradient(T)/gradient(h)
roots=((dudh)**2)+((dvdh)**2)
s=roots**(1/2)
l=((0.1)**(4/3)) * ((10)**((0.362)+ (16.728*s)-(192.347*dTdh)))

PT=(dthetadh)**2

CT=(2.8)*(l)*(PT)

CN=CT*k

#figure
fig, ax = plt.subplots()

ax.set_title('Refractive index\n structure parameter (9.486N,75.17E,HMNSP99 model)')
ax.set_xlabel('$\mathregular{Cn^{2}}$')
ax.set_ylabel('Height (m)')

plt.plot(CN,h,'r--')




