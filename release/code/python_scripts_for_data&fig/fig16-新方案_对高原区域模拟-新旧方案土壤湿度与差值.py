""""
@author & attributor: guxinyi
"""

import xarray as xr
import numpy as np
import pandas as pd
import shapefile
import cmaps

import scipy.stats as stats
from scipy.stats import pearsonr
from scipy.stats import t

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.shapereader import Reader
from cftime import DatetimeNoLeap

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
import matplotlib.colors as mcolors
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.colors as mpl_colors
import matplotlib.patches as patches
from matplotlib.ticker import FormatStrFormatter, FixedLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec

def load_and_process_humidity_data(nc_path, time_slice, shallow_slice, deep_slice):
    # 打开NetCDF文件
    nc_data = xr.open_dataset(nc_path)

    # 处理浅层湿度数据
    humid_shallow = nc_data['SOILLIQ'].sel(time=time_slice).rename({'levsoi': 'levgrnd'}).isel(levgrnd=shallow_slice)
    dzsoi_shallow = nc_data['DZSOI'].isel(levgrnd=shallow_slice)
    humid_shallow = humid_shallow / dzsoi_shallow / 1000.0
    humid_shallow_mean = humid_shallow.mean(dim='time').mean(dim='levgrnd')

    # 处理深层湿度数据
    humid_deep = nc_data['SOILLIQ'].sel(time=time_slice).rename({'levsoi': 'levgrnd'}).isel(levgrnd=deep_slice)
    dzsoi_deep = nc_data['DZSOI'].isel(levgrnd=deep_slice)
    humid_deep = humid_deep / dzsoi_deep / 1000.0
    humid_deep_mean = humid_deep.mean(dim='time').mean(dim='levgrnd')

    # 处理最小湿度数据
    humid_min = nc_data['SOILLIQ'].sel(time=time_slice).rename({'levsoi': 'levgrnd'}).isel(levgrnd=slice(0, 20))
    dzsoi_min = nc_data['DZSOI'].isel(levgrnd=slice(0, 20))
    humid_min = humid_min / dzsoi_min / 1000.0
    humid_min_val = humid_min.mean(dim='levgrnd').mean(dim='time')

    return humid_shallow_mean, humid_deep_mean, humid_min_val

def shp2clip(originfig, ax, shpfile, fieldVals):
        sf = shapefile.Reader(shpfile)
        vertices = []
        codes = []

        pts = sf.shapes()[0].points
        prt = list(sf.shapes()[0].parts) + [len(pts)]
        for i in range(len(prt) - 1):
                for j in range(prt[i], prt[i+1]):
                        vertices.append((pts[j][0], pts[j][1]))
                codes += [Path.MOVETO]
                codes += [Path.LINETO] * (prt[i+1] - prt[i] -2)
                codes += [Path.CLOSEPOLY]
        clip = Path(vertices, codes)
        clip = PathPatch(clip, transform=ax.transData)

        for contour in originfig.collections:
                contour.set_clip_path(clip)

        for line in ax.lines:
                line.set_clip_path(clip)
        return clip

def plot_map_set(ax, lon, lat, data, levels, title, shapefile_path, cb_ticks, cmap, clip_shape=False, clip_args=None):
        ax.set_extent([75.2, 105.5, 25.0, 42.3], crs=ccrs.PlateCarree())
        ax.set_xticks(list(range(74, 107, 8)), crs=ccrs.PlateCarree())
        ax.set_yticks(list(range(24, 41, 4)), crs=ccrs.PlateCarree())
        ax.tick_params(labelcolor='k', length=5, width=3, labelsize=18)

        lon_formatter = LongitudeFormatter(zero_direction_label=False)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.spines['geo'].set_linewidth(3)

        reader = Reader(shapefile_path)
        geoms = reader.geometries()
        ax.add_geometries(geoms, ccrs.PlateCarree(), lw=2, fc='none')

        norm = mpl.colors.BoundaryNorm(levels, cmap.N, extend='both')
        cn = ax.contourf(lon, lat, data, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, levels=levels, extend='both')

        ax.set_title(title, fontsize=20, loc='left')

        if clip_shape and clip_args is not None:
                clip = shp2clip(cn, ax, shapefile_path, clip_args)

        cax = inset_axes(ax, width="100%", height="9%", loc='lower center',
                        bbox_to_anchor=(0.0, -0.3, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        cb = plt.colorbar(cn, cax=cax, orientation='horizontal', ticks=cb_ticks)
        cb.ax.tick_params(labelsize=18, direction='out')
        cb.outline.set_linewidth(1.5)
        cax.xaxis.set_ticks_position('bottom')
        cb.ax.text(1.0, -0.88, 'm³/m³', fontsize=18, ha='left', va='center', transform=cb.ax.transAxes)

ft_slice_time = slice("2016-10-01", "2017-06-01")
shallow_slice = slice(0, 3)
deep_slice = slice(8, 11)

original_path = '/Volumes/Finn‘sT7/Data/博士资料/Development_and_Validation_of_a_New_Soil_frost_heave_shceme/CTSM-Regional/after_modified_surfdata/150x325_TP_xpf.clm2.h0.2016-06-01-00000.nc'
modified_path = '/Volumes/Finn‘sT7/Data/博士资料/Development_and_Validation_of_a_New_Soil_frost_heave_shceme/CTSM-Regional/after_modified_surfdata/150x325_TP_xpf_ft.clm2_SOILLIQ.h0.2016-06-01-00000.nc'

humid_original_shallow, humid_original_deep, humid_original_min = load_and_process_humidity_data(original_path, ft_slice_time, shallow_slice, deep_slice)
humid_modified_shallow, humid_modified_deep, humid_modified_min = load_and_process_humidity_data(modified_path, ft_slice_time, shallow_slice, deep_slice)

humid_df_shallow = humid_modified_shallow - humid_original_shallow
humid_df_deep = humid_modified_deep - humid_original_deep
humid_df_min = humid_modified_min - humid_original_min

lon = humid_original_shallow.lon.values
lat = humid_original_shallow.lat.values

shapefile_path = '/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/老师学生资料相关/盛丹睿/气候区/tibet_shp/qingz(bnd)(geo).shp'

fig = plt.figure(figsize=(21, 15))
gs = GridSpec(3, 3, figure=fig, wspace=0.3, hspace=0.3)
crs = ccrs.PlateCarree()

plot_args = [
    (gs[0, 0], humid_original_shallow, np.arange(0.05, 0.3, 0.015), "(a)", np.arange(0.05, 0.3, 0.05)),
    (gs[0, 1], humid_modified_shallow, np.arange(0.05, 0.3, 0.015), "(b)", np.arange(0.05, 0.3, 0.05)),
    (gs[0, 2], humid_df_shallow, np.arange(-0.003, 0.003, 0.001), "(c)", np.arange(-0.003, 0.003, 0.002)),
    (gs[1, 0], humid_original_deep, np.arange(0.05, 0.3, 0.015), "(d)", np.arange(0.05, 0.3, 0.05)),
    (gs[1, 1], humid_modified_deep, np.arange(0.05, 0.3, 0.015), "(e)", np.arange(0.05, 0.3, 0.05)),
    (gs[1, 2], humid_df_deep, np.arange(-0.003, 0.003, 0.001), "(f)", np.arange(-0.003, 0.003, 0.002)),
    (gs[2, 0], humid_original_min, np.arange(0.05, 0.25, 0.005), "(g)", np.arange(0.05, 0.25, 0.05)),
    (gs[2, 1], humid_modified_min, np.arange(0.05, 0.25, 0.005), "(h)", np.arange(0.05, 0.25, 0.05)),
    (gs[2, 2], humid_df_min, np.arange(-0.003, 0.003, 0.001), "(i)", np.arange(-0.003, 0.003, 0.002)),
]

for arg in plot_args:
    ax = fig.add_subplot(arg[0], projection=crs)
    cmap = cmaps.cmocean_balance if arg[3] in ["(c)", "(f)", "(i)"] else cmaps.cmocean_tempo
    plot_map_set(ax, lon, lat, arg[1], arg[2], arg[3], shapefile_path, arg[4], cmap=cmap, clip_shape=True, clip_args=[0])

plt.savefig(fname=
'/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/Development_and_Validation_of_a_New_Soil_frost_heave_shceme/fig/after20241223/fig16_new-Comparison of average soil moisture simulations on the Qinghai-Tibet Plateau between the new and original schemes.pdf',
bbox_inches='tight', dpi=1200)
plt.show()
