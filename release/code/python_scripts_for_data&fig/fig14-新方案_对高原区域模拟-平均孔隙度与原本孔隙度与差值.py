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

# 打开NetCDF文件并提取变量
nc_modified = xr.open_dataset(
'/Volumes/Finn‘sT7/Data/博士资料/Development_and_Validation_of_a_New_Soil_frost_heave_shceme/CTSM-Regional/2024.10/150x325_TP_xpf_ft_modified.clm2.h0.2016-06-01-00000.nc'
)
ft_slice_time = slice(DatetimeNoLeap(2016, 12, 1), DatetimeNoLeap(2017, 6, 1))
dynporosity = nc_modified['DYNPOROSITY'].sel(time=ft_slice_time).mean(dim='levgrnd').mean(dim='time')
porosity = nc_modified['WATSAT'].mean(dim='levgrnd')
print(porosity.max())
# dynporosity = dynporosity-porosity
# dynporosity = dynporosity.where(dynporosity>=0.0, 0.0)
print(dynporosity.max())
porosity_dif = dynporosity-porosity
porosity_dif = porosity_dif.where(porosity_dif>=0.0, 0.0)
print(porosity_dif.max())

# 提取经纬度
lon = porosity.lon.values
lat = porosity.lat.values

num_lat = len(lat)
num_lon = len(lon)

#%% mask函数
def shp2clip(originfig, ax, shpfile, fieldVals):
    """
    for clip map with shape file
    """
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

#%%绘图
# 设置全局字体为 New Times Roman
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'

def plot_map(ax, lon, lat, data, levels, title, cmap, shapefile_path, cb_ticks, clip_shape=False, clip_args=None):
    ax.set_extent([75.2, 105.5, 25.0, 42.3], crs=ccrs.PlateCarree())
    ax.set_xticks(list(range(74, 107, 8)), crs=ccrs.PlateCarree())
    ax.set_yticks(list(range(24, 41, 4)), crs=ccrs.PlateCarree())
    ax.tick_params(labelcolor='k', length=5, width=3, labelsize=18)

    #把40转换为40^oN的经纬度格式
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.spines['geo'].set_linewidth(3)

    #下面是对shape文件进行读取然后画出来，没有颜色但是边缘黑色
    reader = Reader(shapefile_path)
    geoms = reader.geometries() #把形状读出来
    ax.add_geometries(geoms, ccrs.PlateCarree(), lw=2, fc='none') #画出来shape文件，facecolor没有，边框粗细度为3

    # 下面是画数据的contourf图
    norm = mpl.colors.BoundaryNorm(levels, cmap.N, extend='both') #设置标准化，level的标准，extend=both表示色标三角形延展出去
    cn = ax.contourf(lon, lat, data, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, levels=levels, extend='both')
    ax.set_title(title, fontsize=20, loc='left')

    # 使用shapefile对图形进行裁切使shapefile之外的图形白化掉
    if clip_shape and clip_args is not None:
        clip = shp2clip(cn, ax, shapefile_path, clip_args)

    # 为每个图形添加colorbar
    cax = inset_axes(ax, width="100%", height="9%", loc='lower center',
                bbox_to_anchor=(0.0, -0.3, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cb = plt.colorbar(cn, cax=cax, orientation='horizontal', ticks=cb_ticks)
    cb.ax.tick_params(labelsize=18, direction='out')
    cb.outline.set_linewidth(1.5)
    cax.xaxis.set_ticks_position('bottom')
    cb.ax.text(1.0, -0.88, 'm³/m³', fontsize=18, ha='left', va='center', transform=cb.ax.transAxes)


fig = plt.figure(figsize=(12,8))
crs = ccrs.PlateCarree()

# Constants and paths
shapefile_path = '/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/老师学生资料相关/盛丹睿/气候区/tibet_shp/qingz(bnd)(geo).shp'

# Plot the first map
gs = GridSpec(2, 4, figure=fig, hspace=0.8)

ax1 = fig.add_subplot(gs[0,0:2], projection=crs)
plot_map(ax1, lon, lat, porosity, np.arange(0.4, 0.7, 0.03),
        "(a) ", cmaps.cmocean_speed,
        shapefile_path, np.arange(0.4, 0.7, 0.05),
        clip_shape=True, clip_args=[0])

# Plot the second map
ax2 = fig.add_subplot(gs[0,2:], projection=crs)
plot_map(ax2, lon, lat, dynporosity, np.arange(0.4, 0.7, 0.03),
        "(b) ", cmaps.cmocean_speed,
        shapefile_path, np.arange(0.4, 0.7, 0.05),
        clip_shape=True, clip_args=[0])

ax3 = fig.add_subplot(gs[1,1:3], projection=crs)
plot_map(ax3, lon, lat, porosity_dif, np.arange(0.0, 0.2, 0.01),
        "(c)", cmaps.cmocean_speed,
        shapefile_path, np.arange(0., 0.2, 0.04),
        clip_shape=True, clip_args=[0])

# plt.tight_layout()

plt.savefig(fname=
'/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/Development_and_Validation_of_a_New_Soil_frost_heave_shceme/fig/after20241207/fig14-Regional simulation comparison of average porosity on the Qinghai-Tibet Plateau between the original and new schemes.pdf',
bbox_inches='tight', dpi=1200)
plt.show()
