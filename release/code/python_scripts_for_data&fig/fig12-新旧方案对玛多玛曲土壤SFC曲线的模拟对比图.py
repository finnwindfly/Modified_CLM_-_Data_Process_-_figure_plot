import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import xarray as xr
from cftime import DatetimeNoLeap

# 更新绘图风格
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['axes.unicode_minus'] = False

def get_time_slices():
    """定义所有站点的时间切片"""
    slices = {
        'maduo': [
            slice(DatetimeNoLeap(2017, 10, 1), DatetimeNoLeap(2018, 2, 1)),
            slice(DatetimeNoLeap(2018, 2, 1), DatetimeNoLeap(2018, 7, 1))
        ],
        'maqu': [
            slice(DatetimeNoLeap(2022, 10, 1), DatetimeNoLeap(2023, 2, 1)),
            slice(DatetimeNoLeap(2023, 2, 1), DatetimeNoLeap(2023, 7, 1))
        ],
        'waerma': [
            slice(DatetimeNoLeap(2023, 10, 1), DatetimeNoLeap(2024, 2, 1)),
            slice(DatetimeNoLeap(2024, 2, 1), DatetimeNoLeap(2024, 6, 30))
        ]
    }
    return slices

def load_data(station):
    """加载指定站点的数据"""
    base_path = '/Users/finn/Library/CloudStorage/OneDrive-Personal/share_clm5/model_output/2024-10/soil_defomation_new_scheme/'
    paths = {
        'original': f'soil_ft_defo_{station.upper()}_202410_{"spinup_" if station != "waerma" else ""}original.clm2.h0.{("2012-09-01" if station == "maduo" else "2022-06-01" if station == "maqu" else "2022-07-01")}-00000.nc',
        'modified': f'soil_ft_defo_{station.upper()}_202410_{"spinup_" if station != "waerma" else ""}modified.clm2.h0.{("2012-09-01" if station == "maduo" else "2022-06-01" if station == "maqu" else "2022-07-01")}-00000.nc'
    }
    return {key: xr.open_dataset(base_path + path) for key, path in paths.items()}

def get_processed_data(data, time_slice):
    """处理数据"""
    temp = data.TSOI.isel(lndgrid=0, levgrnd=slice(0, 15)).sel(time=time_slice) - 273.15
    humid = data.SOILLIQ.isel(lndgrid=0).rename({'levsoi': 'levgrnd'}).isel(levgrnd=slice(0,15))
    thickness = data.DZSOI.isel(lndgrid=0, levgrnd=slice(0, 15))
    humid = humid / thickness / 1000.0
    humid = humid.sel(time=time_slice)
    return temp, humid

def plot_soil_curve(ax, temp, humid, label, show_xlabel=False):
    """绘制土壤曲线"""
    temp = temp.mean(dim='levgrnd')
    humid = humid.mean(dim='levgrnd')
    ax.scatter(temp, humid, alpha=0.5, label=label)
    if show_xlabel:
        ax.set_xlabel('Soil Temperature (°C)', fontsize=12)
    ax.set_ylabel('Unfrozen Water Content \n (m³/m³)', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10, length=6, width=1.5)
    ax.tick_params(axis='both', which='minor', length=3, width=1.0)
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    # 设置y轴刻度格式为两位小数
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.legend(frameon=False, loc='best')

# 创建图形和子图
fig = plt.figure(figsize=(10,8))
gs = GridSpec(3, 2, figure=fig, wspace=0, hspace=0)
axes = gs.subplots(sharey='row')

# 设置spines宽度
for row in axes:
    for ax in row:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

# 获取时间切片和数据
time_slices = get_time_slices()
stations = ['maqu', 'waerma', 'maduo']
titles = ['a', 'b', 'c', 'd', 'e', 'f']

# 绘制所有站点数据
for i, station in enumerate(stations):
    data = load_data(station)
    for j, time_slice in enumerate(time_slices[station]):
        temp_orig, humid_orig = get_processed_data(data['original'], time_slice)
        temp_mod, humid_mod = get_processed_data(data['modified'], time_slice)
        
        show_xlabel = (i == 2)  # 只在最后一行显示xlabel
        plot_soil_curve(axes[i,j], temp_orig, humid_orig, 'Original_scheme', show_xlabel)
        plot_soil_curve(axes[i,j], temp_mod, humid_mod, 'New_scheme', show_xlabel)
        
        # 添加标题
        if j == 0:
            axes[i,j].text(0.1, 0.1, f'({titles[i*2]})', transform=axes[i,j].transAxes, fontsize=18)
        else:
            axes[i,j].text(0.9, 0.1, f'({titles[i*2+1]})', transform=axes[i,j].transAxes, fontsize=18)
            
        if i < 2:  # 隐藏前两行的x轴标签
            axes[i,j].set_xticklabels([])

# 设置右侧y轴
for i in range(3):
    axes[i,1].yaxis.tick_right()
    axes[i,1].yaxis.set_label_position("right")
    axes[i,1].yaxis.set_tick_params(labelright=True)

# 反转左侧列x轴
for i in range(3):
    axes[i,0].invert_xaxis()

plt.tight_layout(rect=[0, 0.02, 1, 0.98])
plt.savefig(fname=
'/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/Development_and_Validation_of_a_New_Soil_frost_heave_shceme/fig/after20241223/fig12-Comparative analysis of soil freezing characteristics as simulated by the new and original schemes.png',
bbox_inches='tight', dpi=1200)
plt.show()
