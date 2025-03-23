import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cftime import DatetimeNoLeap
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
import matplotlib.ticker as mtick

def setup_plot_style():
    """设置绘图的基本样式"""
    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['axes.unicode_minus'] = False

def load_site_data(site, times, lev):
    """加载站点数据
    Args:
        site: 站点名称 ('maduo', 'maqu', 'waerma')
        times: 时间序列
        lev: 土壤层深度数组
    Returns:
        original_temp: 原始方案数据
        modified_temp: 新方案数据
        modified_dyndz: 土壤形变数据
    """
    # 文件路径配置
    base_path = '/Users/finn/Library/CloudStorage/OneDrive-Personal/share_clm5/model_output/'
    paths = {
        'maduo': {
            'original': f'{base_path}2024.08/soil_deformation_study_based_on_CLM5.0/2024.08.13/soil_ft_defo_MADUO_20240813_original.clm2.h0.2012-09-01-00000.nc',
            'modified': f'{base_path}2024.08/soil_deformation_study_based_on_CLM5.0/2024.08.13/soil_ft_defo_MADUO_20240813_dyndz_STEP20.clm2.h0.2012-09-01-00000.nc'
        },
        'maqu': {
            'original': f'{base_path}2024.08/soil_deformation_study_based_on_CLM5.0/2024.08.13/soil_ft_defo_MAQU_20240813_v2_original.clm2.h0.2022-06-01-00000.nc',
            'modified': f'{base_path}2024.08/soil_deformation_study_based_on_CLM5.0/2024.08.13/soil_ft_defo_MAQU_20240813_v2_dyndz_STEP20.clm2.h0.2022-06-01-00000.nc'
        },
        'waerma': {
            'original': f'{base_path}2024-10/soil_defomation_new_scheme/soil_ft_defo_Waerma_202410_original.clm2.h0.2022-07-01-00000.nc',
            'modified': f'{base_path}2024-10/soil_defomation_new_scheme/soil_ft_defo_Waerma_202410_modified.clm2.h0.2022-07-01-00000.nc'
        }
    }

    # 读取原始数据
    data_original = xr.open_dataset(paths[site]['original'])
    original_temp = []
    for i in range(2):
        data = data_original.WATSAT.isel(lndgrid=0, levgrnd=lev[i])
        data_interval = np.zeros(len(times))
        data_interval[:] = data
        original_temp.append(data_interval)

    # 读取修改后的数据
    data_modified = xr.open_dataset(paths[site]['modified'])
    modified_temp = []
    time_ranges = {
        'maduo': (DatetimeNoLeap(2017, 9, 1), DatetimeNoLeap(2018, 9, 1)),
        'maqu': (DatetimeNoLeap(2022, 9, 1), DatetimeNoLeap(2023, 6, 1)),
        'waerma': (DatetimeNoLeap(2023, 9, 1), DatetimeNoLeap(2024, 7, 2))
    }
    
    for i in range(2):
        data = data_modified.DYNPOROSITY.isel(lndgrid=0, levgrnd=lev[i])
        data_interval = data.sel(time=slice(*time_ranges[site]))
        modified_temp.append(data_interval)

    # 计算土壤形变
    total_dyndz = xr.zeros_like(data_modified.DYNDZ.sel(levsoi=0).sel(time=slice(*time_ranges[site])))
    for i in range(20):
        data = data_modified.DYNDZ.isel(lndgrid=0, levsoi=i)
        data_interval = data.sel(time=slice(*time_ranges[site]))
        total_dyndz += data_interval

    return original_temp, modified_temp, total_dyndz

def setup_axis(ax, title, xlim, ylabel, has_xlabel=True, row=0, site=None, show_ylabel=True):
    """设置坐标轴样式
    Args:
        ax: matplotlib轴对象
        title: 图标题
        xlim: x轴范围
        ylabel: y轴标签
        has_xlabel: 是否显示x轴标签
        row: 行号(0或1)
        site: 站点名称
        show_ylabel: 是否显示y轴标签
    """
    ax.tick_params(axis='both', direction='out', length=5, width=2, labelsize=12)
    ax.spines[:].set_linewidth(2.5)
    ax.set_title(title, fontdict=dict(fontsize=14), loc='left')
    ax.set_xlim(xlim)
    
    # 根据站点名称设置不同的ylim
    if site == 'maqu':
        ax.set_ylim(top=0.6 if row==0 else 0.42)
    elif site == 'waerma':
        ax.set_ylim(top=0.7 if row==0 else 0.6)
    elif site == 'maduo':
        ax.set_ylim(top=0.5 if row==0 else 0.45)
        
    if show_ylabel:
        ax.set_ylabel(ylabel, fontdict=dict(fontsize=12))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    
    # 设置5个均匀分布的x轴刻度
    dates = pd.date_range(xlim[0], xlim[1], periods=5)
    ax.set_xticks(dates)
    
    if has_xlabel:
        ax.set_xticklabels([d.strftime('%b') for d in dates], fontsize=12)
        ax.set_xlabel('Time', fontdict=dict(fontsize=12))
    else:
        ax.xaxis.set_ticklabels([])

def plot_porosity_deformation():
    """主绘图函数"""
    # 设置基本样式
    setup_plot_style()
    
    # 准备时间序列和深度数据
    times = {
        'maduo': pd.date_range('9/1/2017', '9/1/2018'),
        'maqu': pd.date_range('9/1/2022', '6/1/2023'),
        'waerma': pd.date_range('9/1/2023', '6/30/2024')
    }
    
    lev = {'maduo': np.array([3, 7]), 
            'maqu': np.array([3, 7]),
            'waerma': np.array([3, 6])}

    # 加载数据
    maduo_data = load_site_data('maduo', times['maduo'], lev['maduo'])
    maqu_data = load_site_data('maqu', times['maqu'], lev['maqu'])
    waerma_data = load_site_data('waerma', times['waerma'], lev['waerma'])

    # 创建图形
    fig = plt.figure(figsize=(9, 12))
    
    # 绘制土壤孔隙度变化
    sites = ['maqu', 'waerma', 'maduo']
    colors = ['#3b658c', '#076e6d', '#9b5959']
    data = [maqu_data, waerma_data, maduo_data]
    times_data = [times['maqu'], times['waerma'], times['maduo']]
    
    for i in range(2):
        for j, (site, color, site_data, time_series) in enumerate(zip(sites, colors, data, times_data)):
            ax = plt.subplot2grid((4,3), (i+2,j))
            ax.plot(time_series, site_data[0][i], color='black', linewidth=4)
            ax.plot(time_series, site_data[1][i], label='enhanced model', color=color, linewidth=4)
            title_letter = chr(ord("d") + i*3 + j)
            title = f'({title_letter}) {site.capitalize()} {level[i]} m'
            setup_axis(ax, title, (time_series[0], time_series[-1]), 
                        r'soil porosity/$(mm^3/mm^3)$', i==1, i, site, show_ylabel=(j==0))
            # 设置y轴下限为数据最小值
            ymin = min(min(site_data[0][i]), min(site_data[1][i].values))
            ax.set_ylim(bottom=ymin)
            # 只在第二行显示图例
            if i == 1:
                ax.legend(frameon=False, loc='upper left', fontsize=12)

    # 绘制土壤形变
    for i, (site, color, site_data, time_series) in enumerate(zip(sites, colors, data, times_data)):
        ax = plt.subplot2grid((4,3), (0,i), rowspan=2)
        ax.bar(time_series, site_data[2].values.flatten()-12.9, color=color, alpha=1.0)
        title = f'({chr(ord("a")+i)}) {site.capitalize()}'
        setup_axis(ax, title, (time_series[0], time_series[-1]), 
                    'soil deformation/(m)', False, site=site, show_ylabel=(i==0))
        ax.set_ylim(0, 0.035)

    plt.tight_layout()
    plt.savefig('/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/Development_and_Validation_of_a_New_Soil_frost_heave_shceme/fig/after20241223/fig5-Comparison of soil deformation and porosity changes using the new and old schemes.png',
                bbox_inches='tight', dpi=1200)
    plt.show()

# 全局变量
level = np.array([0.2, 0.8])

# 执行绘图
if __name__ == "__main__":
    plot_porosity_deformation()
