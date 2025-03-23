import matplotlib.pyplot as plt
import xarray as xr
from cftime import DatetimeNoLeap
import numpy as np
from matplotlib.ticker import FormatStrFormatter

# 设置绘图参数
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['axes.unicode_minus'] = False

# 定义时间片段
time_slices = {
    'maduo': slice(DatetimeNoLeap(2017, 11, 1), DatetimeNoLeap(2018, 5, 1)),
    'maqu': slice(DatetimeNoLeap(2022, 12, 1), DatetimeNoLeap(2023, 4, 1)), 
    'waerma': slice(DatetimeNoLeap(2023, 11, 1), DatetimeNoLeap(2024, 6, 1))
}

# 读取数据
data_files = {
    'maduo': '/Users/finn/Library/CloudStorage/OneDrive-Personal/share_clm5/model_output/2024-10/soil_defomation_new_scheme/soil_ft_defo_MADUO_202410_spinup_modified.clm2.h0.2012-09-01-00000.nc',
    'maqu': '/Users/finn/Library/CloudStorage/OneDrive-Personal/share_clm5/model_output/2024-10/soil_defomation_new_scheme/soil_ft_defo_MAQU_202410_spinup_modified.clm2.h0.2022-06-01-00000.nc',
    'waerma': '/Users/finn/Library/CloudStorage/OneDrive-Personal/share_clm5/model_output/2024-10/soil_defomation_new_scheme/soil_ft_defo_Waerma_202410_modified.clm2.h0.2022-07-01-00000.nc'
}

data = {site: xr.open_dataset(path) for site, path in data_files.items()}

# 处理数据函数
def process_site_data(data_site, time_slice):
    levsoi = data_site.ZSOI.isel(lndgrid=0, levgrnd=slice(0, 20))
    dynporosity = data_site.DYNPOROSITY.isel(lndgrid=0, levgrnd=slice(0, 20)).sel(time=time_slice).mean(dim='time')
    watsat = data_site.WATSAT.isel(lndgrid=0, levgrnd=slice(0, 20))
    organic = data_site.ORGANIC_COL.isel(lndgrid=0, levsoi=slice(0, 20)).sel(time=time_slice).mean(dim='time')
    ctriice = data_site.DYNCTRIWAT.isel(lndgrid=0, levgrnd=slice(0, 20)).sel(time=time_slice).fillna(0).mean(dim='time')
    bd_col = data_site.BD_COL.isel(lndgrid=0, levgrnd=slice(0, 20)).sel(time=time_slice)
    
    return {
        'levsoi': levsoi,
        'dynporosity': dynporosity,
        'watsat': watsat,
        'organic': organic,
        'ctriice': ctriice,
        'bd_col_mean': bd_col.mean(dim='time'),
        'bd_col_min': bd_col.min(dim='time'),
        'bd_col_max': bd_col.max(dim='time')
    }

site_data = {site: process_site_data(data[site], time_slices[site]) for site in data.keys()}

# 创建图形
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(10, 12))
plt.subplots_adjust(wspace=0.27, hspace=0.25)

# 设置轴参数
for ax_row in axs:
    for ax in ax_row:
        ax.tick_params(axis='both', which='both', labelsize=12)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# 绘图参数
plot_params = {
    'Maqu': {'ylim': (0, 2), 'alia': ['(a)', '(b)', '(c)', '(d)']},
    'Waerma': {'ylim': (0, 3), 'alia': ['(e)', '(f)', '(g)', '(h)']},
    'Maduo': {'ylim': (0, 4), 'alia': ['(i)', '(j)', '(k)', '(l)']}
}

colors = {
    'organic': '#3b658c',
    'ctriice': '#c82423',
    'porosity': '#076e6d',
    'bd': '#a6a682'
}

# 绘图
for row, (location, params) in enumerate(plot_params.items()):
    site = location.lower()
    data = site_data[site]
    
    # 绘制四列图
    for col in range(4):
        ax = axs[row, col]
        ax.set_ylim(*params['ylim'])
        ax.invert_yaxis()
        ax.text(0.08, 0.01, params['alia'][col], transform=ax.transAxes,
                fontsize=15, fontweight='bold', ha='left', va='bottom')
        
        # 通用设置
        ax.xaxis.tick_top()
        ax.tick_params(bottom=False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylabel('Soil Depth (m)')
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        
        if col == 0:  # Organic
            ax.plot(data['organic'], data['levsoi'], color=colors['organic'], 
                marker='o', linewidth=2.5, markersize=8)
            # 设置x轴范围和刻度
            x_min = min(data['organic'])
            x_max = max(data['organic'])
            ax.set_xlim(x_min, x_max)
            xticks = np.linspace(x_min, x_max, num=5)
            ax.set_xticks(xticks)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
            
        elif col == 1:  # CtrIice
            ax.plot(data['ctriice'], data['levsoi'], color=colors['ctriice'], 
                marker='s', linewidth=2.5, markersize=8)
            # 设置x轴范围和刻度
            x_min = min(data['ctriice'])
            x_max = max(data['ctriice'])
            ax.set_xlim(x_min, x_max)
            xticks = np.linspace(x_min, x_max, num=5)
            ax.set_xticks(xticks)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
            
        elif col == 2:  # Porosity
            ax.plot(data['watsat'], data['levsoi'], color=colors['porosity'], 
                marker='*', label='original_model', linewidth=2.5, markersize=8)
            ax.plot(data['dynporosity'], data['levsoi'], color=colors['porosity'], 
                linestyle='--', marker='*', label='new_model', linewidth=2.5, markersize=6)
            if row == 0: ax.legend(loc='center right', frameon=False)
            # 设置x轴范围和刻度
            x_min = min(min(data['watsat']), min(data['dynporosity']))
            x_max = max(max(data['watsat']), max(data['dynporosity']))
            ax.set_xlim(x_min, x_max)
            xticks = np.linspace(x_min, x_max, num=5)
            ax.set_xticks(xticks)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
            
        else:  # BD_COL
            ax.plot(data['bd_col_mean'], data['levsoi'], color=colors['bd'], 
                marker='p', label='mean_values', linewidth=2.5, markersize=8)
            ax.fill_betweenx(data['levsoi'], data['bd_col_min'], data['bd_col_max'], 
                        color='gray', alpha=0.3, label='range')
            if row == 0: ax.legend(loc='center left', frameon=False)
            # 设置x轴范围和刻度
            x_min = min(min(data['bd_col_mean']), min(data['bd_col_min']), min(data['bd_col_max']))
            x_max = max(max(data['bd_col_mean']), max(data['bd_col_min']), max(data['bd_col_max']))
            ax.set_xlim(x_min, x_max)
            xticks = np.linspace(x_min, x_max, num=4)
            ax.set_xticks(xticks)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

# 保存图形
plt.savefig('/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/Development_and_Validation_of_a_New_Soil_frost_heave_shceme/fig/after20241223/fig8-Soil profile diagrams illustrating porosity texture and bulk density according to the new scheme.png',
            bbox_inches='tight', dpi=1200)

plt.show()
