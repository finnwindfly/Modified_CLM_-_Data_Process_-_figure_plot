import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from cftime import DatetimeNoLeap
from matplotlib.lines import Line2D
import matplotlib.dates as mdates

# 设置字体和参数以便正确显示中文

plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['axes.unicode_minus']=False

lev_obs_maqu = np.array([0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
lev_obs_maduo = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2])
lev_obs_waerma = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])

da_nonan1 = pd.read_excel(
    '/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/2024基于CLM5.0冻融过程土壤形变对土壤水热的影响研究/单点强迫场地表数据制作/玛曲马场站-土壤加密数据/2021.11-2023.09玛曲站加密观测土壤温湿度数据整理.xlsx', na_values='NAN')
da_nonan1['times'] = pd.to_datetime(da_nonan1['times'], dayfirst=True) - pd.Timedelta(hours=8)  #! 这里是为了将time这个时间换成datetime格式，这样更方便用pandas处理
da_nonan1 = da_nonan1.set_index('times')#! 这里就是把这个dataframe的index设为time这一列。
da_nonan1_select = da_nonan1.loc['2022-11-15':'2023-6-1']
group1 = da_nonan1_select.resample('D').mean()
times_maqu = pd.date_range('11/15/2022', '6/1/2023')

columns_maqu = ['5cm_TSoil_Avg', '10cm_TSoil_Avg', '20cm_TSoil_Avg', '40cm_TSoil_Avg', '50cm_TSoil_Avg', '60cm_TSoil_Avg',
                '80cm_TSoil_Avg', '100cm_TSoil_Avg', '120cm_TSoil_Avg', '140cm_TSoil_Avg','160cm_TSoil_Avg']

data_obs_maqu = group1[columns_maqu]
print(data_obs_maqu)

#!##########################===接下来是读取观测数据====#######################################################################################   
da_nonan2 = pd.read_csv(
       '/Users/finn/Library/CloudStorage/OneDrive-Personal/share_clm5/FORCEDATA/maduo/observation_data_maduo/selected_obs_data/data_obs_maduo.csv', na_values='NAN')
#da_nonan2 = da_obs_maduo.interpolate() # 这里是为了去除NAN值，通过前后两个值的插值来代替
da_nonan2['time'] = pd.to_datetime(da_nonan2['time'], dayfirst=True) - pd.Timedelta(hours=8)  #! 这里是为了将time这个时间换成datetime格式，这样更方便用pandas处理
da_nonan2 = da_nonan2.set_index('time')#! 这里就是把这个dataframe的index设为time这一列。
da_nonan2_select = da_nonan2.loc['2017-10-1':'2018-6-1']
group = da_nonan2_select.resample('D').mean()
times_maduo = pd.date_range('10/1/2017', '6/1/2018')

columns_maduo = ['Soil_T_5cm_Avg', 'Soil_T_10cm_Avg', 'Soil_T_20cm_Avg', 'Soil_T_40cm_Avg',
                'Soil_T_80cm_Avg', 'Soil_T_160cm_Avg', 'Soil_T_320cm_Avg']

data_obs_maduo = group[columns_maduo]
print(data_obs_maduo)

da_nonan1 = pd.read_excel(
    '/Users/finn/Library/CloudStorage/OneDrive-Personal/share_clm5/FORCEDATA/Waerma/1x1_Waerma_EC_202410/Waerma-soil-EC-edit.xlsx'
    , na_values='NAN').drop(index=0)
da_nonan1['times'] = pd.to_datetime(da_nonan1['time'], dayfirst=True) - pd.Timedelta(hours=8)  #! 这里是为了将time这个时间换成datetime格式，这样更方便用pandas处理
da_nonan1 = da_nonan1.set_index('times')#! 这里就是把这个dataframe的index设为time这一列。
da_nonan1_select = da_nonan1.loc['2023-9-1':'2024-6-30']
group1 = da_nonan1_select.resample('D').mean()
times_waerma = pd.date_range('9/1/2023', '6/30/2024')

columns_waerma = ['Soil_Tem_5cm_Avg', 'Soil_Tem_10cm_Avg', 'Soil_Tem_20cm_Avg', 'Soil_Tem_40cm_Avg',
                'Soil_Tem_60cm_Avg', 'Soil_Tem_80cm_Avg', 'Soil_Tem_100cm_Avg', 'Soil_Tem_120cm_Avg', 'Soil_Tem_140cm_Avg', 'Soil_Tem_160cm_Avg']

data_temp_day_waerma = group1[columns_waerma]
data_obs_waerma = pd.DataFrame(data_temp_day_waerma)
print(data_obs_waerma)

#读取nc文件之中的ZSOI土壤深度数据
data_one = xr.open_dataset(
'/Users/finn/Library/CloudStorage/OneDrive-Personal/share_clm5/model_output/2024-10/soil_defomation_new_scheme/soil_ft_defo_MADUO_202410_spinup_modified.clm2.h0.2012-09-01-00000.nc')
lev_de = data_one.ZSOI.isel(lndgrid=0)

#定义一个读取nc文件的函数
def read_data(file_path, time_slice):
    data = xr.open_dataset(file_path)
    dataset_soil_temp = data.TSOI.isel(lndgrid=0) - 273.15
    dataset_soil_temp = dataset_soil_temp.sel(time=slice(*time_slice)).values.T

    return dataset_soil_temp

# 读取玛曲和玛多的数据
time_slice_maduo = (DatetimeNoLeap(2017, 10, 1), DatetimeNoLeap(2018, 6, 1))
time_slice_maqu = (DatetimeNoLeap(2022, 11, 15), DatetimeNoLeap(2023, 6, 1))
time_slice_waerma = (DatetimeNoLeap(2023, 9, 1), DatetimeNoLeap(2024, 7, 1))

# 玛多
files_maduo = [
    'soil_ft_defo_MADUO_202410_spinup_original.clm2.h0.2012-09-01-00000.nc',
    'soil_ft_defo_MADUO_202410_spinup_modified.clm2.h0.2012-09-01-00000.nc',
]

dataset_maduo = [read_data(f'/Users/finn/Library/CloudStorage/OneDrive-Personal/share_clm5/model_output/2024-10/soil_defomation_new_scheme/{file}', time_slice_maduo) for file in files_maduo]
print(dataset_maduo[0].shape)

# 玛曲
files_maqu = [
    'soil_ft_defo_MAQU_202410_spinup_original.clm2.h0.2022-06-01-00000.nc',
    'soil_ft_defo_MAQU_202410_spinup_modified.clm2.h0.2022-06-01-00000.nc',
]

dataset_maqu = [read_data(f'/Users/finn/Library/CloudStorage/OneDrive-Personal/share_clm5/model_output/2024-10/soil_defomation_new_scheme/{file}', time_slice_maqu) for file in files_maqu]

# 玛曲
files_waerma = [
    'soil_ft_defo_Waerma_202410_original.clm2.h0.2022-07-01-00000.nc',
    'soil_ft_defo_Waerma_202410_modified.clm2.h0.2022-07-01-00000.nc',
]

dataset_waerma = [read_data(f'/Users/finn/Library/CloudStorage/OneDrive-Personal/share_clm5/model_output/2024-10/soil_defomation_new_scheme/{file}', time_slice_waerma) for file in files_waerma]

# 定义绘图函数
def plot_data(datasets, obs_data, title_prefix, lev_de, lev_obs, ylabel, ax, row_offset, ylim, text, colors):
        ax[row_offset].tick_params(axis='both',
        direction='out', # 刻度线朝内
        length=5, width=2, # 长度和宽度
        )
        ax[row_offset].spines[:].set_linewidth(2.5)
        for i, dataset in enumerate(datasets):  # 文件
        # 画ax.contour()
            x1, y1 = np.meshgrid(obs_data.index, lev_de)
            contour_data = dataset
            times = obs_data.index
            obs_data_values = obs_data.values.T
            CS1 = ax[row_offset].contour(x1, y1, contour_data, levels=[0], colors=colors[i], linewidths=3.5)
        x2, y2 = np.meshgrid(obs_data.index, lev_obs)
        CS2 = ax[row_offset].contour(x2, y2, obs_data_values, levels=[0], colors='k', linewidths=3.5)

        ax[row_offset].set_ylim(0, ylim)
        ax[row_offset].invert_yaxis()

        # 设置ylabel和spines粗细
        ax[row_offset].set_ylabel(ylabel, fontsize=15)
        for spine in ax[row_offset].spines.values():
            spine.set_linewidth(2.0)

        # 设置xlim和xlabel
        ax[row_offset].set_xlim(obs_data.index[0], obs_data.index[-1])
        ax[row_offset].set_xlabel('Time' , fontsize=18)

        # 设置x轴刻度格式为%b，不旋转
        ax[row_offset].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.setp(ax[row_offset].get_xticklabels(), ha="center", fontsize=16)

        # 设置ticklabel字体大小
        ax[row_offset].tick_params(axis='both', labelsize=15)

        ax[row_offset].set_title(text, loc='left', fontsize=20)


colors = ['#3b658c', '#c82423', 'red', 'c']  # 为每个文件指定颜色
# colors = ['#F94141', '#F3B169', '#589FF3', '#37AB78']  # 为每个文件指定颜色
labes = ['original_scheme', 'new_scheme', '30sl-scheme','40sl-scheme']

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
# fig.suptitle('Soil Moisture and Temperature Comparison')

plot_data(dataset_maqu, data_obs_maqu, 'Maqu Soil Moisture', lev_de, lev_obs_maqu, r'Soil depth (m)', ax, 0, 2.0, '(a)', colors)
plot_data(dataset_waerma, data_obs_waerma, 'Waerma Soil Moisture', lev_de, lev_obs_waerma, r'Soil depth (m)', ax, 1, 2.0, '(b)', colors)
plot_data(dataset_maduo, data_obs_maduo, 'Maduo Soil Moisture', lev_de, lev_obs_maduo, r'Soil depth (m)', ax, 2, 3.0, '(c)', colors)

custom_lines=[]
for j in np.arange(3):
    color_legend = ['k', '#3b658c','#c82423']
    # color_legend = ['#808080','#F94141', '#F3B169', '#589FF3', '#37AB78']
    line = Line2D([0], [0], color=color_legend[j],lw=2.5)
    custom_lines.append(line)
ax[0].legend(custom_lines, ['observation', 'original_scheme', 'new_scheme','30sl-scheme','40sl-scheme'],
            frameon=False, loc='lower center', fontsize=12, handlelength=2)

plt.tight_layout()
plt.savefig(fname=
'/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/Development_and_Validation_of_a_New_Soil_frost_heave_shceme/fig/after20241223/fig11-Comparison of simulated soil freezing front between the new and original schemes against observed data.png',
bbox_inches='tight', dpi=1200)
plt.show()
