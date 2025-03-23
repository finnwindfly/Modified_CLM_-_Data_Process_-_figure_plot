import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cftime import DatetimeNoLeap
from matplotlib.ticker import ScalarFormatter
import matplotlib.dates as mdates

#!壤离散化方案回到CLM5.0默认；
#!然后是soil_resis_method = 0 将土壤蒸发阻抗换成CLM4.5的。
#!并且驱动场数据也修改为正确的1x1_MAQU_20221102
#!今天这个是使用CLM5.0默认的导热率参数化方案。

# import matplotlib.font_manager as fm
# a=sorted([f.name for f in fm.fontManager.ttflist])
# for i in a:
#     print(i)
#!上面这四行是print出mac之中valid的字体

plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['axes.unicode_minus']=False
#!上面是为了macos正确显示中文

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!然后读取模式模拟的maduo站结果
lev = np.array([1, 3, 5, 7])
data_original_maduo = xr.open_dataset(
'/Users/finn/Library/CloudStorage/OneDrive-Personal/share_clm5/model_output/2024-10/soil_defomation_new_scheme/soil_ft_defo_MADUO_202410_spinup_original.clm2.h0.2012-09-01-00000.nc'
)
data_original_humid_maduo = []
for i in np.arange(4):
    data_soilliq = data_original_maduo.SOILLIQ.isel(lndgrid=0, levsoi=lev[i])
    data_thickness = data_original_maduo.DZSOI.isel(lndgrid=0, levgrnd=lev[i])
    data_water = data_soilliq / data_thickness / 1000
    data_interval12 = data_water.sel(time=slice(DatetimeNoLeap(2017, 9, 1), DatetimeNoLeap(2018, 9, 1)))
    data_original_humid_maduo.append(data_interval12)

data_modified_maduo = xr.open_dataset(
'/Users/finn/Library/CloudStorage/OneDrive-Personal/share_clm5/model_output/2024-10/soil_defomation_new_scheme/soil_ft_defo_MADUO_202410_spinup_modified.clm2.h0.2012-09-01-00000.nc'
)
data_modified_humid_maduo = []
for i in np.arange(4):
    data_soilliq = data_modified_maduo.SOILLIQ.isel(lndgrid=0, levsoi=lev[i])
    data_thickness = data_modified_maduo.DYNDZ.isel(lndgrid=0, levsoi=lev[i])
    data_water = data_soilliq / data_thickness / 1000
    data_interval12 = data_water.sel(time=slice(DatetimeNoLeap(2017, 9, 1), DatetimeNoLeap(2018, 9, 1)))
    data_modified_humid_maduo.append(data_interval12)
#!####################################程序开始读取模式输出的nc文件###########################################################################
data_original_maqu = xr.open_dataset(
'/Users/finn/Library/CloudStorage/OneDrive-Personal/share_clm5/model_output/2024-10/soil_defomation_new_scheme/soil_ft_defo_MAQU_202410_spinup_original.clm2.h0.2022-06-01-00000.nc'
)
data_original_humid_maqu = []
for i in np.arange(4):
    data = data_original_maqu.SOILLIQ.isel(lndgrid=0, levsoi=lev[i])
    data_thickness = data_original_maqu.DZSOI.isel(lndgrid=0, levgrnd=lev[i])
    data_water = data / data_thickness / 1000
    data_interval_2 = data_water.sel(time=slice(DatetimeNoLeap(2022, 6, 1), DatetimeNoLeap(2023, 9, 1)))
    data_original_humid_maqu.append(data_interval_2)

data_modified_maqu = xr.open_dataset(
'/Users/finn/Library/CloudStorage/OneDrive-Personal/share_clm5/model_output/2024-10/soil_defomation_new_scheme/soil_ft_defo_MAQU_202410_spinup_modified.clm2.h0.2022-06-01-00000.nc'
)
data_modified_humid_maqu = []
for i in np.arange(4):
    data = data_modified_maqu.SOILLIQ.isel(lndgrid=0, levsoi=lev[i])
    dyndz = data_modified_maqu.DYNDZ.isel(lndgrid=0, levsoi=lev[i])
    data = data / (dyndz * 1000.0)
    # data = data_original.SOILLIQ.isel(lndgrid=0, levsoi=lev[i])
    # data = data / dzoi[i]
    data_interval_2 = data.sel(time=slice(DatetimeNoLeap(2022, 6, 1), DatetimeNoLeap(2023, 9, 1)))
    data_modified_humid_maqu.append(data_interval_2)

#!####################################程序开始读取模式输出的nc文件###########################################################################
data_original_waerma = xr.open_dataset(
'/Users/finn/Library/CloudStorage/OneDrive-Personal/share_clm5/model_output/2024-10/soil_defomation_new_scheme/soil_ft_defo_Waerma_202410_original.clm2.h0.2022-07-01-00000.nc'
)
data_original_humid_waerma = []
for i in np.arange(4):
    data = data_original_waerma.SOILLIQ.isel(lndgrid=0, levsoi=lev[i])
    data_thickness = data_original_waerma.DZSOI.isel(lndgrid=0, levgrnd=lev[i])
    data_water = data / data_thickness / 1000
    data_interval_2 = data_water.sel(time=slice(DatetimeNoLeap(2023, 9, 1), DatetimeNoLeap(2024, 7, 1)))
    data_original_humid_waerma.append(data_interval_2)

data_modified_waerma = xr.open_dataset(
'/Users/finn/Library/CloudStorage/OneDrive-Personal/share_clm5/model_output/2024-10/soil_defomation_new_scheme/soil_ft_defo_Waerma_202410_8wat_dynbd_dynsand.clm2.h0.2022-07-01-00000.nc'
)
data_modified_humid_waerma = []
for i in np.arange(4):
    data = data_modified_waerma.SOILLIQ.isel(lndgrid=0, levsoi=lev[i])
    dyndz = data_modified_waerma.DYNDZ.isel(lndgrid=0, levsoi=lev[i])
    data = data / (dyndz * 1000.0)
    # data = data_original.SOILLIQ.isel(lndgrid=0, levsoi=lev[i])
    # data = data / dzoi[i]
    data_interval_2 = data.sel(time=slice(DatetimeNoLeap(2023, 9, 1), DatetimeNoLeap(2024, 7, 1)))
    data_modified_humid_waerma.append(data_interval_2)

#!#########################################################################################################################################
#!#################################接下来这一部分是读取观测数据##############################################################################
da_observational = pd.read_csv(
           '/Users/finn/OneDrive/share_clm5/FORCEDATA/maduo/observation_data_maduo/selected_obs_data/data_obs_maduo201108_201812.csv',
                                 na_values='NAN')
da_nonan1 = da_observational.interpolate() # 这里是为了去除NAN值，通过前后两个值的插值来代替
da_nonan1['time'] = pd.to_datetime(da_nonan1['time'], dayfirst=True)  #! 这里是为了将time这个时间换成datetime格式，这样更方便用pandas处理
da_nonan1['time'] = da_nonan1['time'] - pd.Timedelta(hours=8) #! 这个地方更加重要，时间减去8hours由北京时间转换成世界时
da_nonan1 = da_nonan1.set_index('time')#! 这里就是把这个dataframe的index设为time这一列。
da_nonan1_select = da_nonan1.loc['2017-9-1':'2018-9-1']
group = da_nonan1_select.resample('D').mean()
print(group.Soil_T_5cm_Avg)

times_maduo = pd.date_range('9/1/2017', '9/1/2018')
print(times_maduo)

data_soiltemp_day_maduo = {
    'time': times_maduo,
    'Soil_VWC_5cm_AVG': group.VWC_5cm_Avg.values,
    # 'Soil_VWC_10cm_AVG': group.VWC_10cm_Avg.values,
    'Soil_VWC_20cm_AVG': group.VWC_20cm_Avg.values,
    'Soil_VWC_40cm_AVG': group.VWC_40cm_Avg.values,
    'Soil_VWC_80cm_AVG': group.VWC_80cm_Avg.values,
    # 'Soil_VWC_160cm_AVG': group.VWC_160cm_Avg.values,
}
data_obs_maduo = pd.DataFrame(data_soiltemp_day_maduo)
data_obs_maduo['time'] = pd.to_datetime(data_obs_maduo['time'])
data_obs_maduo = data_obs_maduo.set_index('time')
print(data_obs_maduo)

#!#################################接下来这一部分是读取观测数据##############################################################################
da_nonan1 = pd.read_excel(
    '/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/2024基于CLM5.0冻融过程土壤形变对土壤水热的影响研究/单点强迫场地表数据制作/玛曲马场站-土壤加密数据/2021.11-2023.09玛曲站加密观测土壤温湿度数据整理.xlsx', na_values='NAN')
da_nonan1['times'] = pd.to_datetime(da_nonan1['times'], dayfirst=True) - pd.Timedelta(hours=8)  #! 这里是为了将time这个时间换成datetime格式，这样更方便用pandas处理
da_nonan1 = da_nonan1.set_index('times')#! 这里就是把这个dataframe的index设为time这一列。
da_nonan1_select = da_nonan1.loc['2022-6-1':'2023-9-1']
group1 = da_nonan1_select.resample('D').mean()
times_maqu = pd.date_range('6/1/2022', '9/1/2023')

columns_maqu = ['5cm_VWC_Avg', '20cm_VWC_Avg',
                '40cm_VWC_Avg', '80cm_VWC_Avg']

data_temp_day_maqu = group1[columns_maqu]
data_obs_maqu = pd.DataFrame(data_temp_day_maqu)
print(data_obs_maqu)

#!#################################接下来这一部分是读取观测数据##############################################################################
da_nonan1 = pd.read_excel(
    '/Users/finn/Library/CloudStorage/OneDrive-Personal/share_clm5/FORCEDATA/Waerma/1x1_Waerma_EC_202410/Waerma-soil-EC-edit.xlsx',
    na_values='NAN').drop(index=0)
print(da_nonan1)

da_nonan1['times'] = pd.to_datetime(da_nonan1['time'], dayfirst=True) - pd.Timedelta(hours=8)  #! 这里是为了将time这个时间换成datetime格式，这样更方便用pandas处理
da_nonan1 = da_nonan1.set_index('times')#! 这里就是把这个dataframe的index设为time这一列。
da_nonan1_select = da_nonan1.loc['2023-9-1':'2024-6-30']
group1 = da_nonan1_select.resample('D').mean()
times_waerma = pd.date_range('9/1/2023', '6/30/2024')

columns_waerma = ['Soil_VWC_5cm_Avg', 'Soil_VWC_20cm_Avg',
                'Soil_VWC_40cm_Avg', 'Soil_VWC_80cm_Avg']

data_temp_day_waerma = group1[columns_waerma]
data_temp_day_waerma = data_temp_day_waerma / 100.0
data_obs_waerma = pd.DataFrame(data_temp_day_waerma)
print(data_obs_waerma)

# frstart_maduo = ['2017-10-25','2017-10-26','2017-11-1','2017-11-5','2017-11-20','2017-12-20']
frstart_maduo = ['2017-10-25','2017-11-1','2017-11-5','2017-11-20']
frstart_maduo_da = pd.to_datetime(frstart_maduo)
# frend_maduo = ['2018-3-28','2018-4-18','2018-4-19','2018-4-22','2018-5-2','2018-5-16']
frend_maduo = ['2018-3-28','2018-4-19','2018-4-22','2018-5-2']
frend_maduo_da = pd.to_datetime(frend_maduo)
# frstart_maqu = ['2022-11-22','2022-11-30','2022-12-11','2022-12-23','2023-1-22']
frstart_maqu = ['2022-11-22','2022-12-11','2022-12-23','2023-1-22']
frstart_maqu_da = pd.to_datetime(frstart_maqu)
# frend_maqu = ['2023-3-13','2023-3-17','2023-3-20','2023-4-04','2023-3-18']
frend_maqu = ['2023-3-13','2023-3-20','2023-4-04','2023-3-18']
frend_maqu_da = pd.to_datetime(frend_maqu)

frstart_waerma = ['2023-11-06','2023-11-25','2023-12-21','2024-2-15']
frstart_waerma_da = pd.to_datetime(frstart_waerma)
frend_waerma = ['2024-4-09','2024-4-21','2024-5-28','2024-6-18']
frend_waerma_da = pd.to_datetime(frend_waerma)

#!###########################################################################################################################################
#!#################################下面就是正式画图部分代码了##################################################################################
alia = np.array(['(a)', '(b)', '(c)', '(d)'])
aliab = np.array(['(e)', '(f)', '(g)', '(h)'])
aliac = np.array(['(i)', '(j)', '(k)', '(l)'])
level = np.array([0.05, 0.2, 0.4, 0.8])
# 假设 data_obs_maduo 是你的观测数据 DataFrame
# 并且第三行第二列对应的深度索引是 2

# 定义异常值的时间范围
start_date = '2017-07-01'
end_date = '2017-09-01'

# 将异常值替换为 NaN
# data_obs_maduo.loc[start_date:end_date, 'Soil_VWC_80cm_AVG'] = np.nan


# plt.figure(figsize=(14, 12))
fig, ax = plt.subplots(4, 3, figsize=(20, 15))
plt.subplots_adjust(hspace=0.3)  # 设置子图之间的垂直间距为0.5

# 在创建 fig 之后，遍历所有的轴对象，设置刻度标签为加粗，并格式化刻度标签
for i in np.arange(4):
    for j in np.arange(3):
        # 设置刻度标签的字体大小
        ax[i,j].tick_params(axis='both', which='both', labelsize=12)
        # 设置ylim
        ax[i,j].set_ylim(0.0,0.55)
        # 获取 x 和 y 轴的刻度标签，并设置为加粗
        for label in ax[i,j].get_xticklabels() + ax[i,j].get_yticklabels():
            label.set_fontweight('bold')

for i in np.arange(4):
    if i <= 3:
        # 计算 data_original_humid_maqu[i] 和 data_obs_maqu.iloc[:,i] 的相关系数和均方根误差
        corr_original = np.corrcoef(data_original_humid_maqu[i], data_obs_maqu.iloc[:, i])[0, 1]
        rmse_original = np.sqrt(np.mean((data_original_humid_maqu[i] - data_obs_maqu.iloc[:, i]) ** 2))

        # 计算 data_modified_humid_maqu[i] 和 data_obs_maqu.iloc[:,i] 的相关系数和均方根误差
        corr_modified = np.corrcoef(data_modified_humid_maqu[i], data_obs_maqu.iloc[:, i])[0, 1]
        rmse_modified = np.sqrt(np.mean((data_modified_humid_maqu[i] - data_obs_maqu.iloc[:, i]) ** 2))

        ax[i,0].plot(times_maqu, data_original_humid_maqu[i], label='original_scheme', color='#3b658c', linewidth=3)
        ax[i,0].plot(times_maqu, data_modified_humid_maqu[i], label='new_scheme', color='#c82423', linewidth=3)
        ax[i,0].plot(times_maqu, data_obs_maqu.iloc[:,i], label='observation', color='black', linewidth=3)
        # 在每个子图上显示相关系数和均方根误差
        ax[i, 0].text(0.03, 0.97, f'r={corr_original:.2f}',
                transform=ax[i, 0].transAxes, fontsize=18, color='#3b658c', va='top', ha='left')
        ax[i, 0].text(0.03, 0.87, f'RMSE={rmse_original:.2f}',
                transform=ax[i, 0].transAxes, fontsize=18, color='#3b658c', va='top', ha='left')
        ax[i, 0].text(0.7, 0.97, f'r={corr_modified:.2f}',
                transform=ax[i, 0].transAxes, fontsize=18, color='#c82423', va='top', ha='left')
        ax[i, 0].text(0.7, 0.87, f'RMSE={rmse_modified:.2f}',
                transform=ax[i, 0].transAxes, fontsize=18, color='#c82423', va='top', ha='left')
        ax[i,0].axvspan(frstart_maqu_da[i], frend_maqu_da[i], facecolor='gray', alpha=0.2)
        ax[i,0].set_title(str(alia[i]) + ' ' + str(level[i]) + ' m', fontdict=dict(fontsize=20), loc='left')
        ax[i,0].set_ylabel(r'Soil Water Content $(m^3/m^3)$', fontdict=dict(fontsize=15))
        ax[i,0].set_xlim(times_maqu[0], times_maqu[-1])
        if i == 3: ax[i,0].legend(frameon=False, loc='center left', fontsize=14)
        ax[i,0].yaxis.set_tick_params(labelsize=15)
        ax[i,0].tick_params(axis='both',
        direction='in',
        length=5, width=2,
        )
        ax[i,0].spines[:].set_linewidth(2.5)

        # 计算 data_original_humid_waerma[i] 和 data_obs_waerma.iloc[:,i] 的相关系数和均方根误差
        corr_original = np.corrcoef(data_original_humid_waerma[i], data_obs_waerma.iloc[:, i])[0, 1]
        rmse_original = np.sqrt(np.mean((data_original_humid_waerma[i] - data_obs_waerma.iloc[:, i]) ** 2))

        # 计算 data_modified_humid_waerma[i] 和 data_obs_waerma.iloc[:,i] 的相关系数和均方根误差
        corr_modified = np.corrcoef(data_modified_humid_waerma[i], data_obs_waerma.iloc[:, i])[0, 1]
        rmse_modified = np.sqrt(np.mean((data_modified_humid_waerma[i] - data_obs_waerma.iloc[:, i]) ** 2))

        ax[i,1].plot(times_waerma, data_original_humid_waerma[i], label='original_scheme', color='#3b658c', linewidth=3)
        ax[i,1].plot(times_waerma, data_modified_humid_waerma[i], label='new_scheme', color='#c82423', linewidth=3)
        ax[i,1].plot(times_waerma, data_obs_waerma.iloc[:,i], label='observation', color='black', linewidth=3)
        # 在每个子图上显示相关系数和均方根误差，第二列的r往右移
        ax[i,1].text(0.03, 0.15, f'RMSE={rmse_original:.2f}',
                transform=ax[i,1].transAxes, fontsize=18, color='#3b658c', va='top', ha='left')
        ax[i,1].text(0.06, 0.25, f'r={corr_original:.2f}',
                transform=ax[i,1].transAxes, fontsize=18, color='#3b658c', va='top', ha='left')
        ax[i,1].text(0.7, 0.15, f'RMSE={rmse_modified:.2f}',
                transform=ax[i,1].transAxes, fontsize=18, color='#c82423', va='top', ha='left')
        ax[i,1].text(0.8, 0.25, f'r={corr_modified:.2f}',
                transform=ax[i,1].transAxes, fontsize=18, color='#c82423', va='top', ha='left')
        ax[i,1].axvspan(frstart_waerma_da[i], frend_waerma_da[i], facecolor='gray', alpha=0.2)
        ax[i,1].set_title(str(aliab[i]) + ' ' + str(level[i]) + ' m', fontdict=dict(fontsize=20), loc='left')
        ax[i,1].set_ylabel(r'Soil Water Content $(m^3/m^3)$', fontdict=dict(fontsize=15))
        ax[i,1].set_xlim(times_waerma[0], times_waerma[-1])
        ax[i,1].yaxis.set_tick_params(labelsize=15)
        ax[i,1].tick_params(axis='both',
        direction='in',
        length=5, width=2,
        )
        ax[i,1].spines[:].set_linewidth(2.5)

        # 计算 data_original_humid_maduo[i] 和 data_obs_maduo.iloc[:,i] 的相关系数和均方根误差
        corr_original_maduo = np.corrcoef(data_original_humid_maduo[i], data_obs_maduo.iloc[:, i])[0, 1]
        rmse_original_maduo = np.sqrt(np.mean((data_original_humid_maduo[i] - data_obs_maduo.iloc[:, i]) ** 2))

        # 计算 data_modified_humid_maduo[i] 和 data_obs_maduo.iloc[:,i] 的相关系数和均方根误差
        corr_modified_maduo = np.corrcoef(data_modified_humid_maduo[i], data_obs_maduo.iloc[:, i])[0, 1]
        rmse_modified_maduo = np.sqrt(np.mean((data_modified_humid_maduo[i] - data_obs_maduo.iloc[:, i]) ** 2))

        if i == 3 : data_obs_maduo.loc[start_date:end_date, 'Soil_VWC_80cm_AVG'] = np.nan

        ax[i,2].plot(times_maduo, data_original_humid_maduo[i], label='CLM5.0-original', color='#3b658c', linewidth=3)
        ax[i,2].plot(times_maduo, data_modified_humid_maduo[i], label='CLM5.0-modified', color='#c82423', linewidth=3)
        ax[i,2].plot(times_maduo, data_obs_maduo.iloc[:,i], label='obs', color='black', linewidth=3)
        ax[i,2].text(0.03, 0.97, f'r={corr_original_maduo:.2f}',
                    transform=ax[i,2].transAxes, fontsize=18, color='#3b658c', va='top', ha='left')
        ax[i,2].text(0.03, 0.87, f'RMSE={rmse_original_maduo:.2f}',
                    transform=ax[i,2].transAxes, fontsize=18, color='#3b658c', va='top', ha='left')
        ax[i,2].text(0.7, 0.97, f'r={corr_modified_maduo:.2f}',
                    transform=ax[i,2].transAxes, fontsize=18, color='#c82423', va='top', ha='left')
        ax[i,2].text(0.7, 0.87, f'RMSE={rmse_modified_maduo:.2f}',
                    transform=ax[i,2].transAxes, fontsize=18, color='#c82423', va='top', ha='left')

        ax[i,2].set_title(str(aliac[i]) + ' ' + str(level[i]) + ' m', fontdict=dict(fontsize=20), loc='left')
        ax[i,2].set_ylabel(r'Soil Water Content $(m^3/m^3)$', fontdict=dict(fontsize=15))
        ax[i,2].axvspan(frstart_maduo_da[i], frend_maduo_da[i], facecolor='gray', alpha=0.2)
        ax[i,2].yaxis.set_tick_params(labelsize=15)
        ax[i,2].set_xlim(times_maduo[0], times_maduo[-1])
        ax[i,2].tick_params(axis='both',
        direction='in',
        length=5, width=2,
        )
        ax[i,2].spines[:].set_linewidth(2.5)
    if i <= 2:
        ax[i,0].xaxis.set_ticklabels([])
        ax[i,1].xaxis.set_ticklabels([])
        ax[i,2].xaxis.set_ticklabels([])
    else:
        # 设置x轴刻度格式为%b，不旋转
        ax[i,0].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.setp(ax[i,0].get_xticklabels(), ha="center", fontsize=16)
        ax[i,0].set_xlabel('Time', fontdict=dict(weight='bold', fontsize=18))
        ax[i,1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.setp(ax[i,1].get_xticklabels(), ha="center", fontsize=16)
        ax[i,1].set_xlabel('Time', fontdict=dict(weight='bold', fontsize=18))
        ax[i,2].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.setp(ax[i,2].get_xticklabels(), ha="center", fontsize=16)
        ax[i,2].set_xlabel('Time', fontdict=dict(weight='bold', fontsize=18))

plt.savefig(fname=
'/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/Development_and_Validation_of_a_New_Soil_frost_heave_shceme/fig/after20241223/fig10-fig10-Comparison of simulated and observed soil moisture across various layers between the new and original schemes.png',
bbox_inches='tight', dpi=1200)

plt.show()
