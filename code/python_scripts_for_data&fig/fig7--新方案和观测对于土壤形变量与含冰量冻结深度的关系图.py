import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from cftime import DatetimeNoLeap
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

# 设置字体和参数
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['axes.unicode_minus'] = False

def plot_ax(ax, time_index, temp_data, depth_source, depth_values, totice_data, dyndz_data, max_depth, title):
    """
    绘制单个子图，取决于温度数据的深度来源类型。
    """
    # 确定深度信息的来源
    if depth_source == 'model':
        depth_levels = temp_data['levgrnd'].values
    elif depth_source == 'observation':
        depth_levels = depth_values
    else:
        raise ValueError("depth_source must be either 'model' or 'observation'.")

    # 绘制0°C的等值线
    cs = ax.contour(
        time_index,
        depth_levels,
        temp_data.values.T,  # 这里需要将温度数据转置
        levels=[0],
        colors='#3b658c',
        linewidths=3
    )

    ax.set_ylim(0, max_depth)  # 设置y轴范围
    ax.set_xlim([time_index.min(), time_index.max()])  # 设置x轴范围
    ax.invert_yaxis()  # 翻转y轴
    
    # 只在c和e子图上显示xlabel
    if title in ['(c)', '(e)']:
        ax.set_xlabel('Time', fontsize=16)  # 设置x轴标签
    
    ax.set_ylabel('Soil Depth (m)', color='#3b658c', fontsize=16)  # 设置左侧y轴标签 蓝色
    ax.tick_params(axis='x', labelsize=16, width=3)  # 设置x轴刻度标签大小和宽度
    ax.tick_params(axis='y', labelcolor='#3b658c', labelsize=16, width=3)  # 设置y轴刻度标签颜色、大小和宽度
    # 使用 FuncFormatter 格式化 y 轴刻度标签
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))
    ax.set_title(title, fontsize=20, loc='left')  # 设置标题
    
    # 设置x轴刻度格式为%b，不旋转
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.setp(ax.get_xticklabels(), ha="center", fontsize=16)

    for spine in ax.spines.values():  # 设置spine的线条宽度
        spine.set_linewidth(3)
    ax.spines['left'].set_color('#3b658c')  # 设置left spine的颜色

    # 总冰含量条形图，双轴
    ax2 = ax.twinx()
    ax2.bar(time_index, totice_data, color='grey', alpha=0.6, label='Total Soil Ice')
    ax2.set_ylabel('Total Soil Ice (kg/m$^2$)', color='grey', fontsize=16)
    ax2.tick_params(axis='y', labelcolor='grey', labelsize=16, width=3)  # 增加width参数
    ax2.set_ylim(bottom=0.0,top=2.0)
    # 使用 FuncFormatter 格式化 y 轴刻度标签
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))
    ax2.spines['right'].set_color((0.5, 0.5, 0.5, 0.6))  # 设置spine的颜色为灰色
    ax2.spines['right'].set_linewidth(3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.setp(ax2.get_xticklabels(), ha="center", fontsize=16)

    # 土壤变形线图，第三个双轴
    ax3 = ax.twinx()
    ax3.plot(time_index, dyndz_data, color='#c82423', linewidth=3.0, label='Soil Deformation') # #c82423橙色
    ax3.spines['right'].set_position(('outward', 80))  # 将spine向外移动
    ax3.spines['right'].set_linewidth(3)
    ax3.set_ylabel('Soil Deformation (m)', color='#c82423', fontsize=16)
    ax3.tick_params(axis='y', labelcolor='#c82423', labelsize=16, width=3)  # 增加width参数
    ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.3f}'.format(y)))
    ax3.spines['right'].set_color('#c82423')
    ax3.set_ylim(bottom=0.0, top=0.03)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.setp(ax3.get_xticklabels(), ha="center", fontsize=16)
    if ax == ax5 : ax3.set_ylim(bottom=0.0, top=0.04)

    return cs, ax, ax2, ax3

def add_legends(ax, ax2, ax3):
    """
    为每个图形添加图例
    """
    ax.legend([Line2D([0], [0], color='#3b658c', lw=2.5)], ['0°C Isotherm'], loc='upper left', bbox_to_anchor=(0, 1.0), frameon=False, fontsize=15)
    ax2.legend(loc='upper left', bbox_to_anchor=(0., 0.9), frameon=False, fontsize=15)
    ax3.legend(loc='upper left', bbox_to_anchor=(0., 0.8), frameon=False, fontsize=15)


# 数据读取与处理
time_slice_maduo = slice(DatetimeNoLeap(2017, 9, 1), DatetimeNoLeap(2018, 9, 1)) # 设置切片时候的time值。
data_modified_maduo = xr.open_dataset(
'/Users/finn/Library/CloudStorage/OneDrive-Personal/share_clm5/model_output/2024.08/soil_deformation_study_based_on_CLM5.0/2024.08.13/soil_ft_defo_MADUO_20240813_dyndz_STEP20.clm2.h0.2012-09-01-00000.nc')
data_modified_maduo_temp = data_modified_maduo.TSOI.isel(lndgrid=0).sel(time=time_slice_maduo) - 273.15
data_modified_maduo_totice = data_modified_maduo.SOILICE.isel(lndgrid=0).sel(time=time_slice_maduo) / \
    (data_modified_maduo.DYNDZ.isel(lndgrid=0).sel(time=time_slice_maduo)) / 1000.0
data_modified_maduo_totice = data_modified_maduo_totice.sum(dim='levsoi')
data_modified_maduo_dyndz = data_modified_maduo.DYNDZ.isel(lndgrid=0).sel(time=time_slice_maduo).sum(dim='levsoi') - 12.9
data_modified_maduo_dyndz[data_modified_maduo_dyndz < 10**(-4)] = 0.0 #这里是将dyndz值小于e-4的值设为0.0

time_slice_maqu = slice(DatetimeNoLeap(2022, 9, 1), DatetimeNoLeap(2023, 6, 1))
data_modified_maqu = xr.open_dataset(
'/Users/finn/Library/CloudStorage/OneDrive-Personal/share_clm5/model_output/2024.08/soil_deformation_study_based_on_CLM5.0/2024.08.13/soil_ft_defo_MAQU_20240813_v2_dyndz_STEP20.clm2.h0.2022-06-01-00000.nc')
data_modified_maqu_temp = data_modified_maqu.TSOI.isel(lndgrid=0).sel(time=time_slice_maqu) - 273.15
data_modified_maqu_totice = data_modified_maqu.SOILICE.isel(lndgrid=0).sel(time=time_slice_maqu) / \
    (data_modified_maqu.DYNDZ.isel(lndgrid=0).sel(time=time_slice_maqu)) / 1000.0
data_modified_maqu_totice = data_modified_maqu_totice.sum(dim='levsoi')
data_modified_maqu_dyndz = data_modified_maqu.DYNDZ.isel(lndgrid=0).sel(time=time_slice_maqu).sum(dim='levsoi') - 12.9
data_modified_maqu_dyndz[data_modified_maqu_dyndz < 10**(-4)] = 0.0

time_slice_waerma = slice(DatetimeNoLeap(2023, 9, 1), DatetimeNoLeap(2024, 7, 1))
data_modified_waerma = xr.open_dataset(
'/Users/finn/Library/CloudStorage/OneDrive-Personal/share_clm5/model_output/2024-10/soil_defomation_new_scheme/soil_ft_defo_Waerma_202410_modified.clm2.h0.2022-07-01-00000.nc')
data_modified_waerma_temp = data_modified_waerma.TSOI.isel(lndgrid=0).sel(time=time_slice_waerma) - 273.15
data_modified_waerma_totice = data_modified_waerma.SOILICE.isel(lndgrid=0).sel(time=time_slice_waerma) / \
    (data_modified_waerma.DYNDZ.isel(lndgrid=0).sel(time=time_slice_waerma)) / 1000.0
data_modified_waerma_totice = data_modified_waerma_totice.sum(dim='levsoi')
data_modified_waerma_dyndz = data_modified_waerma.DYNDZ.isel(lndgrid=0).sel(time=time_slice_waerma).sum(dim='levsoi') - 12.9
data_modified_waerma_dyndz[data_modified_waerma_dyndz < 10**(-4)] = 0.0

#!#################################接下来这一部分是读取观测数据##############################################################################
def process_soil_data(file_path, start_date, end_date, depth_layers, temp_col_names, moisture_col_names, convert_to_decimal=False):
    # 读取并处理数据
    df = (
        pd.read_csv(file_path, na_values='NAN')
        .interpolate()  # 去除NAN值
        .assign(time=lambda df: pd.to_datetime(df['time'], dayfirst=True) - pd.Timedelta(hours=8))  # 转换时间格式并调整时区
        .set_index('time')  # 将时间列设置为索引
        .loc[start_date:end_date]  # 选择特定时间范围的数据
    )

    # 如果需要，将特定的列数据除以100
    if convert_to_decimal:
        df[moisture_col_names] = df[moisture_col_names] / 100

    # 定义水和冰的密度
    rho_liq = 1.0  # 水的密度，单位 g/cm³
    rho_ice = 0.917  # 冰的密度，单位 g/cm³

    # 将深度（单位：cm）转换为米
    depth_values_m = [int(d[:-2]) / 100 for d in depth_layers]

    def calculate_freezing_start_and_ice_content(df, temp_col, moisture_col, depth_m):
        freezing_start = None
        melting_start = None
        ice_content_volume = [0] * len(df)  # 初始化体积含冰量
        dyndz = [depth_m] * len(df)  # 初始化dyndz为depth_values_m

        for i in range(len(df)):
            # 检查温度是否连续三个读数低于 0
            if i >= 2 and df[temp_col].iloc[i] < 0 and df[temp_col].iloc[i-1] < 0 and df[temp_col].iloc[i-2] < 0:
                if freezing_start is None:
                    freezing_start = df.index[i-2]  # 设置冻结开始时间
                    initial_moisture = df[moisture_col].iloc[i-2]

            # 检查温度是否连续三个读数大于 0
            if i >= 2 and df[temp_col].iloc[i] > 0 and df[temp_col].iloc[i-1] > 0 and df[temp_col].iloc[i-2] > 0:
                if melting_start is None:
                    melting_start = df.index[i-2]  # 设置消融开始时间

            if freezing_start is not None:
                current_moisture = df[moisture_col].iloc[i]
                previous_ice_content = ice_content_volume[i-1] if i > 0 else 0

                if melting_start is not None and df.index[i] >= melting_start:
                    # 计算消融期间的体积含冰量
                    updated_ice_volume = previous_ice_content - rho_liq * (current_moisture - df[moisture_col].iloc[i-1]) / rho_ice
                else:
                    # 计算冻结期间的体积含冰量
                    updated_ice_volume = previous_ice_content + rho_liq * (df[moisture_col].iloc[i-1] - current_moisture) / rho_ice

                # 确保体积含冰量不为负
                ice_content_volume[i] = max(0, updated_ice_volume)

                # 检查并设置当前时间点的土壤含冰量为 0
                if melting_start is not None and current_moisture >= initial_moisture:
                    ice_content_volume[i] = 0

                # 计算 dyndz
                if ice_content_volume[i] > 0:
                    ctriice = ice_content_volume[i] - (rho_ice / rho_liq) * ice_content_volume[i]
                    if ice_content_volume[i] > previous_ice_content:
                        dyndz[i] = depth_m * (1 + ctriice)
                        dyndz[i] = min(dyndz[i], 1.5 * depth_m)  # 最大值约束
                    elif ice_content_volume[i] == previous_ice_content:
                        dyndz[i] = dyndz[i-1] if i > 0 else depth_m
                    else:
                        if df[temp_col].iloc[i] < 0:
                            dyndz[i] = dyndz[i-1] if i > 0 else depth_m
                        else:
                            if dyndz[i-1] > depth_m:
                                dyndz[i] = dyndz[i-1] - (dyndz[i-1] - depth_m) * 0.5
                            else:
                                dyndz[i] = depth_m
                else:
                    dyndz[i] = depth_m

        # 计算 dyndz 与 depth_m 的差值并确保结果非负
        dyndz_delta = [max(dz - depth_m, 0) for dz in dyndz]

        df[f'ice_content_volume_{moisture_col}'] = ice_content_volume
        df[f'dyndz_delta_{moisture_col}'] = dyndz_delta


    # 对每个土壤层应用函数
    for temp_col, moisture_col, depth_m in zip(temp_col_names, moisture_col_names, depth_values_m):
        calculate_freezing_start_and_ice_content(df, temp_col, moisture_col, depth_m)

    # 计算所有层数的 dyndz_delta 的和
    dyndz_sum = df[[f'dyndz_delta_{moisture_col}' for moisture_col in moisture_col_names]].sum(axis=1)

    # 计算所有层数的土壤含冰量之和
    ice_content_sum = df[[f'ice_content_volume_{moisture_col}' for moisture_col in moisture_col_names]].sum(axis=1)

    # 计算每日平均
    dyndz_sum_daily = dyndz_sum.resample('D').mean()
    soil_temp_avg_daily = df[temp_col_names].resample('D').mean()  # 直接对每个深度的温度进行每日平均
    ice_content_sum_daily = ice_content_sum.resample('D').mean()

    return soil_temp_avg_daily, ice_content_sum_daily, dyndz_sum_daily

# 保存每个文件的处理结果
results = []

# 例子：处理不同文件或不同时间范围的数据
files_and_ranges = [
    {
        'file_path': '/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/Development_and_Validation_of_a_New_Soil_frost_heave_shceme/fig/after20241008/data/fig6/2021.11-2023.09玛曲站加密观测土壤温湿度数据整理.csv',
        'start_date': '2022-9-1',
        'end_date': '2023-6-1',
        'depth_layers': ['5cm','10cm','20cm','30cm','40cm','50cm','60cm','80cm','100cm','120cm','140cm','160cm'],
        'temp_col_names': ['5cm_TSoil_Avg','10cm_TSoil_Avg','20cm_TSoil_Avg','30cm_TSoil_Avg','40cm_TSoil_Avg','50cm_TSoil_Avg','60cm_TSoil_Avg','80cm_TSoil_Avg','100cm_TSoil_Avg','120cm_TSoil_Avg','140cm_TSoil_Avg','160cm_TSoil_Avg'],
        'moisture_col_names': ['5cm_VWC_Avg','10cm_VWC_Avg','20cm_VWC_Avg','30cm_VWC_Avg','40cm_VWC_Avg','50cm_VWC_Avg','60cm_VWC_Avg','80cm_VWC_Avg','100cm_VWC_Avg','120cm_VWC_Avg','140cm_VWC_Avg','160cm_VWC_Avg'],
        'convert_to_decimal': False  # 不需要在这个数据集中进行数据转换
    },
    {
        'file_path': '/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/Development_and_Validation_of_a_New_Soil_frost_heave_shceme/fig/after20241008/data/fig6/Waerma-soil-EC-edit.csv',
        'start_date': '2023-9-1',
        'end_date': '2024-6-30',
        'depth_layers': ['5cm','10cm','20cm','40cm','60cm','80cm','100cm','120cm','140cm','160cm'],
        'temp_col_names': ['Soil_Tem_5cm_Avg','Soil_Tem_10cm_Avg','Soil_Tem_20cm_Avg','Soil_Tem_40cm_Avg','Soil_Tem_60cm_Avg','Soil_Tem_80cm_Avg','Soil_Tem_100cm_Avg','Soil_Tem_120cm_Avg','Soil_Tem_140cm_Avg','Soil_Tem_160cm_Avg'],
        'moisture_col_names': ['Soil_VWC_5cm_Avg','Soil_VWC_10cm_Avg','Soil_VWC_20cm_Avg','Soil_VWC_40cm_Avg','Soil_VWC_60cm_Avg','Soil_VWC_80cm_Avg','Soil_VWC_100cm_Avg','Soil_VWC_120cm_Avg','Soil_VWC_140cm_Avg','Soil_VWC_160cm_Avg'],
        'convert_to_decimal': True  # 需要在这个数据集中进行数据转换
    },
    {
        'file_path': '/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/Development_and_Validation_of_a_New_Soil_frost_heave_shceme/fig/after20241008/data/fig6/data_obs_maduo201108_201812.csv',
        'start_date': '2017-9-1',
        'end_date': '2018-9-1',
        'depth_layers': ['5cm','10cm','20cm','40cm','80cm','160cm','320cm'],
        'temp_col_names': ['Soil_T_5cm_Avg','Soil_T_10cm_Avg','Soil_T_20cm_Avg','Soil_T_40cm_Avg','Soil_T_80cm_Avg','Soil_T_160cm_Avg','Soil_T_320cm_Avg'],
        'moisture_col_names': ['VWC_5cm_Avg','VWC_10cm_Avg','VWC_20cm_Avg','VWC_40cm_Avg','VWC_80cm_Avg','VWC_160cm_Avg','VWC_320cm_Avg'],
        'convert_to_decimal': False  # 不需要在这个数据集中进行数据转换
    }
]

for config in files_and_ranges:
    result = process_soil_data(**config)
    results.append(result)

# 每个result是一个元组，包含(dyndz_sum_daily, soil_temp_avg_daily, ice_content_sum_daily)
# 你可以在这里用result去绘图，例如:
# dsoil_temp_avg_daily, ice_content_sum_daily, yndz_sum_daily  = results[0]
print(results[0][1])

#!##################################################################################################################################

# 创建时间索引
time_index_maduo = pd.date_range('2017-09-01', '2018-09-01', freq='D')
time_index_maqu = pd.date_range('2022-09-01', '2023-06-01', freq='D')
time_index_waerma = pd.date_range('2023-09-01', '2024-6-30', freq='D')

# 创建3行2列的子图
fig, axs = plt.subplots(3, 2, figsize=(20, 12))

# 以1D数组的形式引用子图
ax1 = axs[0, 0]  # 第1行，第1列
ax2 = axs[0, 1]  # 第1行，第2列
ax3 = axs[1, 0]  # 第2行，第1列
ax4 = axs[1, 1]  # 第2行，第2列
ax5 = axs[2, 0]  # 第3行，第1列
ax6 = axs[2, 1]  # 第3行，第2列

depth_values_obs_maqu = np.array([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.2,1.4,1.6])
depth_values_obs_waerma = np.array([5,10,20,40,60,80,100,120,140,160]) / 100.0
depth_values_obs_maduo = np.array([5,10,20,40,80,160,320]) / 100.0

# 绘制子图
cs1, ax1, ax12, ax13 = plot_ax(ax1, time_index_maqu, data_modified_maqu_temp, 'model', depth_values_obs_maqu, data_modified_maqu_totice, data_modified_maqu_dyndz, 2.0, '(a)')
cs2, ax2, ax22, ax23 = plot_ax(ax2, time_index_maqu, results[0][0], 'observation', depth_values_obs_maqu, results[0][1], results[0][2], 2.0, '(d)')
cs3, ax3, ax32, ax33 = plot_ax(ax3, time_index_waerma, data_modified_waerma_temp, 'model',depth_values_obs_waerma, data_modified_waerma_totice, data_modified_waerma_dyndz, 2.0, '(b)')
cs4, ax4, ax42, ax43 = plot_ax(ax4, time_index_waerma, results[1][0], 'observation', depth_values_obs_waerma, results[1][1], results[1][2], 2.0, '(e)')
cs5, ax5, ax52, ax53 = plot_ax(ax5, time_index_maduo, data_modified_maduo_temp, 'model', depth_values_obs_maduo,data_modified_maduo_totice, data_modified_maduo_dyndz, 4.5, '(c)')
# cs6, ax6, ax62, ax63 = plot_ax(ax6, time_index_maduo, results[2][0], 'observation', depth_values_obs_maduo, results[2][1], results[2][2], 4.5, '(f)')

# 在第6个子图位置添加文本
ax6.text(0.5, 0.5, ' ', fontsize=20, ha='center', va='center')
ax6.set_axis_off()


# 调整子图间距，包括设置水平间距
plt.subplots_adjust(wspace=1.0)  # wspace 参数决定水平间距，默认是0.2

# 添加图例
add_legends(ax1, ax12, ax13)
# add_legends(ax2, ax22, ax23)
# add_legends(ax3, ax32, ax33)
# add_legends(ax4, ax42, ax43)
# add_legends(ax5, ax52, ax53)
# add_legends(ax6, ax62, ax63)

# 计算 data_original_temp_maqu[i] 和 data_obs_maqu.iloc[:,i] 的相关系数和均方根误差
corr_maqu_totice = np.corrcoef(data_modified_maqu_totice.values, results[0][1].values)[0,1]
rmse_maqu_totice = np.sqrt(np.mean((data_modified_maqu_totice.values - results[0][1].values) ** 2))
print(corr_maqu_totice, rmse_maqu_totice)

corr_maqu_dyndz = np.corrcoef(data_modified_maqu_dyndz.values, results[0][2].values)[0,1]
rmse_maqu_dyndz = np.sqrt(np.mean((data_modified_maqu_dyndz.values - results[0][2].values) ** 2))
print(corr_maqu_dyndz, rmse_maqu_dyndz)

corr_waerma_totice = np.corrcoef(data_modified_waerma_totice.values, results[1][1].values)[0,1]
rmse_waerma_totice = np.sqrt(np.mean((data_modified_waerma_totice.values - results[1][1].values) ** 2))
print(corr_waerma_totice, rmse_waerma_totice)

corr_waerma_dyndz = np.corrcoef(data_modified_waerma_dyndz.values, results[1][2].values)[0,1]
rmse_waerma_dyndz = np.sqrt(np.mean((data_modified_waerma_dyndz.values - results[1][2].values) ** 2))
print(corr_waerma_dyndz, rmse_waerma_dyndz)

# 自动调整布局
fig.tight_layout()

# 保存并显示图形
plt.savefig(
    fname='/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/Development_and_Validation_of_a_New_Soil_frost_heave_shceme/fig/after20241223/fig7-Comparison of soil deformation, ice content, and freezing phases between the new scheme and the model based on observed data.png',
    bbox_inches='tight', dpi=1200
)
plt.show()
