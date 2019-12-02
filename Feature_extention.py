
import pandas as pd

def cal_diff(df, sensor_name,diff_periods = 40):
    _id =1
    sensor_diff = []
    sensor_diff_temp = []
    for _id in set(df[id]):
        trainFD001_of_one_id =  df[df[id] == _id]
        s2 = pd.Series(trainFD001_of_one_id[sensor_name])
        sensor_diff_temp=s2.diff(periods=diff_periods)
        # 第0到39 应该是每一个值-第一个值
        for i in range(diff_periods):
            sensor_diff.append(s2.iloc[i]-s2.iloc[0])
        # 第40个值之后应该是每一个值-向前推40位的值
        for j in range (len(s2)-diff_periods):
            sensor_diff.append(sensor_diff_temp.iloc[diff_periods+j])
            
    return sensor_diff