import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# 读取csv文件
df = pd.read_csv('dataset/weather/weather.csv')  
# 只对数值型列做归一化

scaler = StandardScaler()
data_scaled = scaler.fit_transform(df.iloc[:,1:])

col_names = df.columns[1:]  
for name, mean, std in zip(col_names, scaler.mean_, scaler.scale_):
    print(f"{name}: 均值={mean}, 标准差={std}")

last_col = df.iloc[:, -1]
mean = last_col.mean()
std = last_col.std()
print(f"最后一列手动计算：均值={mean}, 标准差={std}")

arr = df.iloc[:, -1].values
# arr转为np.ndarray
arr = np.array(arr)
mean = np.mean(arr)
diff_square = (arr - mean) ** 2
# 3. 求和再除以(n-1)，最后开根号（样本标准差）
variance = diff_square.sum() / (len(arr))
std = np.sqrt(variance)
print(f"最后一列手动计算：均值={mean}, 标准差={std}")

# 找到最后一列最大的值
max_value = df.iloc[:, -1].max()
# 找到最后一列的最小值  并返回下标
min_value = df.iloc[:, -1].min()
min_index = df.iloc[:, -1].idxmin()
print(f"最后一列最小值：{min_value}, 下标：{min_index}")
print(f"最后一列最大值：{max_value}")