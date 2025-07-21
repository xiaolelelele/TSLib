import pandas as pd

# 读入weather数据集
df = pd.read_csv('dataset/weather/weather.csv')

# 取最后一列
last_col = df.columns[-1]
# 计算差分
diff = df[last_col].diff()

# 构建标签
trend = diff.apply(lambda x: 0 if x < 0 else (2 if x > 0 else 1))

# 插入为倒数第二列，命名为'trend'
df.insert(len(df.columns) - 1, 'trend', trend)

# 保存新数据集
df.to_csv('dataset/weather/weather_trend.csv', index=False)