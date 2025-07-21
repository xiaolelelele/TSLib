import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取带标签的数据
input_path = 'data_new.csv'   # 请替换为你的文件路径
df = pd.read_csv(input_path)

# 2. 整体标签分布
label_counts = df['delta'].value_counts(normalize=True).sort_index()
plt.figure()
label_counts.plot(kind='bar')
plt.xlabel('Label')
plt.ylabel('Proportion')
plt.title('Overall Label Distribution')
plt.show()

# 3. 时间段内标签比例变化
time_col = 'date'

# 转换为 datetime 并设为索引
df[time_col] = pd.to_datetime(df[time_col])
df.set_index(time_col, inplace=True)

# 按天重采样，计算每天每个标签的比例
daily_props = df.groupby(pd.Grouper(freq='D'))['delta'] \
                .value_counts(normalize=True) \
                .unstack(fill_value=0) \
                .sort_index()


daily_props.plot(figsize=(10, 6))
plt.xlabel('Date')
plt.ylabel('Proportion')
plt.title('Daily Label Proportion Over Time')
plt.legend(title='Label')
plt.show()
