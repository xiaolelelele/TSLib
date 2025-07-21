import pandas as pd

# 1. 读取原始 CSV 文件
input_path = 'weather.csv'  # 请替换为你的文件路径
df = pd.read_csv(input_path)

# 2. 找到最后一列列名
last_col = df.columns[-1]

# 3. 计算与上一个时刻的差分（第一行差分设为 0）
diff = df[last_col].diff().fillna(0)

# 4. 根据差分打标签：大于→0，等于→1，小于→2
df['delta'] = diff.apply(lambda x: 0 if x > 0 else 1 if x == 0 else 2)

# 5. 保存到一个新的 CSV 文件
output_path = 'data_new.csv'  # 请替换为输出路径
df.to_csv(output_path, index=False)

print(f"已生成带标签的新文件：{output_path}")
