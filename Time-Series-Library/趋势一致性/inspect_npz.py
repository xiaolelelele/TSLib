import numpy as np
import matplotlib.pyplot as plt

# 加载 .npz 文件
train_data = np.load('metrics.npy', allow_pickle=True)
# test_data = np.load('dataset/m4/test.npz', allow_pickle=True)
print(type(train_data))  # <class 'numpy.ndarray'>
print(train_data.shape)
# print(train_data[0,:])  # float64
print(train_data)
# # 查看包含的键
# print("Training keys:", train_data.files)
# print("Test keys:", test_data.files)

# # 提取时间序列列表
# train_series_list = train_data['series']
# test_series_list = test_data['series']

# print(f"Total series (train): {len(train_series_list)}")
# print(f"Total series (test): {len(test_series_list)}")

# 检查第一个序列
# idx = 0  # 可以修改为查看其他序列
# train_series = train_series_list[idx]
# test_series = test_series_list[idx]

# print(f"Series {idx} - Train length: {len(train_series)}")
# print(f"Series {idx} - Test length: {len(test_series)}")
# print(f"Forecast horizon (test - train): {len(test_series) - len(train_series)}")

# # 可视化对比
# plt.figure(figsize=(10, 5))
# plt.plot(range(len(train_series)), train_series, label='Training Data', marker='o')
# plt.plot(range(len(train_series), len(test_series)), test_series[-6:], label='Ground Truth (Future)', marker='x')
# plt.title(f"M4 Yearly Series {idx}")
# plt.xlabel('Time Step')
# plt.ylabel('Value')
# plt.legend()
# plt.show()
