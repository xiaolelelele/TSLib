import pandas as pd

# 读取真实值和预测值文件
test_file = "dataset/m4/Yearly-test.csv"
forecast_file = "m4_results/TimesNet/Yearly_forecast.csv"

# 加载数据
test_data = pd.read_csv(test_file, header=None)
test_data = test_data.iloc[1:, 1:]  # 去掉第一列
# 读取预测值
forecast_data = pd.read_csv(forecast_file, header=None)
forecast_data = forecast_data.iloc[1:, :]  

# 确保两个文件的行数和列数一致
assert test_data.shape == forecast_data.shape, "两个文件的形状不一致！"
print(f"真实值数据形状: {test_data.shape}")
print(f"预测值数据形状: {forecast_data.shape}")

# 计算趋势一致的比例
def calculate_trend_agreement(test_data, forecast_data):
    total_trends = 0
    matching_trends = 0

    for i in range(test_data.shape[0]):  # 遍历每一行
        test_row = test_data.iloc[i].values
        forecast_row = forecast_data.iloc[i].values

        # 计算真实值和预测值的趋势
        test_trend = [1 if test_row[j] < test_row[j + 1] else -1 if test_row[j] > test_row[j + 1] else 0 for j in range(len(test_row) - 1)]
        forecast_trend = [1 if forecast_row[j] < forecast_row[j + 1] else -1 if forecast_row[j] > forecast_row[j + 1] else 0 for j in range(len(forecast_row) - 1)]

        # 比较趋势是否一致
        for t, f in zip(test_trend, forecast_trend):
            total_trends += 1
            if t == f:
                matching_trends += 1

    # 计算趋势一致的比例
    trend_agreement_ratio = matching_trends / total_trends
    print(total_trends)
    return trend_agreement_ratio

# 调用函数计算趋势一致比例
trend_agreement_ratio = calculate_trend_agreement(test_data, forecast_data)
print(f"趋势一致的比例: {trend_agreement_ratio:.2%}")