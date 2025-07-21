import numpy as np
import pandas as pd

# 读取真实值和预测值文件
forecast_file = "results/long_term_forecast_class_traffic_96_96_TimesNet_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df512_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/pred.npy"
true_file = "results/long_term_forecast_class_traffic_96_96_TimesNet_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df512_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/true.npy"

# 加载数据
fore = np.load(forecast_file, allow_pickle=True)
# 如果数据是三维的，可以选择最后一个时间步的数据（OT）
# fore_last = fore[:, :, -1]
# fore = pd.DataFrame(fore_last)
true = np.load(true_file, allow_pickle=True)
# true_last = true[:, :, -1]
# true = pd.DataFrame(true_last)

# 确保两个文件的行数和列数一致
assert fore.shape == true.shape, "两个文件的形状不一致！"
print(f"真实值数据形状: {true.shape}")
print(f"预测值数据形状: {fore.shape}")

def calculate_trend_agreement_2(test_data, forecast_data):
    """
    计算趋势一致率，并高效统计真实趋势中各类趋势（-1, 0, 1）的数量和比例。
    """

    assert test_data.shape == forecast_data.shape, "test 和 forecast 的 shape 必须一致"

    if not isinstance(test_data, np.ndarray):
        test_data = test_data.to_numpy()
        forecast_data = forecast_data.to_numpy()

    # 差分趋势
    test_trend = np.sign(np.diff(test_data, axis=1))
    forecast_trend = np.sign(np.diff(forecast_data, axis=1))

    # 一致匹配
    matches = (test_trend == forecast_trend)
    total_trends = matches.size
    matching_trends = np.sum(matches)

    # 高效趋势统计（-1, 0, 1）
    test_trend_flat = test_trend.ravel()
    counts = np.array([
        np.sum(test_trend_flat == -1),
        np.sum(test_trend_flat == 0),
        np.sum(test_trend_flat == 1)
    ])
    labels = np.array([-1, 0, 1])
    ratios = counts / total_trends

    print("趋势标签统计（真实值）:")
    for label, count, ratio in zip(labels, counts, ratios):
        print(f"  {label}: {count} ({ratio:.2%})")

    print("总趋势数:", total_trends)
    return matching_trends / total_trends



# 调用函数计算趋势一致比例
trend_agreement_ratio = calculate_trend_agreement_2(true, fore)
print(f"趋势一致的比例: {trend_agreement_ratio:.2%}")




