import pandas as pd
import re

# 文件路径
txt_file = "result_long_term_forecast.txt"
excel_file = "result.xlsx"

# 1. 解析txt文件
with open(txt_file, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

pattern_model = re.compile(r"^(long_term_forecast_.*?)$")
pattern_metrics = re.compile(r"mse:(.*?),\s*mae:(.*?),.*trend_statistic:(.*)$")

records = []

for i in range(0, len(lines), 2):  # 每两行一条记录
    model_line = lines[i]
    metric_line = lines[i + 1] if i + 1 < len(lines) else ""
    m_model = pattern_model.match(model_line)
    m_metrics = pattern_metrics.search(metric_line)
    if m_model and m_metrics:
        model = m_model.group(1)
        mse, mae, trend = m_metrics.groups()
        records.append({
            "model": model,
            "mse": float(mse),
            "mae": float(mae),
            "trend_statistic": float(trend)
        })

df_results = pd.DataFrame(records)
df_results[['mse', 'mae', 'trend_statistic']] = df_results[['mse','mae','trend_statistic']].round(3)

# 2. 读取Excel（双行表头 + 双列行索引）
df_excel = pd.read_excel(excel_file, header=[0, 1], index_col=[0, 1])

# 3. 模型列映射
model_map = {
    'MICN': [("MICN","MSE"), ("MICN","MAE"), ("MICN","MDA")],
    'iTransformer': [("iTransformer","MSE"), ("iTransformer","MAE"), ("iTransformer","MDA")],
    'FreTS': [("FreTS","MSE"), ("FreTS","MAE"), ("FreTS","MDA")],
    'TimesNet': [("TimesNet","MSE"), ("TimesNet","MAE"), ("TimesNet","MDA")],
    'DLinear': [("DLinear","MSE"), ("DLinear","MAE"), ("DLinear","MDA")],
    'FEDformer': [("FEDformer","MSE"), ("FEDformer","MAE"), ("FEDformer","MDA")],
    'TCN': [("TCN","MSE"), ("TCN","MAE"), ("TCN","MDA")],
}

# 4. 数据集映射
dataset_map = {
    'ettm1': 'ETTm1',
    'ettm2': 'ETTm2',
    'etth1': 'ETTh1',
    'etth2': 'ETTh2',
    'ecl': 'ECL',
    'traffic': 'Traffic',
    'weather': 'Weather',
    'exchange': 'Exchange'
}

forecast_lengths = [96, 192, 336, 720]

# 5. 填充Excel
for _, row in df_results.iterrows():
    model_name = row['model'].lower()
    mse = row['mse']
    mae = row['mae']
    mda = row['trend_statistic']

    # (1) 数据集
    dataset = next((val for key,val in dataset_map.items() if key in model_name), None)
    if dataset is None:
        continue

    # (2) 预测长度匹配（修正后的安全处理）
    length_match = re.search(r'_pl(\d+)_', model_name)
    if length_match:
        pred_len = int(length_match.group(1))
    else:
        length_match = re.search(r'_(\d{2,3})_(\d{2,3})_', model_name)
        pred_len = int(length_match.group(2)) if length_match else None

    if pred_len not in forecast_lengths:
        continue

    # (3) 模型类型
    model_type = next((key for key in model_map if key.lower() in model_name), None)
    if model_type is None:
        continue

    # (4) 填入Excel
    if (dataset, pred_len) in df_excel.index:
        mse_col, mae_col, mda_col = model_map[model_type]
        df_excel.loc[(dataset, pred_len), mse_col] = mse
        df_excel.loc[(dataset, pred_len), mae_col] = mae
        df_excel.loc[(dataset, pred_len), mda_col] = mda

# 6. 自动计算 Avg 行
for dataset in df_excel.index.get_level_values(0).unique():
    if ('Avg' in df_excel.loc[dataset].index):
        for model_type, cols in model_map.items():
            mse_col, mae_col, mda_col = cols
            # 只对96/192/336/720求平均
            valid_rows = [l for l in forecast_lengths if (dataset,l) in df_excel.index]
            if valid_rows:
                avg_mse = round(df_excel.loc[(dataset, valid_rows), mse_col].mean(), 3)
                avg_mae = round(df_excel.loc[(dataset, valid_rows), mae_col].mean(), 3)
                avg_mda = round(df_excel.loc[(dataset, valid_rows), mda_col].mean(), 3)

                df_excel.loc[(dataset, 'Avg'), mse_col] = avg_mse
                df_excel.loc[(dataset, 'Avg'), mae_col] = avg_mae
                df_excel.loc[(dataset, 'Avg'), mda_col] = avg_mda

# 7. 保存结果
output_file = "result_filled.xlsx"
df_excel.to_excel(output_file)
print(f"登记完成，包含Avg行，已保存到 {output_file}")
