import pandas as pd
import re

# 文件路径
txt_file = "result_long_term_forecast_trend.txt"
excel_file = "result.xlsx"

# 1. 解析txt文件
with open(txt_file, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

pattern_model = re.compile(r"^(long_term_forecast_.*?)$")
pattern_metrics = re.compile(r"mse:(.*?),\s*mae:(.*?),.*trend_statistic:(.*)$")

records = []
for i in range(0, len(lines), 2):
    model_line = lines[i]
    metric_line = lines[i + 1] if i + 1 < len(lines) else ""
    m_model = pattern_model.match(model_line)
    m_metrics = pattern_metrics.search(metric_line)
    if m_model and m_metrics:
        model = m_model.group(1)
        mse, mae, trend = m_metrics.groups()

        # 提取 alpha 值，如果存在
        alpha_match = re.search(r'_(0\.\d+|1\.0|0)$', model_line)
        alpha = float(alpha_match.group(1)) if alpha_match else None

        records.append({
            "model": model,
            "alpha": alpha,
            "mse": float(mse),
            "mae": float(mae),
            "trend_statistic": float(trend)
        })

df_results = pd.DataFrame(records)
df_results[['mse', 'mae', 'trend_statistic']] = df_results[['mse','mae','trend_statistic']].round(3)

# 2. Excel 读取（双行表头 + 双列索引）
df_excel = pd.read_excel(excel_file, header=[0, 1], index_col=[0, 1])

# 3. 模型列映射
model_map = {
    'MICN': [("ours","MSE"), ("ours","MAE"), ("ours","MDA")],
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

# 5. 先处理 df_results，提取关键信息：dataset, pred_len, model_type
parsed_records = []
for _, row in df_results.iterrows():
    model_name = row['model'].lower()
    mse, mae, mda = row['mse'], row['mae'], row['trend_statistic']

    # (1) 数据集
    dataset = next((val for key,val in dataset_map.items() if key in model_name), None)
    if dataset is None:
        continue

    # (2) 预测长度
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

    parsed_records.append({
        "dataset": dataset,
        "pred_len": pred_len,
        "model_type": model_type,
        "alpha": row["alpha"],
        "mse": mse,
        "mae": mae,
        "mda": mda
    })

df_parsed = pd.DataFrame(parsed_records)

# 6. 对 (dataset, pred_len, model_type) 分组，选 MSE 最小的记录
df_best = df_parsed.loc[df_parsed.groupby(['dataset','pred_len','model_type'])['mse'].idxmin()]

# 7. 填充 Excel
for _, row in df_best.iterrows():
    dataset, pred_len, model_type = row['dataset'], row['pred_len'], row['model_type']
    mse, mae, mda = row['mse'], row['mae'], row['mda']

    if (dataset, pred_len) in df_excel.index:
        mse_col, mae_col, mda_col = model_map[model_type]
        df_excel.loc[(dataset, pred_len), mse_col] = mse
        df_excel.loc[(dataset, pred_len), mae_col] = mae
        df_excel.loc[(dataset, pred_len), mda_col] = mda

# 8. 自动计算 Avg 行
for dataset in df_excel.index.get_level_values(0).unique():
    if ('Avg' in df_excel.loc[dataset].index):
        for model_type, cols in model_map.items():
            mse_col, mae_col, mda_col = cols
            valid_rows = [l for l in forecast_lengths if (dataset,l) in df_excel.index]
            if valid_rows:
                avg_mse = round(df_excel.loc[(dataset, valid_rows), mse_col].mean(), 3)
                avg_mae = round(df_excel.loc[(dataset, valid_rows), mae_col].mean(), 3)
                avg_mda = round(df_excel.loc[(dataset, valid_rows), mda_col].mean(), 3)

                df_excel.loc[(dataset, 'Avg'), mse_col] = avg_mse
                df_excel.loc[(dataset, 'Avg'), mae_col] = avg_mae
                df_excel.loc[(dataset, 'Avg'), mda_col] = avg_mda

# 9. 保存结果
output_file = "result_filled_trend.xlsx"
df_excel.to_excel(output_file)
print(f"登记完成（已选取每组 MSE 最小的 alpha），已保存到 {output_file}")
