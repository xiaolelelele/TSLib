export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

python -u run.py \
  --task_name long_term_forecast_class \
  --is_training 0 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
 --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 512 \
  --d_ff 512 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --dropout 0.3 \
  --lradj 'cosine' \

