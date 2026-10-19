export CUDA_VISIBLE_DEVICES=3

model_name=iTransformer
seq_len=96

# for pred_len in 96 192 336 720;do
#   for alpha in 0.0 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 1.0;do
for pred_len in 192 336 720;do
  for alpha in  1 ;do
    python -u run.py \
      --task_name long_term_forecast_class \
      --is_training 1 \
      --root_path ./dataset/traffic/ \
      --data_path traffic.csv \
      --model_id traffic_$seq_len'_'$pred_len'_'$alpha \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 4 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --des 'Exp' \
      --d_model 512 \
      --d_ff 512 \
      --batch_size 16 \
      --learning_rate 0.001 \
      --itr 1 \
      --lradj 'cosine' \
      --trend_alpha $alpha
  done
done
