export CUDA_VISIBLE_DEVICES=2

model_name=DLinear
seq_len=96

for pred_len in 96 192 336 720;do
  for alpha in 0.0 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 ;do
    python -u run.py \
      --task_name long_term_forecast_class \
      --is_training 1 \
      --root_path ./dataset/electricity/ \
      --data_path electricity.csv \
      --model_id ECL_$seq_len'_'$pred_len'_'$alpha \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --itr 1 \
      --learning_rate 0.001 \
      --train_epochs 20 \
      --patience 3 \
      --lradj 'cosine' \
      --trend_alpha $alpha
  done
done