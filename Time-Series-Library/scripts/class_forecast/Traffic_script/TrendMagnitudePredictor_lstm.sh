export CUDA_VISIBLE_DEVICES=0

model_name=TrendMagnitudePredictor
# enc_in +1 程序内隐形多了mag一列

python -u run.py \
  --task_name classify_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom_cla_fore \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 863 \
  --dec_in 863 \
  --c_out 22 \
  --d_model 512 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 30 \
  --learning_rate 0.001 \
  --batch_size 64 \

