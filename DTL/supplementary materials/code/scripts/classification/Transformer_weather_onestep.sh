export CUDA_VISIBLE_DEVICES=0

model_name=Transformer

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --model_id weather \
  --model $model_name \
  --data weather_trend \
  --e_layers 3 \
  --batch_size 8 \
  --d_model 64 \
  --d_ff 128 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 10 \
  --patience 5