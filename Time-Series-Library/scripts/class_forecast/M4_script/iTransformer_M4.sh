export CUDA_VISIBLE_DEVICES=1

model_name=iTransformer
#   for alpha in 0.0 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 1.0;do
for alpha in 0.0 0.2  0.4  0.6  0.8 0.9 0.99 ;do
python -u run.py \
  --task_name short_term_forecast_class \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Monthly' \
  --model_id m4_Monthly_$alpha \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 512 \
  --des 'Exp' \
  --itr 1 \
  --loss 'SMAPE' \
  --train_epochs 20 \
  --patience 3 \
  --learning_rate 0.001\
  --lradj 'cosine' \
  --trend_alpha $alpha

python -u run.py \
  --task_name short_term_forecast_class \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Yearly' \
  --model_id m4_Yearly_$alpha  \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 512 \
  --des 'Exp' \
  --itr 1 \
  --loss 'SMAPE' \
  --train_epochs 20 \
  --patience 3 \
  --learning_rate 0.001\
  --lradj 'cosine' \
  --trend_alpha $alpha

python -u run.py \
  --task_name short_term_forecast_class \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Quarterly' \
  --model_id m4_Quarterly_$alpha \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 512 \
  --des 'Exp' \
  --itr 1 \
  --loss 'SMAPE' \
  --train_epochs 20 \
  --patience 3 \
  --learning_rate 0.001\
  --lradj 'cosine' \
  --trend_alpha $alpha

python -u run.py \
  --task_name short_term_forecast_class \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Weekly' \
  --model_id m4_Weekly_$alpha \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 512 \
  --des 'Exp' \
  --itr 1 \
  --loss 'SMAPE' \
  --train_epochs 20 \
  --patience 3 \
  --learning_rate 0.001\
  --lradj 'cosine' \
  --trend_alpha $alpha

python -u run.py \
  --task_name short_term_forecast_class \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Daily' \
  --model_id m4_Daily_$alpha \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 512 \
  --des 'Exp' \
  --itr 1 \
  --loss 'SMAPE' \
  --train_epochs 20 \
  --patience 3 \
  --learning_rate 0.001\
  --lradj 'cosine' \
  --trend_alpha $alpha

python -u run.py \
  --task_name short_term_forecast_class \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Hourly' \
  --model_id m4_Hourly_$alpha \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 512 \
  --des 'Exp' \
  --itr 1 \
  --loss 'SMAPE' \
  --train_epochs 20 \
  --patience 3 \
  --learning_rate 0.001\
  --lradj 'cosine' \
  --trend_alpha $alpha

done