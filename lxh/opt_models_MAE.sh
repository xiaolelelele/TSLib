export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

python -u run.py \
  --model "GRU" \
  --loss "MAE" \
  --optimization 'TPE'

python -u run.py \
  --model "seq2seq" \
  --loss "MAE" \
  --optimization 'TPE'

  python -u run.py \
  --model "TCN" \
  --loss "MAE" \
  --optimization 'TPE'

  python -u run.py \
  --model "Timesnet" \
  --loss "MAE" \
  --optimization 'TPE'

python -u run.py \
  --model "GRU" \
  --loss "MAE" \
  --optimization 'BO'

python -u run.py \
  --model "seq2seq" \
  --loss "MAE" \
  --optimization 'BO'

  python -u run.py \
  --model "TCN" \
  --loss "MAE" \
  --optimization 'BO'

  python -u run.py \
  --model "Timesnet" \
  --loss "MAE" \
  --optimization 'BO'
