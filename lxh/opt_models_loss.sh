export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

python -u run.py \
  --model "GRU" \
  --loss "MixLoss" \
  --optimization 'TPE'

python -u run.py \
  --model "GRU" \
  --loss "MixLoss" \
  --optimization 'seq2seq'

  python -u run.py \
  --model "GRU" \
  --loss "MixLoss" \
  --optimization 'TCN'

  python -u run.py \
  --model "GRU" \
  --loss "MixLoss" \
  --optimization 'Timesnet'
