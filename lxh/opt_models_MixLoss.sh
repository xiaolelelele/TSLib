export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

python -u run.py \
  --model "GRU" \
  --loss "MixLoss" \
  --optimization 'TPE'

python -u run.py \
  --model "seq2seq" \
  --loss "MixLoss" \
  --optimization 'TPE'

  python -u run.py \
  --model "TCN" \
  --loss "MixLoss" \
  --optimization 'TPE'

  python -u run.py \
  --model "Timesnet" \
  --loss "MixLoss" \
  --optimization 'TPE'

python -u run.py \
  --model "GRU" \
  --loss "MixLoss" \
  --optimization 'BO'

python -u run.py \
  --model "seq2seq" \
  --loss "MixLoss" \
  --optimization 'BO'

  python -u run.py \
  --model "TCN" \
  --loss "MixLoss" \
  --optimization 'BO'

  python -u run.py \
  --model "Timesnet" \
  --loss "MixLoss" \
  --optimization 'BO'
