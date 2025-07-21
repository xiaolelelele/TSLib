export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

python -u run.py \
  --model "GRU" \
  --loss "MSE" \
  --optimization 'TPE'

python -u run.py \
  --model "seq2seq" \
  --loss "MSE" \
  --optimization 'TPE'

  python -u run.py \
  --model "TCN" \
  --loss "MSE" \
  --optimization 'TPE'

  python -u run.py \
  --model "Timesnet" \
  --loss "MSE" \
  --optimization 'TPE'

python -u run.py \
  --model "GRU" \
  --loss "MSE" \
  --optimization 'BO'

python -u run.py \
  --model "seq2seq" \
  --loss "MSE" \
  --optimization 'BO'

  python -u run.py \
  --model "TCN" \
  --loss "MSE" \
  --optimization 'BO'

  python -u run.py \
  --model "Timesnet" \
  --loss "MSE" \
  --optimization 'BO'
