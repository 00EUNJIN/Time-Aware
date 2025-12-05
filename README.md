# Time-Aware: Doc2Vec-based Time Series Forecasting

A PyTorch implementation of time series forecasting model that integrates Doc2Vec text embeddings with temporal data for climate prediction.

## Installation
```bash
pip install torch numpy pandas scikit-learn matplotlib tqdm einops transformers peft gensim
```

## Dataset Format

CSV file with columns:
- `date`: timestamp
- `temp`: temperature (or target variable)
- `text`: text description
- other features (optional)

Example:
```csv
date,temp,humidity,text
2014-01-01,37.9,57.4,"Weather description..."
```

## Quick Start
```bash
python run.py \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id climate_forecast \
    --model TimeAware \
    --data custom \
    --root_path ./dataset/ \
    --data_path climate_2014_2023_final.csv \
    --features S \
    --target temp \
    --seq_len 70 \
    --pred_len 70 \
    --train_epochs 10 \
    --batch_size 64
```

## Output

- Checkpoints: `checkpoints/`
- Predictions: `results/`
- Logs: `result_long_term_forecast.txt`

## Key Features

- Doc2Vec text encoding for temporal context
- GPT-2 with LoRA for efficient training
- Automatic Doc2Vec freezing at best checkpoint
- RevIN normalization

## Model Architecture

Input → RevIN → Doc2Vec Text Fusion → GPT-2 (LoRA) → Output