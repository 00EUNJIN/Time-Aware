# Time-Aware Multimodal Climate Forecasting: Aligning Numerical and Textual Signals for Context-Aware Long-Term Predictions

A PyTorch-based multimodal forecasting framework that integrates numerical climate time series with Doc2Vec-generated text embeddings.
The model performs time-aware fusion and uses GPT-2 with LoRA to generate robust long-term climate predictions.

## Installation
```bash
pip install torch numpy pandas scikit-learn matplotlib tqdm einops transformers peft gensim
```

## Dataset Format
A CSV file containing both numerical climate variables and daily weather reports.
Required columns:
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
Generated files during training:
- Checkpoints: `checkpoints/`
- Predictions: `results/`
- Logs: `result_long_term_forecast.txt`

## Key Features

- Doc2Vec-based text encoding for stable handling of variable-length reports
- GPT-2 with LoRA for lightweight and efficient multimodal forecasting
- Adaptive time-aware fusion between text and time series
- RevIN normalization to mitigate non-stationarity
- Automatic freezing of Doc2Vec at the best validation checkpoint

## Model Architecture

Input Time Series + Text Report → RevIN → Doc2Vec Text Encoder → Time-Aware Fusion Layer → GPT-2 (LoRA Fine-Tuned) → Forecast Output
