#!/bin/bash

PYTHON_PATH="python"

RESULT_DIR="experiments"
mkdir -p $RESULT_DIR
SESSION_NAME="grid_search_$(date +%m%d_%H%M)"
RESULT_FILE="$RESULT_DIR/${SESSION_NAME}.txt"
CSV_FILE="$RESULT_DIR/${SESSION_NAME}.csv"

echo "experiment_id,dropout,learning_rate,gpt_layers,patience,batch_size,r,lora_alpha,mse,mae,time" > $CSV_FILE

echo "=== Grid Search Experiments ===" > $RESULT_FILE
echo "Start Time: $(date)" >> $RESULT_FILE
echo "===============================" >> $RESULT_FILE

exp_id=1

dropouts=(0.1)
learning_rates=(0.0001)
gpt_layers=(2)
patience_values=(10)
batch_sizes=(64)
r_values=(32)
lora_alphas=(64)

BASE_CMD="$PYTHON_PATH run.py \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id climate_forecast \
    --model CALF_PatchTST \
    --data custom \
    --root_path ./dataset/ \
    --data_path climate_2014_2023_final.csv \
    --features S \
    --target temp \
    --seq_len 70 \
    --label_len 0 \
    --pred_len 70 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --d_model 768 \
    --lora_dropout 0.1 \
    --train_epochs 10 \
    --use_gpu True \
    --doc2vec_vector_size 768 \
    --doc2vec_buffer_size 300 \
    --doc2vec_alpha 0.025 \
    --doc2vec_epochs 20"

for dropout in "${dropouts[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for gpt_layer in "${gpt_layers[@]}"; do
            for patience in "${patience_values[@]}"; do
                for batch_size in "${batch_sizes[@]}"; do
                    for r in "${r_values[@]}"; do
                        for lora_alpha in "${lora_alphas[@]}"; do
                            
                            echo "=== Experiment $exp_id ===" | tee -a $RESULT_FILE
                            echo "dropout=$dropout, lr=$lr, gpt_layers=$gpt_layer, patience=$patience, batch_size=$batch_size, r=$r, lora_alpha=$lora_alpha" | tee -a $RESULT_FILE
                            
                            start_time=$(date +%s)

                            $BASE_CMD \
                                --dropout $dropout \
                                --learning_rate $lr \
                                --gpt_layers $gpt_layer \
                                --patience $patience \
                                --batch_size $batch_size \
                                --r $r \
                                --lora_alpha $lora_alpha \
                                > temp_exp_${exp_id}.txt 2>&1
                            
                            end_time=$(date +%s)
                            duration=$((end_time - start_time))
                            
                            mse_mae=$(tail -20 temp_exp_${exp_id}.txt | grep -E "mse:|mae:" | tail -1)
                            mse=$(echo $mse_mae | grep -oP 'mse:\K[0-9.]+' || echo "N/A")
                            mae=$(echo $mse_mae | grep -oP 'mae:\K[0-9.]+' || echo "N/A")
                            
                            echo "Result: $mse_mae" | tee -a $RESULT_FILE
                            echo "Duration: ${duration}s" | tee -a $RESULT_FILE
                            echo "---" | tee -a $RESULT_FILE
                            
                            echo "$exp_id,$dropout,$lr,$gpt_layer,$patience,$batch_size,$r,$lora_alpha,$mse,$mae,$duration" >> $CSV_FILE
                            
                            exp_id=$((exp_id + 1))
                            
                        done
                    done
                done
            done
        done
    done
done

echo "=== All Experiments Completed ===" | tee -a $RESULT_FILE
echo "End Time: $(date)" | tee -a $RESULT_FILE
echo "Total Experiments: $((exp_id - 1))" | tee -a $RESULT_FILE

echo "" | tee -a $RESULT_FILE
echo "=== BEST RESULTS ===" | tee -a $RESULT_FILE
echo "Best MSE:" | tee -a $RESULT_FILE
tail -n +2 $CSV_FILE | sort -t',' -k9 -n | head -3 | tee -a $RESULT_FILE
echo "" | tee -a $RESULT_FILE
echo "Best MAE:" | tee -a $RESULT_FILE
tail -n +2 $CSV_FILE | sort -t',' -k10 -n | head -3 | tee -a $RESULT_FILE

echo ""
echo "Results saved to: $RESULT_FILE"
echo "CSV data saved to: $CSV_FILE"