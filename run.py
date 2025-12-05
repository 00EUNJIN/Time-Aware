import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args
import random
import numpy as np
import time

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Time-Aware Forecasting Model')

    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, required=True, default=1)
    parser.add_argument('--model_id', type=str, required=True, default='test')
    parser.add_argument('--model', type=str, required=True, default='CALF_PatchTST')

    parser.add_argument('--data', type=str, required=True, default='custom')
    parser.add_argument('--root_path', type=str, default='./dataset/')
    parser.add_argument('--data_path', type=str, default='data.csv')
    parser.add_argument('--features', type=str, default='S')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='d')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--inverse', action='store_true', default=False)

    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')

    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--des', type=str, default='test')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--task_loss', type=str, default='mse')

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')

    parser.add_argument('--r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--gpt_layers', type=int, default=3)

    parser.add_argument('--percent', type=int, default=100)

    parser.add_argument('--doc2vec_model_path', type=str, default=None)
    parser.add_argument('--doc2vec_vector_size', type=int, default=768)
    parser.add_argument('--doc2vec_update_frequency', type=str, default='epoch')
    parser.add_argument('--doc2vec_buffer_size', type=int, default=1000)
    parser.add_argument('--doc2vec_min_count', type=int, default=2)
    parser.add_argument('--doc2vec_epochs', type=int, default=20)
    parser.add_argument('--doc2vec_dm', type=int, default=1, choices=[0, 1])
    parser.add_argument('--doc2vec_window', type=int, default=5)
    parser.add_argument('--doc2vec_alpha', type=float, default=0.025)
    parser.add_argument('--doc2vec_min_alpha', type=float, default=0.00025)
    parser.add_argument('--doc2vec_negative', type=int, default=5)

    parser.add_argument('--enable_positional_pattern', action='store_true', default=True)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--fusion_strength', type=float, default=0.5)

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print(f"\n📚 Doc2Vec Configuration:")
    print(f"  - Vector Size: {args.doc2vec_vector_size}")
    print(f"  - Update Frequency: {args.doc2vec_update_frequency}")
    print(f"  - Training Epochs: {args.doc2vec_epochs}")
    print(f"  - Algorithm: {'PV-DM' if args.doc2vec_dm == 1 else 'PV-DBOW'}")

    os.makedirs('./models/', exist_ok=True)
    print(f"\n📁 Doc2Vec models directory: ./models/")
    print('\nArgs in experiment:')
    print_args(args)

    Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            start_time = time.time()
            exp = Exp(args)
            
            setting = '{}_{}_{}_{}_ft{}_{}_sl{}_ll{}_pl{}_dm{}_eb{}_{}_gpt{}_doc2vec_v{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.target,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.embed,
                args.des, 
                args.gpt_layers,
                args.doc2vec_vector_size,
                args.doc2vec_update_frequency,
                ii)
            
            setting += f"_dm{args.doc2vec_dm}_buf{args.doc2vec_buffer_size}"

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            
            end_time = time.time()            
            total_time = end_time - start_time

            print("=" * 60)
            print(f"TOTAL TIME: {total_time:.2f}s ({total_time/60:.2f}min)")
            print("=" * 60)

            torch.cuda.empty_cache()

    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_eb{}_{}_gpt{}_doc2vec_v{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.embed,
            args.des, 
            args.gpt_layers,
            args.doc2vec_vector_size,
            args.doc2vec_update_frequency,
            ii)
        
        setting += f"_dm{args.doc2vec_dm}_buf{args.doc2vec_buffer_size}"

        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()