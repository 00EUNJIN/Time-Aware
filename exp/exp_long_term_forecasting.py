import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from datetime import datetime
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, visual
from utils.metrics import metric
from utils.cmLoss import cmLoss

warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.target_feature_idx = 0
        
        self.doc2vec_frozen = False     
        self.freeze_epoch = None        
        self.current_epoch = 0        
        
        print(f"Doc2Vec-based Time-aware Model Configuration:")
        print(f"  - Doc2Vec Model Path: {getattr(args, 'doc2vec_model_path', 'None')}")
        print(f"  - Doc2Vec Update Frequency: {getattr(args, 'doc2vec_update_frequency', 'epoch')}")
        print(f"  - Doc2Vec Vector Size: {getattr(args, 'doc2vec_vector_size', args.d_model)}")
        
        self.monitor = None

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args, self.device).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, vali_test=False):
        data_set, data_loader = data_provider(self.args, flag, vali_test)
        return data_set, data_loader
    
    def _get_validation_data(self, train_data):
        try:
            vali_data, vali_loader = self._get_data(flag='val')
            
            if hasattr(vali_data, '__len__') and hasattr(train_data, '__len__'):
                if len(vali_data) == len(train_data):
                    print("Warning: Validation dataset same size as train - splitting train data...")
                    return self._create_validation_from_train(train_data)
            
            if len(vali_loader) == 0 or len(vali_loader) == 1:
                print("Warning: Invalid validation loader. Creating from train data...")
                return self._create_validation_from_train(train_data)
            
            return vali_data, vali_loader
            
        except Exception as e:
            print(f"Error creating validation data: {e}")
            return self._create_validation_from_train(train_data)
    
    def _create_validation_from_train(self, train_data):
        from torch.utils.data import DataLoader, Subset
        
        train_size = len(train_data)
        val_size = max(1, train_size // 5)
        
        train_indices = list(range(train_size - val_size))
        val_indices = list(range(train_size - val_size, train_size))
        
        print(f"Splitting train data: {len(train_indices)} train + {len(val_indices)} validation")
        
        val_subset = Subset(train_data, val_indices)
        
        val_batch_size = min(self.args.batch_size, len(val_subset))
        vali_loader = DataLoader(
            val_subset,
            batch_size=val_batch_size,
            shuffle=False,
            drop_last=False
        )
        
        return val_subset, vali_loader
    
    def _select_optimizer(self):
        if hasattr(self.model, 'get_optimizer_params'):
            try:
                param_groups = self.model.get_optimizer_params(self.args.learning_rate)
                print(f"Time-aware model parameter groups: {len(param_groups)} groups")
                
                model_optim = optim.AdamW(param_groups, lr=self.args.learning_rate)
                loss_optim = optim.AdamW(param_groups, lr=self.args.learning_rate)
                
                return model_optim, loss_optim
                
            except Exception as e:
                print(f"Optimizer setup failed: {e}, using default")
        
        param_dict = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if p.requires_grad and (
                        "q_proj" in n or "k_proj" in n or "v_proj" in n or
                        "o_proj" in n or "gate_proj" in n or 
                        "up_proj" in n or "down_proj" in n
                    )
                ],
                "lr": 1e-4,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if p.requires_grad and not (
                        "q_proj" in n or "k_proj" in n or "v_proj" in n or
                        "o_proj" in n or "gate_proj" in n or 
                        "up_proj" in n or "down_proj" in n
                    )
                ],
                "lr": self.args.learning_rate,
            },
        ]

        model_optim = optim.AdamW([param_dict[1]], lr=self.args.learning_rate)
        loss_optim = optim.AdamW([param_dict[0]], lr=self.args.learning_rate)

        return model_optim, loss_optim

    def freeze_doc2vec_callback(self):
        if not self.doc2vec_frozen and hasattr(self.model, 'in_layer'):
            try:
                self.model.in_layer.freeze_doc2vec()
                self.doc2vec_frozen = True
                self.freeze_epoch = self.current_epoch
                print(f"Doc2Vec FROZEN at BEST MODEL (epoch {self.freeze_epoch})")
            except Exception as e:
                print(f"Doc2Vec freeze error: {e}")
                
    def _select_criterion(self):
        criterion = cmLoss(self.args.task_loss, self.args.task_name)
        return criterion

    def safe_nan_check(self, tensor, name="tensor"):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"Warning: NaN/Inf detected in {name}")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        return tensor

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_validation_data(train_data)
        test_data, test_loader = self._get_data(flag='test', vali_test=True)

        if hasattr(vali_data, '__len__') and len(vali_data) == len(train_data):
            from torch.utils.data import DataLoader, Subset
            train_size = len(train_data)
            val_size = max(1, train_size // 5)
            train_indices = list(range(train_size - val_size))
            
            train_subset = Subset(train_data, train_indices)
            train_loader = DataLoader(
                train_subset,
                batch_size=self.args.batch_size,
                shuffle=True,
                drop_last=True
            )

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, 
            verbose=True,
            best_model_callback=self.freeze_doc2vec_callback 
        )
        
        model_optim, loss_optim = self._select_optimizer()
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        scheduler = CosineAnnealingWarmRestarts(model_optim, T_0=3, T_mult=1, eta_min=1e-6)
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            self.current_epoch = epoch
            train_losses = []
            task_losses = []

            self.model.train()
            epoch_time = time.time()
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.args.train_epochs}')
            
            for i, batch in enumerate(train_bar):
                model_optim.zero_grad()
                loss_optim.zero_grad()

                if len(batch) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, text_data = batch
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    text_data = None

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y[:, :, self.target_feature_idx:self.target_feature_idx+1].float().to(self.device)
                
                batch_x = self.safe_nan_check(batch_x, "batch_x")
                batch_y = self.safe_nan_check(batch_y, "batch_y")
                
                if text_data is not None:
                    outputs_dict = self.model(batch_x, text_data)
                else:
                    outputs_dict = self.model(batch_x)
                
                for key, value in outputs_dict.items():
                    if torch.is_tensor(value):
                        outputs_dict[key] = self.safe_nan_check(value, f"outputs_{key}")
                
                loss_result = criterion(outputs_dict, batch_y)
                
                if isinstance(loss_result, tuple):
                    total_loss, task_loss, _, _ = loss_result
                    total_loss = self.safe_nan_check(total_loss, "total_loss")
                    task_loss = self.safe_nan_check(task_loss, "task_loss")
                    
                    task_losses.append(task_loss.item() if torch.is_tensor(task_loss) else task_loss)
                    loss = total_loss
                else:
                    loss = loss_result
                    loss = self.safe_nan_check(loss, "loss")

                train_losses.append(loss.item())

                if (i + 1) % 100 == 0:
                    train_bar.set_postfix(loss=loss.item())

                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            param.grad.zero_()
                
                model_optim.step()
                loss_optim.step()
                scheduler.step()

            print("Epoch: {} cost time: {:.2f}s".format(epoch + 1, time.time() - epoch_time))
            train_loss_avg = np.average(train_losses)

            if hasattr(self.model, 'in_layer') and hasattr(self.model.in_layer, 'on_epoch_end'):
                try:
                    self.model.in_layer.on_epoch_end(epoch)
                except Exception:
                    pass

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss_avg:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            
            if np.isnan(vali_loss):
                break
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        best_model_path = os.path.join(path, 'best_complete_checkpoint.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'doc2vec_model' in checkpoint and hasattr(self.model, 'in_layer'):
                self.model.in_layer.doc2vec_model = checkpoint['doc2vec_model']
                if 'doc2vec_metadata' in checkpoint:
                    metadata = checkpoint['doc2vec_metadata']
                    self.model.in_layer.doc2vec_trained = metadata['trained']
                    self.model.in_layer.doc2vec_frozen = metadata['frozen'] 
                    self.model.in_layer.text_buffer = metadata['text_buffer']
                    self.model.in_layer.current_epoch = metadata['current_epoch']
                    self.model.in_layer.last_update_epoch = metadata['last_update_epoch']

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        all_preds = []
        all_trues = []

        self.model.eval()

        with torch.no_grad():
            vali_bar = tqdm(vali_loader, desc='Validation')
            
            for i, batch in enumerate(vali_bar):
                try:
                    if len(batch) == 5:
                        batch_x, batch_y, batch_x_mark, batch_y_mark, text_data = batch
                    else:
                        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                        text_data = None
                        
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y[:, :, self.target_feature_idx:self.target_feature_idx+1].float()

                    batch_x = self.safe_nan_check(batch_x, "vali_batch_x")
                    batch_y = self.safe_nan_check(batch_y, "vali_batch_y")

                    if text_data is not None:
                        outputs = self.model(batch_x, text_data)
                    else:
                        outputs = self.model(batch_x)

                    if 'outputs_text' not in outputs:
                        continue
                        
                    outputs_ensemble = outputs['outputs_text']
                    
                    if torch.isnan(outputs_ensemble).any() or torch.isinf(outputs_ensemble).any():
                        continue
                    
                    outputs_ensemble = outputs_ensemble[:, -self.args.pred_len:, :]
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                    pred = outputs_ensemble.detach().cpu().numpy()
                    true = batch_y.detach().cpu().numpy()

                    if np.isnan(pred).any() or np.isinf(pred).any():
                        continue
                        
                    if np.isnan(true).any() or np.isinf(true).any():
                        continue

                    all_preds.append(pred)
                    all_trues.append(true)
                    
                except Exception as e:
                    continue

        self.model.train()

        if len(all_preds) == 0:
            return float('nan')
        
        try:
            all_preds = np.array(all_preds)
            all_trues = np.array(all_trues)

            all_preds = all_preds.reshape(-1, all_preds.shape[-2], all_preds.shape[-1])
            all_trues = all_trues.reshape(-1, all_trues.shape[-2], all_trues.shape[-1])
            
            mae, mse, rmse, mape, mspe = metric(all_preds, all_trues)
            
            return float(mse) 
            
        except Exception as e:
            return float('nan')

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('Loading model...')
            
            complete_checkpoint_path = os.path.join('./checkpoints/' + setting, 'best_complete_checkpoint.pth')
            
            if os.path.exists(complete_checkpoint_path):
                checkpoint = torch.load(complete_checkpoint_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                if 'doc2vec_model' in checkpoint and hasattr(self.model, 'in_layer'):
                    self.model.in_layer.doc2vec_model = checkpoint['doc2vec_model']
                    
                    if 'doc2vec_metadata' in checkpoint:
                        metadata = checkpoint['doc2vec_metadata']
                        self.model.in_layer.doc2vec_trained = metadata['trained']
                        self.model.in_layer.doc2vec_frozen = metadata['frozen']
                    
                    print(f"Complete model loaded")

        preds = []
        trues = []

        import hashlib
        if len(setting) > 10:
            setting_hash = hashlib.md5(setting.encode()).hexdigest()[:8]
            short_setting = f"test_{setting_hash}"
        else:
            short_setting = f"{setting}"

        folder_path = './test_results/' + short_setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            test_bar = tqdm(test_loader, desc='Testing')
            
            for i, batch in enumerate(test_bar):
                if len(batch) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, text_data = batch
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    text_data = None
                    
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y[:, :, self.target_feature_idx:self.target_feature_idx+1].float().to(self.device)

                if text_data is not None:
                    outputs = self.model(batch_x, text_data)
                else:
                    outputs = self.model(batch_x)

                outputs_ensemble = outputs['outputs_text']
                
                outputs_ensemble = outputs_ensemble[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]

                pred = outputs_ensemble.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
                
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    input_temp = input[:, :, self.target_feature_idx:self.target_feature_idx+1]
                    
                    if test_data.scale and self.args.inverse:
                        shape = input_temp.shape
                        input_temp = test_data.inverse_transform(input_temp.squeeze(0)).reshape(shape)
                    
                    gt = np.concatenate((input_temp[0, :, 0], true[0, :, 0]), axis=0)
                    pd = np.concatenate((input_temp[0, :, 0], pred[0, :, 0]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        folder_path = './results/' + short_setting + '/' 
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        
        result_text = f"Time-Aware Forecasting Model\n"
        result_text += f"mse:{mse}, mae:{mae}\n"
        
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "\n")
        f.write(result_text)
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return




        
