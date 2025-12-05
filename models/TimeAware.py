import os
import re
import numpy as np
import torch
import torch.nn as nn
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from transformers import GPT2Model
from peft import LoraConfig, TaskType, get_peft_model
from layers.RevIN import RevIN

class Encoder_Text(nn.Module):
    def __init__(self, input_dim, hidden_dim=768, pred_len=3, max_seq_len=512, 
                 enable_positional_pattern=True, doc2vec_model_path=None, 
                 doc2vec_vector_size=768, update_frequency='epoch'):
        super(Encoder_Text, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.enable_positional_pattern = enable_positional_pattern
        self.max_seq_len = max_seq_len
        self.doc2vec_vector_size = doc2vec_vector_size
        self.fusion_alpha = nn.Parameter(torch.tensor(0.2))
        
        self.doc2vec_model_path = doc2vec_model_path
        self.update_frequency = update_frequency
        self.doc2vec_model = None
        self.doc2vec_trained = False
        self.doc2vec_frozen = False
        
        self.text_buffer = []
        self.buffer_size = 1000
        self.current_epoch = 0
        self.last_update_epoch = -1
        
        self.fallback_vector_generator = nn.Parameter(
            torch.randn(doc2vec_vector_size) * 0.01, requires_grad=False
        )
        
        if self.doc2vec_vector_size != hidden_dim:
            self.doc2vec_proj = nn.Linear(self.doc2vec_vector_size, hidden_dim)
        else:
            self.doc2vec_proj = None
        
        if enable_positional_pattern:
            self.position_embeddings = nn.Embedding(max_seq_len, hidden_dim)
            self.value_projection = nn.Linear(input_dim, hidden_dim)
            self.position_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
        else:
            self.linear = nn.Linear(input_dim, hidden_dim)
            
        self.text_to_timepoint_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.time_series_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self._initialize_doc2vec()
        
        self.cross_attention_weights = None

    def _initialize_doc2vec(self):
        try:
            if self.doc2vec_model_path and os.path.exists(self.doc2vec_model_path):
                print(f"Loading existing Doc2Vec model from {self.doc2vec_model_path}")
                self.doc2vec_model = Doc2Vec.load(self.doc2vec_model_path)
                self.doc2vec_trained = True
                self.doc2vec_frozen = True
                print(f"Doc2Vec model loaded and frozen!")
                self._update_fallback_vector()
            else:
                self.doc2vec_model = Doc2Vec(
                    vector_size=self.doc2vec_vector_size,
                    min_count=5,
                    epochs=20,
                    dm=1,
                    alpha=0.025,
                    min_alpha=0.00025,
                    negative=5,
                    hs=0,
                    workers=4,
                    window=5,
                    sample=1e-4
                )
                self.doc2vec_trained = False
                self.doc2vec_frozen = False
        except Exception as e:
            print(f"Error initializing Doc2Vec: {e}")
            self.doc2vec_model = Doc2Vec(
                vector_size=self.doc2vec_vector_size,
                min_count=2,
                epochs=20,
                dm=1,
                alpha=0.025,
                min_alpha=0.00025
            )
            self.doc2vec_trained = False
            self.doc2vec_frozen = False

    def _update_fallback_vector(self):
        if self.doc2vec_model and self.doc2vec_trained:
            try:
                if hasattr(self.doc2vec_model.dv, 'vectors') and len(self.doc2vec_model.dv.vectors) > 0:
                    mean_vector = np.mean(self.doc2vec_model.dv.vectors, axis=0)
                elif hasattr(self.doc2vec_model.wv, 'vectors') and len(self.doc2vec_model.wv.vectors) > 0:
                    mean_vector = np.mean(self.doc2vec_model.wv.vectors, axis=0)
                else:
                    mean_vector = np.zeros(self.doc2vec_vector_size)
                
                with torch.no_grad():
                    self.fallback_vector_generator.copy_(torch.from_numpy(mean_vector).float())
            except Exception as e:
                print(f"Failed to update fallback vector: {e}")

    def preprocess_text(self, text):
        if not text:
            return []
        
        if isinstance(text, list):
            text_parts = []
            for item in text:
                if item is not None:
                    text_parts.append(str(item).strip())
            if not text_parts:
                return []
            text = ' '.join(text_parts)
        else:
            text = str(text)
        
        if not text.strip():
            return []
        
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        tokens = text.split()
        tokens = [token for token in tokens if len(token) > 1]
        
        return tokens

    def collect_texts_for_training(self, texts):
        if not texts or self.doc2vec_frozen or self.update_frequency == 'never':
            return
        
        for text in texts:
            try:
                if text is None:
                    continue
                
                if isinstance(text, list):
                    if not text:
                        continue
                    text_parts = [str(item).strip() for item in text if item is not None]
                    if not text_parts:
                        continue
                    combined_text = ' '.join(text_parts)
                else:
                    combined_text = str(text)
                
                if combined_text.strip():
                    self.text_buffer.append(combined_text)
                    
                    if len(self.text_buffer) > self.buffer_size:
                        self.text_buffer = self.text_buffer[-self.buffer_size:]
            except Exception as e:
                continue

    def train_doc2vec_from_buffer(self, force_retrain=False):
        if self.doc2vec_frozen and not force_retrain:
            return
        
        if len(self.text_buffer) < 10:
            return
        
        try:
            print(f"Training Doc2Vec on {len(self.text_buffer)} collected texts...")
            
            tagged_docs = []
            for i, text in enumerate(self.text_buffer):
                tokens = self.preprocess_text(text)
                if tokens:
                    tagged_docs.append(TaggedDocument(words=tokens, tags=[f"doc_{i}"]))
            
            if not tagged_docs:
                return
            
            if not self.doc2vec_trained:
                self.doc2vec_model.build_vocab(tagged_docs)
                self.doc2vec_model.train(tagged_docs, total_examples=len(tagged_docs), epochs=20)
                self.doc2vec_trained = True
                print(f"Doc2Vec initial training completed!")
            else:
                try:
                    self.doc2vec_model.build_vocab(tagged_docs, update=True)
                    self.doc2vec_model.train(tagged_docs, total_examples=len(tagged_docs), epochs=10)
                    print(f"Doc2Vec model updated!")
                except Exception as e:
                    print(f"Doc2Vec update failed: {e}")
            
            self._update_fallback_vector()
            
            keep_size = min(200, len(self.text_buffer) // 2)
            self.text_buffer = self.text_buffer[-keep_size:]
            
        except Exception as e:
            print(f"Doc2Vec training error: {e}")

    def encode_texts_with_doc2vec(self, texts, device):
        if not texts or not any(texts):
            return torch.zeros(1, 1, self.hidden_dim, device=device)
        
        if self.training and self.update_frequency != 'never':
            self.collect_texts_for_training(texts)
        
        try:
            doc_vectors = []
            for text in texts:
                try:
                    if text is None:
                        vector = self.fallback_vector_generator.detach().cpu().numpy()
                        doc_vectors.append(vector)
                        continue
                    
                    if isinstance(text, list):
                        if not text:
                            vector = self.fallback_vector_generator.detach().cpu().numpy()
                            doc_vectors.append(vector)
                            continue
                        
                        text_parts = [str(item).strip() for item in text if item is not None]
                        if not text_parts:
                            vector = self.fallback_vector_generator.detach().cpu().numpy()
                            doc_vectors.append(vector)
                            continue
                            
                        combined_text = ' '.join(text_parts)
                    else:
                        combined_text = str(text)
                    
                    if not combined_text.strip():
                        vector = self.fallback_vector_generator.detach().cpu().numpy()
                        doc_vectors.append(vector)
                        continue
                    
                    tokens = self.preprocess_text(combined_text)
                    if tokens and self.doc2vec_trained:
                        vector = self.doc2vec_model.infer_vector(tokens)
                        doc_vectors.append(vector)
                    else:
                        vector = self.fallback_vector_generator.detach().cpu().numpy()
                        doc_vectors.append(vector)
                        
                except Exception as e:
                    vector = self.fallback_vector_generator.detach().cpu().numpy()
                    doc_vectors.append(vector)
            
            doc_vectors = np.array(doc_vectors)
            doc_embeddings = torch.from_numpy(doc_vectors).float().to(device)
            doc_embeddings = doc_embeddings.unsqueeze(1)
            
            if self.doc2vec_proj is not None:
                doc_embeddings = self.doc2vec_proj(doc_embeddings)
                
            return doc_embeddings
            
        except Exception as e:
            print(f"Doc2Vec encoding error: {e}")
            batch_size = len(texts)
            fallback_emb = self.fallback_vector_generator.unsqueeze(0).repeat(batch_size, 1)
            if self.doc2vec_proj is not None:
                fallback_emb = self.doc2vec_proj(fallback_emb.unsqueeze(1))
            else:
                fallback_emb = fallback_emb.unsqueeze(1)
            return fallback_emb.to(device)

    def on_epoch_end(self, epoch):
        self.current_epoch = epoch
        
        if (self.update_frequency == 'epoch' and 
            epoch != self.last_update_epoch and 
            len(self.text_buffer) >= 50):
            
            print(f"\nEpoch {epoch}: Updating Doc2Vec model...")
            self.train_doc2vec_from_buffer()
            self.last_update_epoch = epoch

    def freeze_doc2vec(self):
        if not self.doc2vec_frozen:
            self.doc2vec_frozen = True
            print("Doc2Vec model frozen")

    def time_aware_cross_modal_fusion(self, x_time, texts):
        B, L, D = x_time.shape
        
        if not texts or not any(texts):
            return x_time
            
        try:
            doc_embeddings = self.encode_texts_with_doc2vec(texts, x_time.device)
        
            enhanced_timepoints, attention_weights = self.text_to_timepoint_attn(
                query=x_time,
                key=doc_embeddings,
                value=doc_embeddings
            )
            self.cross_attention_weights = attention_weights.detach().cpu()

            alpha = torch.sigmoid(self.fusion_alpha)
            attended_output = 0.5 * x_time + 0.5 * enhanced_timepoints
        
            return attended_output
        
        except Exception as e:
            print(f"Doc2Vec fusion error: {e}")
            return x_time

    def create_positional_embeddings(self, x):
        B, L, _ = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.position_embeddings(positions)
        val_emb = self.value_projection(x)
        combined_input = torch.cat([pos_emb, val_emb], dim=-1)
        gate = self.position_gate(combined_input)
        return gate * pos_emb + (1 - gate) * val_emb

    def forward(self, x, texts=None):
        B, L, D = x.shape
        
        if self.enable_positional_pattern:
            x = self.create_positional_embeddings(x)
        else:
            x = self.linear(x)
        
        x = self.time_series_encoder(x)
        x_time = x
        
        if not texts:
            return x_time, x_time
        
        x_text = self.time_aware_cross_modal_fusion(x_time, texts)
        
        return x_time, x_text

class Model(nn.Module):
    def __init__(self, configs, device):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.task_name = configs.task_name
        self.target_feature_idx = 0
        self.revin = RevIN(num_features=1) 
        
        from transformers import GPT2Model
        self.gpt2_text = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        
        gpt_layers = getattr(configs, 'gpt_layers', 3)
        self.gpt2_text.h = self.gpt2_text.h[:gpt_layers]
        
        peft_config = LoraConfig(
            inference_mode=False, 
            r=getattr(configs, 'r', 8),
            lora_alpha=getattr(configs, 'lora_alpha', 16),
            lora_dropout=getattr(configs, 'lora_dropout', 0.1),
            target_modules=["c_attn"]
        )
        
        self.gpt2_text = get_peft_model(self.gpt2_text, peft_config)

        for i, (name, param) in enumerate(self.gpt2_text.named_parameters()):
            if 'wpe' in name or 'lora_' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        self.embedding_analysis_mode = False
        self.collected_embeddings = {
            'ts_vec': [],
            'doc_vec': [], 
            'weighted_vec': [],
            'labels': []
        }
        
        doc2vec_model_path = getattr(configs, 'doc2vec_model_path', None)
        doc2vec_vector_size = getattr(configs, 'doc2vec_vector_size', configs.d_model)
        doc2vec_update_frequency = getattr(configs, 'doc2vec_update_frequency', 'epoch')
        

        self.in_layer = Encoder_Text(
            1,
            hidden_dim=configs.d_model,
            pred_len=configs.pred_len,
            max_seq_len=getattr(configs, 'max_seq_len', 512),
            enable_positional_pattern=getattr(configs, 'enable_positional_pattern', True),
            doc2vec_model_path=doc2vec_model_path,
            doc2vec_vector_size=doc2vec_vector_size,
            update_frequency=doc2vec_update_frequency
        )
        
        self.out_layer = nn.Linear(configs.d_model, configs.pred_len)

        for layer in (self.gpt2_text, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()
    def get_optimizer_params(self, base_lr):
        gpt2_lora_params = []
        gpt2_other_params = []
        doc2vec_proj_params = []
        other_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'gpt2_text' in name:
                    if 'lora_' in name:
                        gpt2_lora_params.append(param)
                    else:
                        gpt2_other_params.append(param)
                elif 'in_layer.doc2vec_proj' in name:
                    doc2vec_proj_params.append(param)
                else:
                    other_params.append(param)
        
        param_groups = [
            {'params': other_params, 'lr': base_lr, 'name': 'model_params'},
            {'params': gpt2_lora_params, 'lr': base_lr * 0.5, 'name': 'gpt2_lora_params'},
            {'params': gpt2_other_params, 'lr': base_lr * 0.1, 'name': 'gpt2_other_params'},
            {'params': doc2vec_proj_params, 'lr': base_lr * 0.8, 'name': 'doc2vec_proj_params'}
        ]
        
        param_groups = [group for group in param_groups if len(group['params']) > 0]
        
        return param_groups

    def safe_nan_check(self, tensor):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=5.0, neginf=-5.0)
        tensor = torch.clamp(tensor, min=-10.0, max=10.0)
        return tensor

    def forecast(self, x, texts=None):
        B, L, M = x.shape

        temp_x = x[:, :, self.target_feature_idx:self.target_feature_idx+1]
        temp_x = self.safe_nan_check(temp_x)

        temp_x = self.revin(temp_x, 'norm')
        temp_x = self.safe_nan_check(temp_x)

        outputs_time1, outputs_text1 = self.in_layer(temp_x, texts)
        outputs_text1 = self.safe_nan_check(outputs_text1)

        gpt_output = self.gpt2_text(inputs_embeds=outputs_text1)
        outputs_text = gpt_output.last_hidden_state
        outputs_text = self.safe_nan_check(outputs_text)
        outputs_text += outputs_text1

        last_hidden = outputs_text[:, -1, :]
        outputs_text = self.out_layer(last_hidden)
        outputs_text = outputs_text.unsqueeze(-1)

        outputs_text = self.revin(outputs_text, 'denorm')
        outputs_text = self.safe_nan_check(outputs_text)

        return {'outputs_text': outputs_text}

    def forward(self, x, texts=None, mask=None):
        return self.forecast(x, texts)



