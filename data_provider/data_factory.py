from data_provider.data_loader import Dataset_Custom
from torch.utils.data import DataLoader
import torch

data_dict = {
    'custom': Dataset_Custom,
}

def timeaware_collate_fn(batch):
    seq_x = torch.stack([torch.FloatTensor(item[0]) if not isinstance(item[0], torch.Tensor) else item[0] for item in batch])
    seq_y = torch.stack([torch.FloatTensor(item[1]) if not isinstance(item[1], torch.Tensor) else item[1] for item in batch])
    seq_x_mark = torch.stack([torch.FloatTensor(item[2]) if not isinstance(item[2], torch.Tensor) else item[2] for item in batch])
    seq_y_mark = torch.stack([torch.FloatTensor(item[3]) if not isinstance(item[3], torch.Tensor) else item[3] for item in batch])
    
    text_data = [item[4] for item in batch if len(item) > 4 and item[4] is not None]
    if not text_data:
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    else:
        return seq_x, seq_y, seq_x_mark, seq_y_mark, text_data

def data_provider(args, flag, vali=False):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    
    if args.model == 'TimeAware':
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=args.percent if hasattr(args, 'percent') else 100
        )
        
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=timeaware_collate_fn
        )
        return data_set, data_loader
    
    return None, None