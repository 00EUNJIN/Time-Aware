import torch
import torch.nn as nn

loss_dict = {
    "l1": nn.L1Loss(),
    "mse": nn.MSELoss(),
    "ce": nn.CrossEntropyLoss(),
}

class cmLoss(nn.Module):
    def __init__(self, task_loss, task_name): 
        super(cmLoss, self).__init__()
        self.task_loss = loss_dict[task_loss]
        self.task_name = task_name
        
    def forward(self, outputs, batch_y, **kwargs):
        outputs_text = outputs["outputs_text"]
        batch_y = batch_y.to(outputs_text.device)
        task_loss = self.task_loss(outputs_text, batch_y)
        return task_loss, task_loss, torch.tensor(0.0), torch.tensor(0.0)