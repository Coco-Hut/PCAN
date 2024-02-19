import torch
from torch import nn

# loss function
class Weighted_Square_Error(nn.Module):
    def __init__(self,alpha):
        super(Weighted_Square_Error, self).__init__()
        self.alpha = alpha # 0.95

    def forward(self,input,target):
        if not (target.size() == input.size()):
            raise ValueError(
            "Using a target size ({}) that is different to the input size ({}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.".format(target.size(), input.size()),
            
        )
        batch_size = input.shape[0]
        lenth = input.shape[1]
        alpha_tensor = torch.pow(self.alpha,torch.arange(0,lenth).to("cuda"))
        loss = torch.pow(torch.add(input,-target),2)
        loss = torch.sum(alpha_tensor*loss,dim = 1)
        return torch.sum(loss)
    
    
import torch
from torch import nn

# metrics
def Weighted_Mean_Absolute_Percentage_Error(input,target):
    abs_minus = torch.sum(torch.abs(input.detach()-target.detach()))
    y_sum = torch.sum(target.detach())

    return abs_minus/y_sum

def Mean_Absolute_Percentage_Error(input,target):
    mape_sum = torch.sum(torch.abs(input.detach()-target.detach())/(target.detach()+0.00001))
    all =  input.shape[0]*input.shape[1]

    return mape_sum/all

def ACC(input,target):
    abs_minus = torch.sum(torch.abs(input.detach()-target.detach()))
    y_sum = torch.sum(target.detach())

    return 1-abs_minus/y_sum