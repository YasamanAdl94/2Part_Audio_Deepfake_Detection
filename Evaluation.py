import argparse
import sys
import os
from data import ADD
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
import pickle
import torch
from torch import nn
from model import RawGAT_ST 
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed




def evaluate_accuracy(data_loader, model, device):
    data_loader = DataLoader(dataset, batch_size=5, shuffle=False)
    val_loss = 0.0
    num_total = 0.0
    model.eval()
    true_y = []
    y_pred = []

    
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for batch_x, batch_y in data_loader:
      
        true_y.extend(batch_y.numpy())
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        batch_out = model(batch_x,Freq_aug=False)
        
        batch_loss = criterion(batch_out, batch_y)
        val_loss += (batch_loss.item() * batch_size)
        
    val_loss /= num_total
   
    return val_loss


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    
    fname_list = []
    key_list = []
    score_list = []

    for batch_x, batch_y, batch_meta in data_loader:
        
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x, Freq_aug=False)
        
        batch_score = (batch_out[:, 1]  
                       ).data.cpu().numpy().ravel()     
        

        # add outputs
        fname_list.extend(list(batch_meta[1]))
        key_list.extend(
          ['genuine' if key == 1 else 'fake' for key in list(batch_meta[4])])
        score_list.extend(batch_score.tolist())
        
    with open(save_path, 'w') as fh:
        for f, k, cm in zip(fname_list, key_list, score_list):
            if dataset.is_eval:
                fh.write('{} {} {}\n'.format(f, k, cm))
            else:
                fh.write('{} {}\n'.format(f, cm))
    print('Result saved to {}'.format(save_path))

track = 'track1'
current_model = ''
model_path = 'home/yahmadia/pythonProject1/pythonProject/RawGAT-ST-antispoofing/models/model_WCE_300_5_0.0001/epoch_40.pth'
 
    
evl_set = ADD(data_path=database_path,label_path=label_path,is_train=False, 
                      transform=transform, is_eval=True, sample_size=None, feature=None, track=track)




model.load_state_dict(torch.load(model_path,map_location=device))
true_y, y_pred, num_total = evaluate_accuracy(evl_set, model, device, eval_output,clf)

fpr, tpr, threshold = roc_curve(true_y, y_pred, pos_label=1)
fnr = 1 - tpr
eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
eer = (eer_1 + eer_2) / 2
print(eer)
