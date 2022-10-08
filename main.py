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


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def train_epoch(data_loader, model, lr,optimizer, device):
    running_loss = 0
    num_total = 0.0
    model.train()

    # set objective (Loss) functions --> WCE
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    for batch_x, batch_y, batch_meta in data_loader:
        
        batch_size = batch_x.size(0)

        num_total += batch_size
        
        batch_x = batch_x.to(device)
       
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        
        batch_out = model(batch_x,Freq_aug=True)
        
        batch_loss = criterion(batch_out, batch_y)
        
        running_loss += (batch_loss.item() * batch_size)
       
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
       
    running_loss /= num_total
    
    return running_loss




if __name__ == '__main__':
    parser = argparse.ArgumentParser('ADD RawGAT-ST model')
    # Dataset


    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='WCE',help='Weighted Cross Entropy Loss ')

    # model
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments
    parser.add_argument('--track', type=str, default=None,choices=None, help='')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--is_eval', action='store_true', default=False,help='eval database')
    parser.add_argument('--eval_part', type=int, default=0)
    parser.add_argument('--features', type=str, default='Raw_GAT')

    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True, 
                        help='use cudnn-deterministic? (default true)')    
    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False, 
                        help='use cudnn-benchmark? (default false)') 
    

    dir_yaml = os.path.splitext('model_config_RawGAT_ST')[0] + '.yaml'

    with open(dir_yaml, 'r') as f_yaml:
            parser1 = yaml.safe_load(f_yaml)
    
    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()

    #make experiment reproducible
    set_random_seed(args.seed, args)


    #define model saving path
    model_tag = 'model_{}_{}_{}_{}'.format(
         args.loss, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('/home/yahmadia/dataset_add/ADD_Data/models', model_tag)
    
    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    
    transforms = transforms.Compose([
        lambda x: pad(x),
        lambda x: Tensor(x)
        
    ])


    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))

    # validation Dataloader
    dev_data = "/home/yahmadia/dataset_add/ADD_Data/ADD_dev"
    dev_label= "/home/yahmadia/dataset_add/ADD_Data/ADD_dev_label.txt"
    dev_set = ADD(database_path=dev_data,label_path=dev_label,is_train=False,
                                    transform=transforms,
                                    feature=args.features, is_eval=args.is_eval, eval_part=args.eval_part)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True)
    
    
    #model 
    model = RawGAT_ST(parser1['model'], device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model =(model).to(device)

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    
    
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path,map_location=device))
        print('Model loaded : {}'.format(args.model_path))


    # Inference
    if args.eval:
        assert args.eval_output is not None, 'You must provide an output path'
        assert args.model_path is not None, 'You must provide model checkpoint'
        produce_evaluation_file(dev_set, model, device, args.eval_output)
        sys.exit(0)

    # Training Dataloader
    data_path = "/home/yahmadia/dataset_add/ADD_Data/ADD_train"
    label_path = "/home/yahmadia/dataset_add/ADD_Data/ADD_train_label.txt"
    data = torch.load('/home/yahmadia/dataset_add/ADD_Data/trainSBt.pkl') #load pickled file from feature extraction here
    train_set = ADD(data,label_path,is_train=True, transform=transforms)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True)

    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    for epoch in range(num_epochs):
        
        running_loss = train_epoch(train_loader,model, args.lr,optimizer, device)
        val_loss = evaluate_accuracy(dev_loader, model, device)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        print('\n{} - {} - {} '.format(epoch,
                                                   running_loss,val_loss))
        torch.save(model.state_dict(), os.path.join(
            model_save_path, 'epoch_{}.pth'.format(epoch)))
