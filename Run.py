
import os
import sys

file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
from models.ASTGCRN import ASTGCRN as Network
from BasicTrainer import Trainer
from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.TrainInits import print_model_parameters
from lib.logger import get_logger
from lib.metrics import masked_mae

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
#*************************************************************************#
Mode = 'train'
DEBUG = 'False'
DATASET = 'PEMS04'      # PEMS03 or PEMS04 or PEMS07 or PEMS08 or PEMS07M
MODEL = 'ASTGCRN'

#get configuration
config_file = 'config/{}_{}.conf'.format(DATASET, MODEL)
config = configparser.ConfigParser()
config.read(config_file)

#parser
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--dataset', default=DATASET, type=str)
args.add_argument('--mode', default=Mode, type=str)
# args.add_argument('--device', default=DEVICE, type=str, help='indices of GPUs')
args.add_argument('--debug', default=DEBUG, type=eval)
args.add_argument('--model', default=MODEL, type=str)
args.add_argument('--cuda', default=True, type=bool)
args.add_argument('--cudaid', default=0, type=int)
#data
args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
args.add_argument('--lag', default=config['data']['lag'], type=int)
args.add_argument('--horizon', default=config['data']['horizon'], type=int)
args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
args.add_argument('--tod', default=config['data']['tod'], type=eval)
args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
args.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
args.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)
args.add_argument('--distance_file', default=config['data']['distance_file'], type=str)
args.add_argument('--id_filename', default=config['data']['id_filename'], type=str)
args.add_argument('--type', default=config['data']['type'], type=str)
#model
args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
args.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
args.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
#train
args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
args.add_argument('--seed', default=config['train']['seed'], type=int)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
args.add_argument('--weight_decay', default=config['train']['weight_decay'], type=float)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
args.add_argument('--teacher_forcing', default=False, type=bool)
#args.add_argument('--tf_decay_steps', default=2000, type=int, help='teacher forcing decay steps')
args.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')
args.add_argument('--exp_id', default=config['train']['exp_id'], type=int)
args.add_argument('--runs', default=1, type=int)
#test
args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
#log
args.add_argument('--log_dir', default='./log', type=str)
args.add_argument('--log_step', default=config['log']['log_step'], type=int)
args.add_argument('--plot', default=config['log']['plot'], type=eval)
args.add_argument('--disc', default='Info', type=str)  ####
args = args.parse_args()

# cuda
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cudaid)
# CPU/GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device
print("args:", args.dataset)
# seed
init_seed(args.seed)

log_dir = os.path.join('./log', args.dataset, args.model)
args.log_dir = log_dir
model_dir = os.path.join('./pre-trained', args.dataset, args.model)
args.model_dir = model_dir
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
# Log
logger = get_logger(args)
logger.info(str(args)[10: -1])
logger.info('Experiment log path in: {}'.format(args.log_dir))


def main(args, run):
    #init model
    model = Network(args)
    model = model.to(args.device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    print_model_parameters(model, logger, only_num=False)

    #load dataset
    train_loader, val_loader, test_loader, scaler = get_dataloader(args, normalizer=args.normalizer,
                                                                   tod=args.tod, weather=False, single=False)

    #init loss function, optimizer
    if args.loss_func == 'mask_mae':
        loss = masked_mae
    elif args.loss_func == 'mae':
        loss = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        loss = torch.nn.MSELoss().to(args.device)

    else:
        raise ValueError
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                                 weight_decay=args.weight_decay, amsgrad=False)
    #learning rate decay
    lr_scheduler = None
    if args.lr_decay:
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                            milestones=lr_decay_steps,
                                                            gamma=args.lr_decay_rate)
        #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64)

    #start training
    trainer = Trainer(model, loss, optimizer, train_loader, val_loader, test_loader, scaler,
                      args, logger, run, lr_scheduler=lr_scheduler)
    if args.mode == 'train':
        mae, rmse, mape = trainer.train()
    elif args.mode == 'test':
        model.load_state_dict(torch.load('pre-trained/{}/{}/{}_{}best_model{}.pth'.format(args.dataset, args.model, str(args.exp_id)), args.disc, run))
        print("Load saved model")
        mae, rmse, mape = trainer.test(model, trainer.args, test_loader, scaler, logger)
    else:
        raise ValueError
    return mae, rmse, mape

if __name__ == "__main__":
    mae = []
    mape = []
    rmse = []
    torch.set_num_threads(4)
    for run in range(args.runs):
        logger.info('\n\n-------------The Running is:' + str(run) + '-------------')
        m1, m2, m3 = main(args, run)
        mae.append(m1)
        rmse.append(m2)
        mape.append(m3*100)

    logger.info('\n\nResults for 1 runs\n\n')
    #valid data
    logger.info('test\tMAE\tRMSE\tMAPE')
    logger.info('mean:\t{:.4f}\t{:.4f}\t{:.4f}%'.format(np.mean(mae), np.mean(rmse), np.mean(mape)))
    logger.info('std:\t{:.4f}\t{:.4f}\t{:.4f}%'.format(np.std(mae), np.std(rmse), np.std(mape)))

