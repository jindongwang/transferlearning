import torch.nn as nn
import torch
import torch.optim as optim

import os
import argparse
import datetime
import numpy as np

from tqdm import tqdm
from utils import utils
from base.AdaRNN import AdaRNN

import pretty_errors
import dataset.data_process as data_process
import matplotlib.pyplot as plt
from tst import Transformer

def pprint(*text):
    # print with UTC+8 time
    time = '['+str(datetime.datetime.utcnow() +
                   datetime.timedelta(hours=8))[:19]+'] -'
    print(time, *text, flush=True)
    if args.log_file is None:
        return
    with open(args.log_file, 'a') as f:
        print(time, *text, flush=True, file=f)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(args, model, optimizer, src_train_loader,  trg_train_loader, epoch, dist_old=None, weight_mat=None):

    model.train()
    criterion = nn.MSELoss()
    criterion_1 = nn.L1Loss()
    loss_all = []
    loss_1_all = []
    dist_mat = torch.zeros(args.num_layer, args.len_seq).cuda()
    len_loader = np.inf
    
    for data_s, data_t in tqdm(zip(src_train_loader, trg_train_loader), total=min(len(src_train_loader), len(trg_train_loader))):
        optimizer.zero_grad()
        feature_s, _ , label_reg_s = data_s[0].cuda().float(), data_s[1].cuda().long(), data_s[2].cuda().float()
        feature_t, _ , label_reg_t = data_t[0].cuda().float(), data_t[1].cuda().long(), data_t[2].cuda().float()
        if feature_s.shape[0] != feature_t.shape[0]:
            continue
        # feature, label_reg = feature.cuda().float(), label_reg.cuda().float()
        feature_all = torch.cat((feature_s, feature_t), 0)
        pred_all, list_encoding = model(feature_all)
        loss_adapt, dist, weight_mat = model.adapt_encoding_weight(list_encoding, args.loss_type, args.train_type, weight_mat)
        dist_mat = dist_mat + dist
        
        pred_s = pred_all[0:feature_s.size(0)]
        pred_t = pred_all[feature_s.size(0):]
        pred_s = torch.mean(pred_s, dim=1).view(pred_s.shape[0])
        pred_t = torch.mean(pred_t, dim=1).view(pred_t.shape[0])
        loss_s = criterion(pred_s, label_reg_s)
        loss_t = criterion(pred_t, label_reg_t)
        total_loss = loss_s + args.dw * loss_adapt + loss_t
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    loss = np.array(loss_all).mean(axis=0)
    loss_l1 = np.array(loss_1_all).mean()
    if epoch > 0:
        weight_mat = model.update_weight_Boosting(
                    weight_mat, dist_old, dist_mat)
    return loss, loss_l1, weight_mat, dist_mat


def test_epoch(model, test_loader, prefix='Test'):
    model.eval()
    total_loss = 0
    total_loss_1 = 0
    total_loss_r = 0
    correct = 0
    criterion = nn.MSELoss()
    criterion_1 = nn.L1Loss()
    for feature, label, label_reg in tqdm(test_loader, desc=prefix, total=len(test_loader)):
        feature, label_reg = feature.cuda().float(), label_reg.cuda().float()
        with torch.no_grad():
            pred,_ = model(feature)
            pred = torch.mean(pred,dim=1).view(pred.shape[0])
        loss = criterion(pred, label_reg)
        loss_r = torch.sqrt(loss)
        loss_1 = criterion_1(pred, label_reg)
        total_loss += loss.item()
        total_loss_1 += loss_1.item()
        total_loss_r += loss_r.item()
    loss = total_loss / len(test_loader)
    loss_1 = total_loss_1 / len(test_loader)
    loss_r = loss_r / len(test_loader)
    return loss, loss_1, loss_r


def test_epoch_inference(model, test_loader, prefix='Test'):
    model.eval()
    total_loss = 0
    total_loss_1 = 0
    total_loss_r = 0
    correct = 0
    criterion = nn.MSELoss()
    criterion_1 = nn.L1Loss()
    i = 0
    for feature, label, label_reg in tqdm(test_loader, desc=prefix, total=len(test_loader)):
        feature, label_reg = feature.cuda().float(), label_reg.cuda().float()
        with torch.no_grad():
            pred,_ = model(feature)
            pred = torch.mean(pred,dim=1).view(pred.shape[0])
        loss = criterion(pred, label_reg)
        loss_r = torch.sqrt(loss)
        loss_1 = criterion_1(pred, label_reg)
        total_loss += loss.item()
        total_loss_1 += loss_1.item()
        total_loss_r += loss_r.item()
        if i == 0:
            label_list = label_reg.cpu().numpy()
            predict_list = pred.cpu().numpy()
        else:
            label_list = np.hstack((label_list, label_reg.cpu().numpy()))
            predict_list = np.hstack((predict_list, pred.cpu().numpy()))

        i = i + 1
    loss = total_loss / len(test_loader)
    loss_1 = total_loss_1 / len(test_loader)
    loss_r = total_loss_r / len(test_loader)
    return loss, loss_1, loss_r, label_list, predict_list


def inference(model, data_loader):
    loss, loss_1, loss_r, label_list, predict_list = test_epoch_inference(
        model, data_loader, prefix='Inference')
    return loss, loss_1, loss_r, label_list, predict_list


def inference_all(output_path, model, model_path, loaders):
    pprint('inference...')
    loss_list = []
    loss_l1_list = []
    loss_r_list = []
    model.load_state_dict(torch.load(model_path))
    i = 0
    list_name = ['train', 'valid', 'test']
    for loader in loaders:
        loss, loss_1, loss_r, label_list, predict_list = inference(
            model, loader)
        loss_list.append(loss)
        loss_l1_list.append(loss_1)
        loss_r_list.append(loss_r)
        i = i + 1
    return loss_list, loss_l1_list, loss_r_list




def main_transfer(args):
    print(args)

    output_path = args.outdir + '_' + args.station + '_' + args.model_name + '_weather_' + \
        args.loss_type + '_' + str(args.pre_epoch) + \
        '_'  + '_' + str(args.lr) + "_" + str(args.train_type) + "-layer-num-" + str(args.num_layer) + "-hidden-" + str(args.hidden_dim) + "-num_head-" + str(args.num_head) + "dw-" + str(args.dw)
        # "-hidden" + str(args.hidden_dim) + "-head" + str(args.num_head)
    save_model_name = args.model_name + '_' + args.loss_type + \
        '_' + str(args.dw) + '_' + str(args.lr) + '.pkl'
    utils.dir_exist(output_path)
    pprint('create loaders...')

    source_loader,  target_loader, valid_loader, test_loader = data_process.load_weather_data(
        args.data_path, args.batch_size, args.station)

    args.log_file = os.path.join(output_path, 'run.log')
    pprint('create model...')
    ######
    # Model parameters
    d_model = args.hidden_dim #32  Lattent dim
    q = 8  # Query size
    v = 8  # Value size
    h = args.num_head #4   Number of heads
    N = args.num_layer  # Number of encoder and decoder to stack
    attention_size = 12  # Attention window size
    pe = "regular"  # Positional encoding
    chunk_mode = None
    d_input = 6  # From dataset
    d_output = 1  # From dataset

    model = Transformer(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size, chunk_mode=chunk_mode, pe=pe, pe_period =24).cuda()


    #####
    num_model = count_parameters(model)
    print('#model params:', num_model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
   
    best_score = np.inf
    best_epoch, stop_round = 0, 0
    weight_mat, dist_mat = None, None

    for epoch in range(args.n_epochs):
        pprint('Epoch:', epoch)
        pprint('training...')

        loss, loss1, weight_mat, dist_mat = train_epoch(
                args, model, optimizer, source_loader,  target_loader, epoch, dist_mat, weight_mat)

        pprint('evaluating...')
        train_loss, train_loss_l1, train_loss_r = test_epoch(
            model, source_loader, prefix='Train')
        val_loss, val_loss_l1, val_loss_r = test_epoch(
            model, valid_loader, prefix='Valid')
        test_loss, test_loss_l1, test_loss_r = test_epoch(
            model, test_loader, prefix='Test')

        pprint('valid %.6f, test %.6f' %
               (val_loss_l1, test_loss_l1))

        if val_loss < best_score:
            best_score = val_loss
            stop_round = 0
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(
                output_path, save_model_name))
        else:
            stop_round += 1
            if stop_round >= args.early_stop:
                pprint('early stop')
                break

    pprint('best val score:', best_score, '@', best_epoch)

    loaders = source_loader, valid_loader, test_loader
    loss_list, loss_l1_list, loss_r_list = inference_all(output_path, model, os.path.join(
        output_path, save_model_name), loaders)
    pprint('MSE: train %.6f, valid %.6f, test %.6f' %
           (loss_list[0], loss_list[1], loss_list[2]))
    pprint('L1:  train %.6f, valid %.6f, test %.6f' %
           (loss_l1_list[0], loss_l1_list[1], loss_l1_list[2]))
    pprint('RMSE: train %.6f, valid %.6f, test %.6f' %
           (loss_r_list[0], loss_r_list[1], loss_r_list[2]))
    pprint('Finished.')


def get_args():

    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_name', default='adapt_tf')
    parser.add_argument('--d_feat', type=int, default=6)

    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--class_num', type=int, default=1)
    parser.add_argument('--pre_epoch', type=int, default=40)  # 25
    parser.add_argument('--num_layer', type=int, default=1)  # 25

    # training
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--early_stop', type=int, default=40)
    parser.add_argument('--smooth_steps', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=36)
    parser.add_argument('--dw', type=float, default=0.5)
    parser.add_argument('--loss_type', type=str, default='cosine')
    parser.add_argument('--train_type', type=str, default='all')
    parser.add_argument('--station', type=str, default='Tiantan')
    parser.add_argument('--data_mode', type=str,
                        default='pre_process')
    parser.add_argument('--num_domain', type=int, default=2)
    parser.add_argument('--len_seq', type=int, default=24)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_head', type=int, default=8)

    # other
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--data_path', default="/root/Messi_du/adarnn/")
    parser.add_argument('--outdir', default='./outputs')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--log_file', type=str, default='run.log')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--len_win', type=int, default=0)
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    main_transfer(args)
