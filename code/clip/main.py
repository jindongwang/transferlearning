import argparse
import numpy as np
import os
import pretty_errors
import torch
import torch.optim as optim

from clip_model import ClipModel
from data.data_loader import ImageTextData
from utils import gather_res, get_logger, set_gpu, set_seed

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--mode', type=str, choices=['zs', 'fe', 'ft'], default='ft') # zeroshot, feature extraction, fine-tuning
    parser.add_argument('--dataset', type=int, default=16) 
    parser.add_argument('--model', type=int, default=2)  # -1 for sweep
    parser.add_argument('--root', type=str, default='/data/jindwang/')  # root path of dataset
    parser.add_argument('--log_file', type=str, default='log.txt')
    parser.add_argument('--seed', type=int, default=42) # random seed
    parser.add_argument('--result', action='store_true')  # if you want to sweep results statistics
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--nepoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.98)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    ## the following test data and test batchsize are only used for fine-tuning mode
    parser.add_argument('--test_batchsize', type=int, default=1024)
    parser.add_argument('--test_data', type=int, default=17)
    args = parser.parse_args()
    return args

def main(args):
    model, dataset = args.model, args.dataset
    model_name = ClipModel.get_model_name_by_index(model)
    dataset_name = ImageTextData.get_data_name_by_index(dataset)
    args.log_file = os.getcwd() + '/log/{}_{}_{}.txt'.format(args.mode, model_name, dataset_name)
    logger = get_logger(args.log_file, args.log_file)
    logger.info(args)
    
    clip = ClipModel(model,logger=logger)
    logger.info(f'Clip model {model_name} loaded')

    itdata = ImageTextData(dataset, root=args.root, preprocess=clip.preprocess)
    train_loader = torch.utils.data.DataLoader(itdata, batch_size=args.batchsize, shuffle=True)
    logger.info(f'Dataset {dataset_name} loaded')

    if args.mode == 'zs':  # zeroshot
        acc, res = clip.evaluate(train_loader)
        logger.info('Results: {}'.format(res))
        logger.info('Accuracy: {:.2f}%'.format(acc * 100))
    elif args.mode == 'fe': # feature extraction
        res = clip.feature_extraction(train_loader)
        logger.info('Feature extracted!')
        if not os.path.exists('feat'):
            os.makedirs('feat')
        feat_file = 'feat/{}_{}_{}.csv'.format(args.mode, model_name, dataset_name)
        np.savetxt(feat_file, res, fmt='%.4f')
    elif args.mode == 'ft': # fine-tuning
        test_data = ImageTextData(args.test_data, root=args.root, preprocess=clip.preprocess)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batchsize, shuffle=False, drop_last=False)
        optimizer = optim.Adam(clip.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay)
        best_acc = clip.finetune(train_loader, test_loader, optimizer, args.nepoch, save_path='/home/jindwang/mine/clipood/model/{}_{}_{}.pt'.format(args.mode, model_name, dataset_name))
        logger.info('Accuracy: {:.2f}%'.format(best_acc * 100))
    else:
        raise NotImplementedError
        

def sweep_index(model=-1, data=-1):
    if model == -1 and data == -1:
        m_sweep_index = range(len(ClipModel.CLIP_MODELS))
        d_sweep_index = range(len(ImageTextData._DATA_FOLDER))
    elif model == -1 and data != -1:
        m_sweep_index = range(len(ClipModel.CLIP_MODELS))
        d_sweep_index = range(data, data + 1)
    elif data == -1 and model != -1:
        m_sweep_index = range(model, model + 1)
        d_sweep_index = range(len(ImageTextData._DATA_FOLDER))
    else:
        m_sweep_index = range(model, model + 1)
        d_sweep_index = range(data, data + 1)
    return m_sweep_index, d_sweep_index


def sweep(model=-1, data=-1):
    m_sweep_index, d_sweep_index = sweep_index(model, data)
    if args.result:
        model_name_lst = [ClipModel.get_model_name_by_index(i) for i in m_sweep_index]
        data_name_lst = [ImageTextData.get_data_name_by_index(i) for i in d_sweep_index]
        res = gather_res(model_name_lst, data_name_lst)
        for line in res:
            print(line)
    else:
        for model in m_sweep_index:
            for data in d_sweep_index:
                args.model = model
                args.dataset = data
                main(args)
    

if __name__ == '__main__':
    args = get_args()
    set_gpu(args.gpu)
    set_seed(args.seed)
    sweep(args.model, args.dataset)
    