import argparse
import torch
import time
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchcontrib
import os
import os.path as osp
import random
import copy
import logging
import pickle

from PIL import Image
from pdb import set_trace as st

from torchvision import transforms

sys.path.append('..')

from clean_dataset.cub200 import CUB200Data
from clean_dataset.mit67 import MIT67Data
from clean_dataset.stanford_40 import Stanford40Data


from model.fe_resnet import resnet18_dropout, resnet34_dropout, resnet50_dropout, resnet101_dropout
from model.fe_resnet import feresnet18, feresnet34, feresnet50, feresnet101

from coverage.my_neuron_coverage import MyNeuronCoverage
from coverage.top_k_coverage import TopKNeuronCoverage
from coverage.strong_neuron_activation_coverage import StrongNeuronActivationCoverage
# from DNNtest.coverage.my_neuron_coverage import MyNeuronCoverage
from coverage.pytorch_wrapper import PyTorchModel

def adv_whitebox(model, loader, args, bounds,):
    model.eval()
    low_bound, up_bound = bounds
    
    loss = nn.CrossEntropyLoss()

    total_ce = 0
    total = 0
    top1 = 0

    total = 0
    top1_clean = 0
    top1_adv = 0
    adv_success = 0
    adv_trial = 0
    for i, (images, label) in enumerate(loader):
        images, label = images.to('cuda'), label.to('cuda')

        total += images.size(0)
        out_clean = model(images)
        _, pred_clean = out_clean.max(dim=1)
        
        adv_images = images.clone().detach()
        if args.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-args.eps, args.eps)
            adv_images[adv_images < low_bound] = low_bound[adv_images < low_bound]
            adv_images[adv_images > up_bound] = up_bound[adv_images > up_bound]
        
        for _ in range(args.pgd_iter):
            adv_images.requires_grad = True
            outputs = model(adv_images)

            # Calculate loss
            if args.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, label)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + args.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-args.eps, max=args.eps)
            adv_images = (images + delta).detach()
            adv_images[adv_images < low_bound] = low_bound[adv_images < low_bound]
            adv_images[adv_images > up_bound] = up_bound[adv_images > up_bound]
            

        out_adv = model(adv_images)
        _, pred_adv = out_adv.max(dim=1)

        clean_correct = pred_clean.eq(label)
        adv_trial += int(clean_correct.sum().item())
        adv_success += int(pred_adv[clean_correct].eq(label[clean_correct]).sum().detach().item())
        top1_clean += int(pred_clean.eq(label).sum().detach().item())
        top1_adv += int(pred_adv.eq(label).sum().detach().item())

        print('{}/{}...'.format(i+1, len(loader)))

    return float(top1_clean)/total*100, float(top1_adv)/total*100, float(adv_trial-adv_success) / adv_trial *100


def get_coverage(args,):
    if args.coverage == "neuron_coverage":
        coverage = MyNeuronCoverage(threshold=args.nc_threshold)
    elif args.coverage == "top_k_coverage":
        coverage = TopKNeuronCoverage(k=10)
    elif args.coverage == "strong_coverage":
        coverage = StrongNeuronActivationCoverage(k=2)
    else:
        raise NotImplementedError
    return coverage

def compute_selected_neuron_value(
    selected_neuron, compressed_intermediate_layer_outputs, global_neuron_id_to_layer_neuron_id
):
    assert isinstance(selected_neuron, list)
    num_input = len(selected_neuron)
    input_neuron_value = []
    for input_id in range(num_input):
        values = []
        for global_id in selected_neuron[input_id]:
            layer_id, layer_neuron_id = global_neuron_id_to_layer_neuron_id[global_id]
            neuron_value = compressed_intermediate_layer_outputs[layer_id][input_id][layer_neuron_id]
            values.append(neuron_value)
        values = torch.stack(values, dim=0).sum()
        input_neuron_value.append(values)
    input_neuron_value = torch.stack(input_neuron_value, dim=0)
    return input_neuron_value

    

def log_coverage(model, loader, args, ):
    model.eval()
    
    measure_model = PyTorchModel(model, intermedia_mode=args.intermedia_mode)
    coverage_metric = get_coverage(args)
    
    log_names = measure_model.full_names
    print(log_names)
    
    intermedia_layers = measure_model._intermediate_layers(model)
    accumulate_coverage = {}
    for name, module in intermedia_layers.items():
        weight = module.weight
        out_shape, in_shape = weight.shape[:2]
        accumulate_coverage[name] = [np.zeros(in_shape), np.zeros(out_shape)] 

    for idx, (images, label) in enumerate(loader):
        images, label = images.to('cuda'), label.to('cuda')
        
        outputs = measure_model.one_sample_intermediate_layer_outputs(
            images, 
            # [coverage_metric.update, coverage_metric.report],
            [coverage_metric.update, ],
        )
        batch_layer_cover = coverage_metric.get()
        for layer_name, (input_coverage, output_coverage) in batch_layer_cover.items():
            # print(layer_idx, accumulate_coverage[layer_idx][0].shape, input_coverage.sum(0).shape)
            accumulate_coverage[layer_name][0] += input_coverage.sum(0)
            accumulate_coverage[layer_name][1] += output_coverage.sum(0)
        
        # if idx > 5:
        #     break
        if idx % 100 == 0:
            print(f"Profiling {idx}/{len(loader)}")

    return accumulate_coverage, log_names


def record_act(self, input, output):
    pass

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default='/data', help='path to the dataset')
    parser.add_argument("--dataset", type=str, default='CUB200Data', help='Target dataset. Currently support: \{SDog120Data, CUB200Data, Stanford40Data, MIT67Data, Flower102Data\}')
    parser.add_argument("--name", type=str, default='test')
    parser.add_argument("--B", type=float, default=0.1, help='Attack budget')
    parser.add_argument("--m", type=float, default=1000, help='Hyper-parameter for task-agnostic attack')
    parser.add_argument("--pgd_iter", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--checkpoint", type=str, default='')
    parser.add_argument("--network", type=str, default='resnet18', help='Network architecture. Currently support: \{resnet18, resnet50, resnet101, mbnetv2\}')
    parser.add_argument("--teacher", default=None)
    parser.add_argument("--output_dir")

    parser.add_argument("--test_num", type=int, default=500)
    parser.add_argument("--num_try_per_sample", type=int, default=10)
    
    parser.add_argument("--sample_queue_length", type=int, default=10)
    parser.add_argument("--nc_threshold", type=float, default=0.5)
    parser.add_argument("--strategy", default="random", choices=["random", "deepxplore", "dlfuzz", "dlfuzzfirst"])
    parser.add_argument("--coverage", default="neuron_coverage")
    parser.add_argument("--k_select_neuron", type=int, default=20)
    parser.add_argument("--intermedia_mode", default="")
    
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--random_start", type=bool, default=True)
    parser.add_argument("--targeted", type=bool, default=False)
    
    
    args = parser.parse_args()
    if args.teacher is None:
        args.teacher = args.network
    return args

def load_student(ckpt, args, num_classes):
    model = eval('{}_dropout'.format(args.network))(
        pretrained=True, 
        dropout=args.dropout, 
        num_classes=num_classes
    ).cuda()
    if not os.path.exists(ckpt):
        raise RuntimeError(f"{args.checkpoint} Not exist")
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded checkpoint from {ckpt}")
    model.eval()
    return model

if __name__ == '__main__':
    seed = 98
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    np.set_printoptions(precision=4)
    
    args = get_args()
    # print(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    args.pid = os.getpid()
    args.log_path = osp.join(args.output_dir, "log.txt")
    if os.path.exists(args.log_path):
        log_lens = len(open(args.log_path, 'r').readlines())
        if log_lens > 5:
            print(f"{args.log_path} exists")
            exit()
    args.info = f"{args.strategy}_{args.coverage}_{args.dataset}_{args.network}"
    
    logging.basicConfig(filename=args.log_path, filemode="w", level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logger = logging.getLogger()
    logger.info(args)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    model_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    test_set = eval(args.dataset)(
        args.datapath, True, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), 
        -1, seed, preload=False
    )
    
    
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=False, drop_last=True)
    
    


    model = eval('{}_dropout'.format(args.network))(pretrained=True, num_classes=test_set.num_classes).cuda()
    
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded student checkpoint from {args.checkpoint}")
    
    accumulate_coverage, log_names = log_coverage(
        model, test_loader, args,
    )
    
    path = osp.join(args.output_dir, "accumulate_coverage.pkl")
    with open(path, "wb") as f:
        pickle.dump(accumulate_coverage, f)
    
    path = osp.join(args.output_dir, "log_module_names.pkl")
    with open(path, "wb") as f:
        pickle.dump(log_names, f)
