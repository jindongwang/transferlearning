import logging
import os
import collections
from espnet.bin.asr_train import get_parser
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.asr.pytorch_backend.asr_init import freeze_modules

from torch.nn.parallel import data_parallel
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch
import numpy as np

import data_load
import random
import json
import sys
from utils import setup_logging, str2bool, dict_average
from utils import load_pretrained_model, torch_save, torch_load
from utils import recognize_and_evaluate

def add_custom_arguments(parser):
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument("--root_path", type=str, required=True, help="Path to the ESPnet features, e.g.: <espnet_path>/egs/an4/asr1/")
    #parser.add_argument("--root_path", type=str, default="/opt/espnet/egs/an4/asr1/")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name to be referred in data_load, e.g.: an4")
    #parser.add_argument("--dataset", type=str, default="an4")
    parser.add_argument("--exp", type=str, default="exp")
    parser.add_argument("--decoding_mode", type=str2bool, default=False, help="if true, then only perform decoding test")
    parser.add_argument("--load_pretrained_model", type=str, default="", nargs="?",
                    help="<model_path>:<load_modules>:<exclude_modules>")
    parser.add_argument("--compute_cer", type=str2bool, default=True)
    parser.add_argument("--compute_cer_interval", type=int, default=1)
    parser.add_argument("--start_eval_errs", type=int, default=70)
    parser.add_argument("--decoding_config", type=str, default=None)
    parser.add_argument(
        "--bpemodel", type=bool, default=True
    )  # Set to true when testing CER/WERs
    parser.add_argument("--dist_train", type=str2bool, default=False)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--result_label", type=str, default=None)
    parser.add_argument("--recog_json", type=str, default=None)
    parser.add_argument("--adam_lr", type=float, default=1e-3)

def train_epoch(dataloader, model, optimizer):
    model.train()
    # acc_lst, loss_lst = [], []
    stats = collections.defaultdict(list)
    for batch_idx, data in enumerate(dataloader):
        fbank, seq_lens, tokens = data
        fbank, seq_lens, tokens = fbank.cuda(), seq_lens.cuda(), tokens.cuda()

        optimizer.zero_grad()
        if args.ngpu <= 1 or args.dist_train:
            loss = model(fbank, seq_lens, tokens).mean() # / self.accum_grad
        else:
            # apex does not support torch.nn.DataParallel
            loss = (
                data_parallel(model, (fbank, seq_lens, tokens), range(args.ngpu)).mean() # / self.accum_grad
            )
        if not hasattr(model, "module"):
            if hasattr(model, "acc") and model.acc is not None:
                stats["acc_lst"].append(model.acc)
                model.acc = None
        else:
            if hasattr(model, "acc") and model.module.acc is not None:
                stats["acc_lst"].append(model.module.acc)
                model.module.acc = None
        loss.backward()
        clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        stats["loss_lst"].append(loss.item())
        logging.warning(f"Training batch: {batch_idx+1}/{len(dataloader)}")
    return dict_average(stats)


def test(dataloader, model, model_path=None):
    if model_path:
        torch_load(model_path, model)
    model.eval()
    stats = collections.defaultdict(list)
    for batch_idx, data in enumerate(dataloader):
        logging.warning(f"Testing batch: {batch_idx+1}/{len(dataloader)}")
        fbank, seq_lens, tokens = data
        fbank, seq_lens, tokens = fbank.cuda(), seq_lens.cuda(), tokens.cuda()
        with torch.no_grad():
            loss = model(fbank, seq_lens, tokens)
        stats["loss_lst"].append(loss.item())
        if not hasattr(model, "module"):
            if model.acc is not None:
                stats["acc_lst"].append(model.acc)
                model.acc = None
        else:
            if model.module.acc is not None:
                stats["acc_lst"].append(model.module.acc)
                model.module.acc = None
    return dict_average(stats)

def train(dataloaders, model, optimizer, save_path):
    train_loader, val_loader, test_loader = dataloaders
    best_loss = float("inf")
    early_stop = 0
    log_json = []
    for epoch in range(args.start_epoch, args.epochs + 1):
        early_stop += 1
        epoch_stats = collections.OrderedDict(epoch=epoch)
        train_stats = train_epoch(train_loader, model, optimizer)
        valid_stats = test(val_loader, model)
        if best_loss > valid_stats["loss"]:  # Save loss best model
            best_loss = valid_stats["loss"]
            torch_save(model, save_path, local_rank=args.local_rank)
            early_stop = 0
            if epoch >= args.start_eval_errs and args.compute_cer \
                and (epoch - args.start_eval_errs) % args.compute_cer_interval == 0 \
                and args.local_rank == 0:
                logging.warning(
                    f"Compute CER/WERs for current best loss ({best_loss}) model"
                )
                dev_errs = recognize_and_evaluate(val_loader, model, args, wer=True)
                logging.warning(
                    f"Dev: cer={dev_errs['cer']['err']}; wer={dev_errs['wer']['err']}"
                )
                test_errs = recognize_and_evaluate(test_loader, model, args, wer=True)
                logging.warning(
                    f"Test: cer={test_errs['cer']['err']}; wer={test_errs['wer']['err']}"
                )
                for key in ["wer", "cer"]:
                    epoch_stats[f"validation/main/{key}"] = dev_errs[key]["err"]
                    epoch_stats[f"test/main/{key}"] = test_errs[key]["err"]

        test_stats = test(test_loader, model)
        logging.warning(
            f"Epoch: {epoch}, Iteration: {epoch * len(train_loader)}, "
            + f"train loss: {train_stats['loss']:.4f}, dev loss: {valid_stats['loss']:.3f}, test loss: {test_stats['loss']:.3f}"
        )

        torch_save(model, f"{args.outdir}/snapshot.ep.{epoch}", optimizer=optimizer, local_rank=args.local_rank)
        for key in test_stats.keys():
            if not key.endswith("_lst"):
                if key in train_stats:
                    epoch_stats[f"main/{key}"] = train_stats[key]
                epoch_stats[f"validation/main/{key}"] = valid_stats[key]
                epoch_stats[f"test/main/{key}"] = test_stats[key]
                
        log_json.append(epoch_stats)
        with open(f"{args.outdir}/log", "w") as f:
            json.dump(log_json, f,
                indent=4,
                ensure_ascii=False,
                separators=(",", ": "),
            )
            logging.warning(f"Log saved at {args.outdir}/log")
            
        if args.patience > 0 and early_stop >= args.patience:
            test_stats = test(test_loader, model, save_path)
            logging.warning(f"=====Early stop! Final test loss: {test_stats['loss']}")
            break

if __name__ == "__main__":
    # 执行该命令运行分布式训练：CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --dist_train true
    setup_logging(verbose=0)  # Should come first before other package import logging
    parser = get_parser()
    add_custom_arguments(parser)

    arg_list = sys.argv[1:] + [
        "--dict", '',
    ]
    if "--config" not in arg_list:
        arg_list += ["--config", "config/train.yaml"]
    if "--outdir" not in arg_list:
        arg_list += ["--outdir", '']

    args, _ = parser.parse_known_args(arg_list)
    # Use all GPUs
    ngpu = torch.cuda.device_count() if args.ngpu is None else args.ngpu
    #ngpu = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
                [str(item) for item in range(ngpu)])
    logging.warning(f"ngpu: {ngpu}")
    
    # set random seed
    logging.info("random seed = %d" % args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_deterministic_pytorch(args)
    torch.cuda.manual_seed(args.seed)
    if ngpu > 1:
        torch.cuda.manual_seed_all(args.seed) # multi-gpu setting
    
    if args.model_module is None:
        model_module = "espnet.nets." + args.backend + "_backend.e2e_asr:E2E"
    else:
        model_module = args.model_module
    model_class = dynamic_import(model_module)

    model_class.add_arguments(parser)
    args = parser.parse_args(arg_list)

    setattr(args, "conf_name", ".".join(os.path.basename(args.config).split(".")[:-1]))
    if not args.outdir:
        args.outdir = f"./outputs/results_{args.dataset}/{args.conf_name}"
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    setattr(args, "model_module", model_module)
    setattr(args, "ngpu", ngpu)
    if args.data_file is not None:
        args.root_path = args.data_file
    
    if args.ngpu > 1:
        if args.opt == "noam" and hasattr(args, "transformer_lr"):
            logging.warning(f"Multi-GPU training: increase transformer lr {args.transformer_lr} --> {args.transformer_lr * np.sqrt(args.ngpu)}")
            args.transformer_lr = args.transformer_lr * np.sqrt(args.ngpu)
        elif args.opt == "adam" and hasattr(args, "adam_lr"):
            logging.warning(f"Multi-GPU training: increase adam lr {args.adam_lr} --> {args.adam_lr * np.sqrt(args.ngpu)}")
            args.adam_lr = args.adam_lr * np.sqrt(args.ngpu)
        
        if args.dist_train:
            torch.distributed.init_process_group(backend="nccl")
            local_rank = torch.distributed.get_rank()
            args.local_rank = local_rank
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            logging.warning(
                "Training batch size is automatically increased (%d -> %d)"
                % (args.batch_size, args.batch_size * args.ngpu)
            )
            args.batch_size *= args.ngpu
    
    if args.accum_grad > 1:
        logging.warning(
                "gradient accumulation is not implemented. batch size is increased (%d -> %d)"
                % (args.batch_size, args.batch_size * args.accum_grad)
            )
        args.batch_size *= args.accum_grad
        args.accum_grad = 1

    dataloaders, in_out_shape = data_load.load_data(args.root_path, args.dataset, args)
    token_list = data_load.load_token_list(
        os.path.join(args.root_path, data_load.data_config[args.dataset]["token"])
    )
    setattr(args, "char_list", token_list)

    model = model_class(in_out_shape[0], in_out_shape[1], args)

    model_conf = args.outdir + "/model.json"
    with open(model_conf, "wb") as f:
        logging.info("writing a model config file to " + model_conf)
        f.write(
            json.dumps(
                (in_out_shape[0], in_out_shape[1], vars(args)),
                indent=4,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf_8")
        )
    
    model.cuda()
    if args.ngpu > 1 and args.dist_train:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                        device_ids=[local_rank],
                                                        output_device=local_rank
                                                        )
            
    if args.freeze_mods:
        model, model_params = freeze_modules(model, args.freeze_mods)
    else:
        model_params = model.parameters()

    # Setup an optimizer
    if args.opt == "adadelta":
        optimizer = torch.optim.Adadelta(
            model_params, rho=0.95, eps=args.eps, weight_decay=args.weight_decay
        )
    elif args.opt == "adam":
        logging.warning(f"Using Adam optimizer with lr={args.adam_lr}")
        optimizer = torch.optim.Adam(model_params, lr=args.adam_lr, weight_decay=args.weight_decay)
    elif args.opt == "noam":
        from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt
        optimizer = get_std_opt(
            model_params, args.adim, args.transformer_warmup_steps, args.transformer_lr
        )

    # Resume from a snapshot
    if args.resume:
        logging.warning("resumed from %s" % args.resume)
        torch_load(args.resume, model, optimizer)
        setattr(args, "start_epoch", int(args.resume.split('.')[-1]) + 1)
    else:
        setattr(args, "start_epoch", 1)
    
    if args.load_pretrained_model:
        model_path, modules_to_load, exclude_modules = args.load_pretrained_model.split(":")
        logging.warning("load pretrained model from %s" % args.load_pretrained_model)
        load_pretrained_model(model=model, model_path=model_path, 
                                modules_to_load=modules_to_load, exclude_modules=exclude_modules)

    logging.warning(
        "Total parameter of the model = "
        + str(sum(p.numel() for p in model.parameters()))
    )
    logging.warning(
        "Trainable parameter of the model = "
        + str(sum(p.numel() for p in filter(lambda x: x.requires_grad, model.parameters())))
    )

    # For CER/WER computing
    if args.bpemodel and "bpemodel" in data_load.data_config[args.dataset]:
        logging.warning("load bpe model")
        args.bpemodel = data_load.load_bpemodel(args.root_path, args.dataset)

    save_path = f"{args.outdir}/model.loss.best"
    if not args.decoding_mode:
        train(dataloaders, model, optimizer, save_path)
                            
    if args.compute_cer and args.local_rank == 0:
        for idx, split in enumerate(["test", "val"], 1):
            logging.warning(f"---------Recognizing {split}----------")
            args.result_label = f"{args.outdir}/{args.dataset}_{split}_recog.json"
            if not data_load.data_config[args.dataset][split]:
                split_path = os.path.join(args.root_path, f"tmp_dev_set_{args.dataset}.json")
            else:
                split_path = data_load.data_config[args.dataset][split]
            args.recog_json = os.path.join(args.root_path, split_path)
            err_dict = recognize_and_evaluate(dataloaders[-idx], model, args, model_path=save_path, wer=True, write_to_json=True)
            logging.warning(f"CER: {err_dict['cer']['err']}")
            logging.warning(f"WER: {err_dict['wer']['err']}")