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
from utils import load_pretrained_model, load_head_from_pretrained_model, torch_save, torch_load
from utils import recognize_and_evaluate
import math
from e2e_asr_adaptertransformer import E2E as E2EAdapterTransformer
import matplotlib.pyplot as plt
import copy

def add_custom_arguments(parser):
    # EasyEspnet arguments
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument("--root_path", type=str, required=True, help="Path to the ESPnet features, e.g.: <espnet_path>/egs/commonvoice/asr1/")
    #parser.add_argument("--root_path", type=str, default="/opt/espnet/egs/an4/asr1/")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name to be referred in data_load, e.g.: an4")
    #parser.add_argument("--dataset", type=str, default="an4")
    parser.add_argument("--exp", type=str, default="exp")
    parser.add_argument("--decoding_mode", type=str2bool, default=False, help="if true, then only perform decoding test")
    parser.add_argument("--load_pretrained_model", type=str, default="", nargs="?",
                    help="<model_path>:<load_modules>:<exclude_modules>")
    parser.add_argument("--compute_cer", type=str2bool, default=True)
    parser.add_argument("--decoding_config", type=str, default=None)
    parser.add_argument(
        "--bpemodel", type=bool, default=True
    )  # Set to true when testing CER/WERs
    parser.add_argument("--dist_train", type=str2bool, default=False)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--result_label", type=str, default=None)
    parser.add_argument("--recog_json", type=str, default=None)
    parser.add_argument("--adam_lr", type=float, default=1e-3)
    
    # Adapter-related
    parser.add_argument("--use_adapters", type=str2bool, default=True, help="whether to inject adapters into the model")
    parser.add_argument("--shared_adapter", type=str, default=None, help="Share one adapter across all languages")
    parser.add_argument('--adapter_train_languages', type=str, default=None,
                                help="Splitted by _, e.g., en_zh_jp")
    parser.add_argument("--train_adapter_with_head", type=str2bool, 
                        default=False, help="whether to train adapter with language-specific head jointly")
    # SimAdapter-related
    parser.add_argument("--sim_adapter", type=str2bool, default=False)
    parser.add_argument("--fusion_languages", type=str, default=None)
    parser.add_argument("--guide_loss_weight", type=float, default=0.1)
    parser.add_argument("--guide_loss_weight_decay_steps", type=int, default=0)

    # MAML-related
    parser.add_argument("--meta_train", type=str2bool, default=False)
    parser.add_argument("--meta_lr", type=float, default=1.0, help="used for meta-learning outer step")
    
    # Others
    parser.add_argument("--load_head_from_pretrained_model", type=str, default="", nargs="?",
                    help="<model_path>")

def train_epoch(dataloader, model, optimizer, epoch=None):
    model.train()
    stats = collections.defaultdict(list)
    for batch_idx, data in enumerate(dataloader):
        fbank, seq_lens, tokens, language = data
        fbank, seq_lens, tokens = fbank.cuda(), seq_lens.cuda(), tokens.cuda()
        if isinstance(optimizer, dict):
            optimizer[language].zero_grad()
        else:
            optimizer.zero_grad()
        model.zero_grad()
        if args.ngpu <= 1 or args.dist_train:
            ctc_att_loss, sim_adapter_guide_loss = model(fbank, seq_lens, tokens, language)# .mean() # / self.accum_grad
        else:
            # apex does not support torch.nn.DataParallel
            ctc_att_loss, sim_adapter_guide_loss = (
                data_parallel(model, (fbank, seq_lens, tokens, language), range(args.ngpu))# .mean() # / self.accum_grad
            )
        loss = ctc_att_loss.mean()
        if args.sim_adapter:
            if hasattr(model, "module"):
                sim_adapter_reg_loss = model.module.get_fusion_regularization_loss()
            else:
                sim_adapter_reg_loss = model.get_fusion_regularization_loss()
            loss = loss + sim_adapter_reg_loss
            stats["sim_adapter_reg_loss_lst"].append(sim_adapter_reg_loss.item())
            if args.guide_loss_weight > 0:
                if args.guide_loss_weight_decay_steps > 0:
                    n_batch = len(dataloader)
                    current_iter = float(batch_idx + (epoch - 1) * n_batch)
                    frac_done = 1.0 * float(current_iter) / args.guide_loss_weight_decay_steps
                    current_weight = args.guide_loss_weight * max(0., 1. - frac_done)
                    stats["sim_adapter_guide_loss_weight"] = current_weight
                else:
                    current_weight = args.guide_loss_weight
                sim_adapter_guide_loss = sim_adapter_guide_loss.mean()
                loss = loss + current_weight * sim_adapter_guide_loss
                stats["sim_adapter_guide_loss_lst"].append(sim_adapter_guide_loss.item())

        if not hasattr(model, "module"):
            if hasattr(model, "acc") and model.acc is not None:
                stats["acc_lst"].append(model.acc)
                model.acc = None
        else:
            if hasattr(model, "acc") and model.module.acc is not None:
                stats["acc_lst"].append(model.module.acc)
                model.module.acc = None
        loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), args.grad_clip)
        if math.isnan(grad_norm):
            logging.warning("grad norm is nan. Do not update model.")
        else:
            if isinstance(optimizer, dict):
                optimizer[language].step()
            else:
                optimizer.step()
            stats["loss_lst"].append(loss.item())
        logging.warning(f"Training batch: {batch_idx+1}/{len(dataloader)}")
    return dict_average(stats)

def train_maml_epoch(dataloader, model, optimizer, epoch=None):
    model.train()
    stats = collections.defaultdict(list)
    
    for batch_idx, total_batches in enumerate(dataloader):
        i = batch_idx # current iteration in epoch
        len_dataloader = len(dataloader) # total iteration in epoch
        meta_iters = args.epochs * len_dataloader
        current_iter = float(i + (epoch - 1) * len_dataloader)
        frac_done = 1.0 * float(current_iter) / meta_iters
        current_outerstepsize = args.meta_lr * (1. - frac_done)

        weights_original = copy.deepcopy(model.state_dict())
        new_weights = []
        for total_batch in total_batches: # Iter by languages
            in_batch_size = int(total_batch[0].shape[0] / 2) # In-language batch size
            for meta_step in range(2): # Meta-train & meta-valid
                if meta_step == 1:
                    last_backup = copy.deepcopy(model.state_dict())
                else:
                    last_backup = None
                batch = list(copy.deepcopy(total_batch))
                for i_batch in range(len(batch)-1):
                    batch[i_batch] = batch[i_batch][meta_step*in_batch_size:(1+meta_step)*in_batch_size]
                batch = tuple(batch)
                
                fbank, seq_lens, tokens, language = batch
                fbank, seq_lens, tokens = fbank.cuda(), seq_lens.cuda(), tokens.cuda()
                optimizer.zero_grad()
                model.zero_grad()
                if args.ngpu <= 1 or args.dist_train:
                    loss = model(fbank, seq_lens, tokens, language).mean() # / self.accum_grad
                else:
                    # apex does not support torch.nn.DataParallel
                    loss = (
                        data_parallel(model, (fbank, seq_lens, tokens, language), range(args.ngpu)).mean() # / self.accum_grad
                    )
                # print(loss.item())
                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), args.grad_clip)
                if math.isnan(grad_norm):
                    logging.warning("grad norm is nan. Do not update model.")
                else:
                    optimizer.step()
                
                if meta_step == 1: # Record meta valid
                    if not hasattr(model, "module"):
                        if hasattr(model, "acc") and model.acc is not None:
                            stats["acc_lst"].append(model.acc)
                            model.acc = None
                    else:
                        if hasattr(model, "acc") and model.module.acc is not None:
                            stats["acc_lst"].append(model.module.acc)
                            model.module.acc = None
                    stats["loss_lst"].append(loss.item())
                    stats["meta_lr"] = current_outerstepsize
                    optimizer.zero_grad()

            for name in last_backup:
                # Compute meta-gradient
                last_backup[name] = model.state_dict()[name] - last_backup[name]
            # Change back to the original parameters for the new language
            new_weights.append(last_backup) # updates.append(subtract_vars(self._model_state.export_variables(), last_backup))
            model.load_state_dict({ name: weights_original[name] for name in weights_original})
        
        ws = len(new_weights)
        # Compute average meta-gradient
        fweights = { name : new_weights[0][name]/float(ws) for name in new_weights[0] }
        for i in range(1, ws):
            for name in new_weights[i]:
                fweights[name] = fweights[name] + new_weights[i][name] / float(ws)
        model.load_state_dict({name : weights_original[name] + (fweights[name] * current_outerstepsize) for name in weights_original})

        logging.warning(f"Training batch: {batch_idx+1}/{len(dataloader)}")
    return dict_average(stats)

def test(epoch, dataloader, model, model_path=None, language=None, visualize_sim_adapter=False):
    if model_path:
        torch_load(model_path, model)
    orig_model = None
    if hasattr(model, "module"):
        orig_model = model
        model = model.module
    model.eval()
    stats = collections.defaultdict(list)
    for batch_idx, data in enumerate(dataloader):
        logging.warning(f"Testing batch: {batch_idx+1}/{len(dataloader)}")
        if len(data) == 4:
            fbank, seq_lens, tokens, language = data
        else:
            assert language is not None
            fbank, seq_lens, tokens = data
        fbank, seq_lens, tokens = fbank.cuda(), seq_lens.cuda(), tokens.cuda()
        with torch.no_grad():
            loss = model(fbank, seq_lens, tokens, language)
        
        if visualize_sim_adapter:
            atts = model.calculate_sim_adapter_attentions(fbank, seq_lens, tokens, language)
            init_mat = lambda: np.zeros((len(model.fusion_languages),))
            avg_atts = collections.defaultdict(init_mat)
            count = collections.defaultdict(int)
            for key in atts.keys():
                avg_atts[key] = avg_atts[key] + atts[key].sum(axis=(0, 1))
                count[key] = count[key] + atts[key].shape[0] * atts[key].shape[1]
        stats["loss_lst"].append(loss.item())
        if not hasattr(model, "module"):
            if model.acc is not None:
                stats["acc_lst"].append(model.acc)
                model.acc = None
        else:
            if model.module.acc is not None:
                stats["acc_lst"].append(model.module.acc)
                model.module.acc = None
    if visualize_sim_adapter:
        for key in avg_atts.keys():
            avg_atts[key] = avg_atts[key] / count[key]
            logging.warning(f"Attention scores of {key}: {avg_atts[key]}")
        fig = plt.figure(figsize=(16, 8))
        ax = fig.subplots()
        atts, labels = [], []
        for key in avg_atts.keys():
            atts.append(avg_atts[key])
            labels.append(key)
        atts = np.stack(atts)
        tick_marks = np.arange(len(labels))
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(labels)
        x_labels = list(sorted(model.fusion_languages))
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels)
        ax.imshow(atts)
        import itertools
        for i, j in itertools.product(range(atts.shape[0]), range(atts.shape[1])):
            plt.text(j, i, "{:0.2f}".format(atts[i, j]),
                    horizontalalignment="center",
                    color="white")
        fig.tight_layout()
        fig.savefig(f"{args.outdir}/att_{epoch}.png")
        plt.close()
    if orig_model is not None:
        model = orig_model
    return dict_average(stats)

def train(dataloaders, model, optimizer, save_path):
    train_loader, val_loader, test_loader = dataloaders
    best_loss = float("inf")
    early_stop = 0
    log_json = []
    for epoch in range(args.start_epoch, args.epochs + 1):
        early_stop += 1
        epoch_stats = collections.OrderedDict(epoch=epoch)
        train_stats = train_epoch(train_loader, model, optimizer, epoch)
        valid_stats = test(f"val_{epoch}", val_loader, model, visualize_sim_adapter=args.sim_adapter)
        if best_loss > valid_stats["loss"]:  # Save loss best model
            best_loss = valid_stats["loss"]
            torch_save(model, save_path)
            early_stop = 0

        test_stats = test(f"test_{epoch}", test_loader, model)
        logging.warning(
            f"Epoch: {epoch}, Iteration: {epoch * len(train_loader)}, "
            + f"train loss: {train_stats['loss']:.4f}, dev loss: {valid_stats['loss']:.3f}, test loss: {test_stats['loss']:.3f}"
        )

        torch_save(model, f"{args.outdir}/snapshot.ep.{epoch}", optimizer=optimizer)
        for key in sorted(list(set(list(train_stats.keys()) + list(test_stats.keys())))):
            if not key.endswith("_lst"):
                if key in train_stats:
                    epoch_stats[f"main/{key}"] = train_stats[key]
                if key in valid_stats:
                    epoch_stats[f"validation/main/{key}"] = valid_stats[key]
                if key in test_stats:
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
            test_stats = test("test_best", test_loader, model, save_path)
            logging.warning(f"=====Early stop! Final best test loss: {test_stats['loss']}")
            break

if __name__ == "__main__":
    # 执行该命令运行4 GPU训练：CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=2 train.py
    setup_logging(verbose=0)  # Should come first before other package import logging
    parser = get_parser()
    add_custom_arguments(parser)

    arg_list = sys.argv[1:] + [
        "--dict", '',
        #"--dataset", "_".join("cv mt cnh ky dv sl el lv fyNL sah".split()),
    ]
    if "--config" not in arg_list:
        arg_list += ["--config", "config/train.yaml"]
    if "--outdir" not in arg_list:
        arg_list += ["--outdir", '']

    args, _ = parser.parse_known_args(arg_list)
    # Use all GPUs
    ngpu = torch.cuda.device_count() if args.ngpu is None else args.ngpu
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

    model_module =  "e2e_asr_adaptertransformer:E2E"
    model_class = E2EAdapterTransformer
    model_class.add_arguments(parser)
    args = parser.parse_args(arg_list)

    setattr(args, "conf_name", ".".join(os.path.basename(args.config).split(".")[:-1]))
    if not args.outdir:
        args.outdir = f"./outputs/results_{args.dataset}/{args.conf_name}"
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    setattr(args, "ngpu", ngpu)
    if args.data_file is not None:
        args.root_path = args.data_file
    
    if args.ngpu > 1:

        if args.opt == "noam" and hasattr(args, "transformer_lr"):
            logging.warning(f"Multi-GPU training: increase transformer lr {args.transformer_lr} --> {args.transformer_lr * np.sqrt(args.ngpu)}")
            args.transformer_lr = args.transformer_lr * np.sqrt(args.ngpu)
        elif (args.opt == "adam" or args.meta_train) and hasattr(args, "adam_lr"):
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

    dataloaders = {}
    token_dict = {}
    idim, odim_dict = None, {}
    args.dataset = args.dataset.split("_")
    languages = args.dataset
    data_load_languages = languages

    if args.adapter_train_languages is not None:
        args.adapter_train_languages = args.adapter_train_languages.split("_")
        data_load_languages = args.adapter_train_languages
    else:
        logging.warning("adapter_train_languages is None, will use all datasets for training")
        args.adapter_train_languages = args.dataset
    
    dataloaders, (idim, odim_dict) = data_load.load_multilingual_data(args.root_path, args.dataset, args, data_load_languages)
    for idx, data_set in enumerate(args.dataset):
        if languages[idx] not in data_load_languages:
            continue
        token_dict[languages[idx]] = data_load.load_token_list(
                os.path.join(args.root_path, data_load.data_config[data_set]["token"])
        )
    setattr(args, "char_list", token_dict)
    model = model_class(idim, odim_dict, args, languages)

    model_conf = args.outdir + "/model.json"
    with open(model_conf, "wb") as f:
        logging.info("writing a model config file to " + model_conf)
        f.write(
            json.dumps(
                (idim, odim_dict, vars(args)),
                indent=4,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf_8")
        )

    model.cuda()
    if args.freeze_mods:
        model, model_params = freeze_modules(model, args.freeze_mods)
    else:
        model_params = model.parameters()

    logging.warning("Trainable parameters:")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            logging.warning(name)
            
    # Setup an optimizer
    if args.meta_train:
        logging.warning(f"Use Adam optimizer with lr={args.adam_lr}, beta0=0 for meta-training inner step")
        optimizer = torch.optim.Adam(model_params, lr=args.adam_lr, betas=(0, 0.999), weight_decay=args.weight_decay)
    else:
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
    
    if len(args.adapter_train_languages) > 1 and not args.sim_adapter and not args.shared_adapter:
        model_params = collections.defaultdict(list)
        optimizer = {}
        for lang in args.adapter_train_languages:
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and lang in name.split("."):
                    model_params[lang].append(parameter)
            logging.warning(f"Number of trainable parameters for language {lang} " + str(sum(p.numel() for p in model_params[lang])))
            optimizer[lang] = torch.optim.Adam(model_params[lang], lr=args.adam_lr, weight_decay=args.weight_decay)
    
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
    if args.load_head_from_pretrained_model:
        logging.warning("load pretrained model head from %s" % args.load_head_from_pretrained_model)
        load_head_from_pretrained_model(model=model, model_path=args.load_head_from_pretrained_model)
        
    logging.warning(
        "Total parameter of the model = "
        + str(sum(p.numel() for p in model.parameters()))
    )
    logging.warning(
        "Trainable parameter of the model = "
        + str(sum(p.numel() for p in filter(lambda x: x.requires_grad, model.parameters())))
    )

    if args.ngpu > 1 and args.dist_train:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                        device_ids=[local_rank],
                                                        output_device=local_rank,
                                                        find_unused_parameters=True,
                                                        )
    
    save_path = f"{args.outdir}/model.loss.best"
    if args.meta_train:
        train_epoch = train_maml_epoch
    if not args.decoding_mode:
        train(dataloaders, model, optimizer, save_path)
    if args.compute_cer and args.local_rank == 0:
        # For CER/WER computing
        for idx, dataset in enumerate(args.dataset):
            language = languages[idx]
            if args.adapter_train_languages and not (language in args.adapter_train_languages):
                continue
            if args.bpemodel and "bpemodel" in data_load.data_config[dataset]:
                logging.warning(f"load bpe model for dataset {dataset}")
                args.bpemodel = data_load.load_bpemodel(args.root_path, dataset)
            dataloaders, _ = data_load.load_data(args.root_path, dataset, args)
            splits = ["test", "val"]
            for split in splits:
                logging.warning(f"---------Recognizing {dataset} {split}----------")
                args.result_label = f"{args.outdir}/{dataset}_{split}_recog.json"
                if not data_load.data_config[dataset][split]:
                    split_path = os.path.join(args.root_path, f"{args.root_path}/tmp_dev_set_{dataset}.json")
                else:
                    split_path = data_load.data_config[dataset][split]
                args.recog_json = os.path.join(args.root_path, split_path)
                idx = ["train", "val", "test"].index(split)
                test_stats = test(f"{split}_best", dataloaders[idx],
                                model, 
                                save_path, 
                                language=language, 
                                visualize_sim_adapter=args.sim_adapter)
                logging.warning(f"Loss: {test_stats['loss']}")
                err_dict = recognize_and_evaluate(dataloaders[idx], model, args, language, 
                                    model_path=save_path, wer=True, write_to_json=True)
                logging.warning(f"CER: {err_dict['cer']['err']}")
                logging.warning(f"WER: {err_dict['wer']['err']}")